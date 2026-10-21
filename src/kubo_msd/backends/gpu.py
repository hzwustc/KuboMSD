"""Optional CuPy backend with shared matrix identity and explicit transfer accounting."""
import time
from ..errors import BackendUnavailable,NumericalError

class GPUBackend:
    name='gpu'
    def __init__(self):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as sparse
            if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('no CUDA devices')
            probe=cp.asarray([1+1j],dtype=cp.complex128)
            if float((probe.conj()*probe).real.sum())!=2.: raise RuntimeError('complex128 GPU kernel smoke check failed')
            cp.cuda.get_current_stream().synchronize()
        except Exception as exc: raise BackendUnavailable(f'GPU requested but CuPy/CUDA unavailable: {exc}; check CUDA_PATH and LD_LIBRARY_PATH select the same compatible toolkit') from exc
        self.xp=cp;self.sparse=sparse;self.upload_bytes=0;self.download_bytes=0;self.device_copy_bytes=0;self.transfer_seconds=0.;self.sync_count=0
        self.device_id=cp.cuda.runtime.getDevice();self.transfers=[];self.stages={};self._matrix_identities={};self.device_peak_used_bytes=0

    def _record(self,direction,a,name,phase,purpose,identity=None):
        size=int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes) if self.sparse.issparse(a) else int(a.nbytes)
        self.transfers.append(dict(direction=direction,object=name,shape=list(a.shape),dtype=str(a.dtype),bytes=size,phase=phase,purpose=purpose,device_id=self.device_id,matrix_identity=identity))
        if direction=='H2D':self.upload_bytes+=size
        elif direction=='D2H':self.download_bytes+=size
        elif direction=='D2D':self.device_copy_bytes+=size
        self.device_peak_used_bytes=max(self.device_peak_used_bytes,self.xp.get_default_memory_pool().used_bytes())

    def asarray(self,a,*,name='array',phase='transport',purpose='numerical input',dtype=None):
        t=time.perf_counter();is_device=isinstance(a,self.xp.ndarray)
        out=self.xp.asarray(a,dtype=dtype)
        if is_device and out.data.ptr==a.data.ptr:return out
        if not is_device:self._record('H2D',out,name,phase,purpose)
        elif out.data.ptr!=a.data.ptr:self._record('D2D',out,name,phase,purpose)
        self.synchronize();self.transfer_seconds+=time.perf_counter()-t;return out

    def matrix(self,a,*,name='matrix',identity=None):
        import scipy.sparse as sp
        t=time.perf_counter()
        if self.sparse.issparse(a):
            if a.data.device.id!=self.device_id:raise NumericalError('device matrix belongs to another GPU')
            if a.dtype==self.xp.complex128:out=a
            else:out=a.astype(self.xp.complex128);self._record('D2D',out,name,'host_to_device','matrix dtype conversion',identity)
        elif sp.issparse(a):
            out=self.sparse.csr_matrix(a.astype('complex128',copy=False));self._record('H2D',out,name,'host_to_device','initial sparse matrix upload',identity)
        else:out=self.asarray(a,name=name,phase='host_to_device',dtype=self.xp.complex128)
        if identity is not None:self._matrix_identities[self._storage_key(out)]=identity
        self.synchronize();self.transfer_seconds+=time.perf_counter()-t;return out

    def _storage_key(self,a):
        if self.sparse.issparse(a):return (a.data.data.ptr,a.indices.data.ptr,a.indptr.data.ptr,a.shape,str(a.dtype))
        return (a.data.ptr,a.shape,str(a.dtype))

    def validate_shared_matrix(self,a,identity):
        if a.data.device.id!=self.device_id or self._matrix_identities.get(self._storage_key(a))!=identity:
            raise NumericalError('shared device matrix identity/device mismatch')

    def to_host(self,a,*,name='result',phase='result_download',purpose='result or diagnostic output'):
        t=time.perf_counter()
        if self.sparse.issparse(a):
            self._record('D2H',a,name,phase,purpose);out=a.get()
        else:
            out=self.xp.asnumpy(a)
            if isinstance(a,self.xp.ndarray):self._record('D2H',a,name,phase,purpose)
        self.transfer_seconds+=time.perf_counter()-t;return out

    def device_dense(self,a,name):
        self.synchronize();start=time.perf_counter();out=a.toarray() if self.sparse.issparse(a) else a
        self.synchronize();self.stages['device_dense']=self.stages.get('device_dense',0.)+time.perf_counter()-start
        self.device_peak_used_bytes=max(self.device_peak_used_bytes,self.xp.get_default_memory_pool().used_bytes())
        return out

    def synchronize(self):self.xp.cuda.get_current_stream().synchronize();self.sync_count+=1
    def memory_info(self):
        free,total=self.xp.cuda.runtime.memGetInfo();pool=self.xp.get_default_memory_pool()
        self.device_peak_used_bytes=max(self.device_peak_used_bytes,pool.used_bytes())
        return dict(device=self.xp.cuda.runtime.getDeviceProperties(self.device_id)['name'].decode(),device_id=self.device_id,free_bytes=free,total_bytes=total,
                    pool_total_bytes=pool.total_bytes(),pool_used_bytes=pool.used_bytes(),device_peak_at_boundaries_bytes=self.device_peak_used_bytes,
                    upload_bytes_tracked=self.upload_bytes,download_bytes_tracked=self.download_bytes,device_copy_bytes_tracked=self.device_copy_bytes,
                    transfer_seconds_tracked=self.transfer_seconds,explicit_synchronizations=self.sync_count,transfers=self.transfers,stages=self.stages,
                    tracking_scope='explicit array/matrix boundary calls; implicit PCG control-scalar reads and library workspace copies are not byte-instrumented')
