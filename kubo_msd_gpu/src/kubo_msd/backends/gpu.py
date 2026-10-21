"""Optional backend; never import CuPy for CPU work and never fall back silently."""
import time
from ..errors import BackendUnavailable

class GPUBackend:
    name='gpu'
    def __init__(self):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as sparse
            if cp.cuda.runtime.getDeviceCount()<1: raise RuntimeError('no CUDA devices')
            # Allocation alone succeeds even when an older NVRTC cannot build
            # kernels for the installed device (e.g. CUDA 12.4 on Blackwell).
            probe=cp.asarray([1+1j],dtype=cp.complex128)
            if float((probe.conj()*probe).real.sum())!=2.:
                raise RuntimeError('complex128 GPU kernel smoke check failed')
            cp.cuda.get_current_stream().synchronize()
        except Exception as exc: raise BackendUnavailable(f'GPU requested but CuPy/CUDA unavailable: {exc}; check CUDA_PATH and LD_LIBRARY_PATH select the same compatible toolkit') from exc
        self.xp=cp; self.sparse=sparse; self.upload_bytes=0; self.download_bytes=0; self.transfer_seconds=0.; self.sync_count=0
    def asarray(self,a):
        t=time.perf_counter(); out=self.xp.asarray(a); self.synchronize(); self.upload_bytes+=out.nbytes; self.transfer_seconds+=time.perf_counter()-t; return out
    def matrix(self,a):
        import scipy.sparse as sp
        t=time.perf_counter()
        out=self.sparse.csr_matrix(a.astype('complex128')) if sp.issparse(a) else self.xp.asarray(a,dtype=self.xp.complex128)
        self.synchronize(); self.upload_bytes+=int(out.data.nbytes+out.indices.nbytes+out.indptr.nbytes) if self.sparse.issparse(out) else out.nbytes
        self.transfer_seconds+=time.perf_counter()-t; return out
    def to_host(self,a):
        t=time.perf_counter(); out=self.xp.asnumpy(a); self.download_bytes+=out.nbytes; self.transfer_seconds+=time.perf_counter()-t; return out
    def synchronize(self): self.xp.cuda.get_current_stream().synchronize(); self.sync_count+=1
    def memory_info(self):
        free,total=self.xp.cuda.runtime.memGetInfo()
        device_id=self.xp.cuda.runtime.getDevice()
        return {'device':self.xp.cuda.runtime.getDeviceProperties(device_id)['name'].decode(),'device_id':device_id,'free_bytes':free,'total_bytes':total,'pool_total_bytes':self.xp.get_default_memory_pool().total_bytes(),'pool_used_bytes':self.xp.get_default_memory_pool().used_bytes(),'upload_bytes_tracked':self.upload_bytes,'download_bytes_tracked':self.download_bytes,'transfer_seconds_tracked':self.transfer_seconds,'explicit_synchronizations':self.sync_count,'tracking_scope':'backend boundary calls; scalar convergence synchronization is additional'}
