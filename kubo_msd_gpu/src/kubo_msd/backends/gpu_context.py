"""One device/stream, explicit transfer boundaries and conservative resource gates."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from ..errors import BackendUnavailable, NumericalError


class GPUContext:
    name = 'gpu'

    def __init__(self, device_id, memory_limit_bytes=None):
        if isinstance(device_id, bool) or not isinstance(device_id, int) or device_id < 0:
            raise ValueError('device must be one nonnegative integer')
        if memory_limit_bytes is not None and (isinstance(memory_limit_bytes, bool)
                or not isinstance(memory_limit_bytes, int) or memory_limit_bytes <= 0):
            raise ValueError('memory limit must be positive integer bytes')
        try:
            import cupy as cp
            import cupyx.scipy.sparse as sparse
            import cupyx
            from cupy.cuda import nvrtc
            if device_id >= cp.cuda.runtime.getDeviceCount():
                raise RuntimeError('requested device does not exist')
            self.device_id = device_id
            self.device = cp.cuda.Device(device_id)
            self.device.use()
            self.stream = cp.cuda.Stream(non_blocking=False)
            self.xp, self.sparse = cp, sparse
            self.transfers, self.timings, self.memory_samples, self.kernel_calls = [], [], [], []
            self.phase_name = 'capability'
            self.sync_count = 0
            self._modules = {}
            free, total = cp.cuda.runtime.memGetInfo()
            self.memory_limit_bytes = min(int(free*.85), memory_limit_bytes or int(free*.85))
            self.start_free_bytes = free
            prop = cp.cuda.runtime.getDeviceProperties(device_id)
            self.capabilities = {
                'device': prop['name'].decode(), 'device_id': device_id,
                'compute_capability': self.device.compute_capability,
                'driver_version': cp.cuda.runtime.driverGetVersion(),
                'runtime_version': cp.cuda.runtime.runtimeGetVersion(),
                'nvrtc_version': list(nvrtc.getVersion()), 'cupy_version': cp.__version__,
                'free_bytes_at_start': free, 'total_bytes': total,
                'compile_options': ['--std=c++17'], 'float64_complex128_smoke': 'NOT RUN',
            }
            with self.activate(), cupyx.errstate(linalg='raise'):
                a = cp.asarray([[2., .2j], [-.2j, 1.]], dtype=cp.complex128)
                L = cp.linalg.cholesky(a)
                e, v = cp.linalg.eigh(a)
                r = sparse.csr_matrix(a) @ v - v*e
                code = 'extern "C" __global__ void probe(double* x){x[0]=x[0]*2.0;}'
                kernel = cp.RawKernel(code, 'probe', options=('--std=c++17',))
                x = cp.asarray([3.], dtype=cp.float64)
                kernel((1,), (1,), (x,))
                if float(cp.linalg.norm(r)) > 1e-12 or float(x[0]) != 6. or not bool(cp.all(cp.isfinite(L))):
                    raise RuntimeError('required numerical capability failed')
            self.synchronize()
            self.capabilities['float64_complex128_smoke'] = 'PASS'
        except (ValueError, RuntimeError, ImportError) as exc:
            raise BackendUnavailable(f'GPU unavailable: {exc}; no CPU fallback') from exc

    @contextmanager
    def activate(self):
        with self.device, self.stream:
            yield self

    def check_device(self, a):
        arrays = (a.data, a.indices, a.indptr) if self.sparse.issparse(a) else (a,)
        if any(not isinstance(x, self.xp.ndarray) or x.device.id != self.device_id for x in arrays):
            raise ValueError('array belongs to wrong device or host')

    def record_transfer(self, direction, name, a, purpose, nbytes=None):
        self.transfers.append(dict(direction=direction, object=name, shape=list(a.shape),
                                   dtype=str(a.dtype), bytes=int(a.nbytes if nbytes is None else nbytes),
                                   phase=self.phase_name, purpose=purpose, device_id=self.device_id))

    def asarray(self, a, name='array', purpose='constants', dtype=None):
        with self.activate():
            if isinstance(a, self.xp.ndarray):
                self.check_device(a)
                out = self.xp.asarray(a, dtype=dtype)
                if out is not a: self.record_transfer('D2D', name, out, purpose)
                return out
            host = np.asarray(a, dtype=dtype)
            self.require_bytes(host.nbytes, name)
            out = self.xp.asarray(host)
            self.record_transfer('H2D', name, out, purpose)
            return out

    def contiguous(self, a, name):
        self.check_device(a)
        if a.flags.c_contiguous: return a
        self.require_bytes(a.nbytes, name)
        with self.activate(): out = self.xp.ascontiguousarray(a)
        self.record_transfer('D2D', name, out, 'layout')
        return out

    def matrix(self, a, name='matrix', purpose='cache_import'):
        if self.sparse.issparse(a):
            self.check_device(a)
            return a
        import scipy.sparse as sp
        a = sp.csr_matrix(a, dtype=np.complex128)
        if max(a.shape, default=0) >= 2**31 or a.nnz >= 2**31:
            raise ValueError('CSR exceeds supported int32 library index range')
        with self.activate():
            return self.sparse.csr_matrix((
                self.asarray(a.data, name+'.data', purpose),
                self.asarray(a.indices, name+'.indices', purpose, np.int32),
                self.asarray(a.indptr, name+'.indptr', purpose, np.int32)), shape=a.shape)

    def to_host(self, a, name='result', purpose='result_output'):
        self.check_device(a)
        with self.activate():
            self.synchronize()
            out = self.xp.asnumpy(a)
        self.record_transfer('D2H', name, a, purpose)
        return out

    def scalar(self, a, name='control'):
        self.check_device(a)
        if a.size != 1: raise ValueError('scalar transfer requires one element')
        return self.to_host(a, name, 'metadata').item()

    def require_bytes(self, additional, purpose, logical_limit=None):
        active = self.xp.get_default_memory_pool().used_bytes()
        free, _ = self.xp.cuda.runtime.memGetInfo()
        pool = self.xp.get_default_memory_pool()
        reusable = pool.total_bytes() - active
        limit = min(self.memory_limit_bytes, logical_limit or self.memory_limit_bytes)
        if additional < 0 or active+additional > limit or additional > (free+reusable)*.85:
            raise NumericalError(f'RESOURCE_BLOCKED: {purpose}, additional={int(additional)}, '
                                 f'active_pool={active}, effective_limit={limit}')

    def synchronize(self):
        self.stream.synchronize()
        self.sync_count += 1

    def memory_info(self):
        with self.device:
            free, total = self.xp.cuda.runtime.memGetInfo()
            pool = self.xp.get_default_memory_pool()
            return {**self.capabilities, 'free_bytes': free, 'total_bytes': total,
                    'pool_total_bytes': pool.total_bytes(), 'pool_used_bytes': pool.used_bytes(),
                    'effective_limit_bytes': self.memory_limit_bytes,
                    'upload_bytes_tracked': sum(t['bytes'] for t in self.transfers if t['direction']=='H2D'),
                    'download_bytes_tracked': sum(t['bytes'] for t in self.transfers if t['direction']=='D2H'),
                    'measurement': 'synchronized phase boundaries; not instantaneous peak; pool-external allocations reflected by free bytes'}

    @contextmanager
    def phase(self, name):
        prior = self.phase_name
        self.phase_name = name
        with self.activate():
            self.synchronize()
            start, end = self.xp.cuda.Event(), self.xp.cuda.Event()
            t = time.perf_counter()
            start.record()
            try: yield
            finally:
                end.record()
                self.synchronize()
                self.timings.append({'phase': name, 'wall_seconds': time.perf_counter()-t,
                                     'cuda_event_seconds': self.xp.cuda.get_elapsed_time(start, end)/1000})
                self.memory_samples.append({'phase': name, **self.memory_info()})
                self.phase_name = prior

    def kernel(self, filename, name):
        path = Path(__file__).parents[1]/'operators/gpu/kernels'/filename
        source = (path.parent/'common.cu').read_text()+'\n'+path.read_text()
        toolchain = {k:self.capabilities[k] for k in ('compute_capability','driver_version','runtime_version','nvrtc_version','cupy_version','compile_options')}
        identity = hashlib.sha256((source+json.dumps(toolchain, sort_keys=True,
                                  default=str)).encode()).hexdigest()
        if filename not in self._modules:
            with self.phase('jit_compile'):
                module = self.xp.RawModule(code=source, options=('--std=c++17',))
                module.compile()
            self._modules[filename] = module
        fn = self._modules[filename].get_function(name)
        def launch(grid, block, args):
            with self.activate():
                fn(grid, block, args)
            self.kernel_calls.append({'kernel': name, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                      'jit_identity': identity, 'phase': self.phase_name})
        return launch

    def report(self):
        return {'capabilities': self.capabilities, 'transfers': self.transfers,
                'stages': self.timings, 'memory_samples': self.memory_samples,
                'kernel_calls': self.kernel_calls,
                'scalar_tracking': 'explicit scalar helper tracked; CuPy boolean/float control checks additionally synchronize scalars'}


def create_gpu_context(device_id=0, memory_limit_bytes=None):
    return GPUContext(device_id, memory_limit_bytes)
