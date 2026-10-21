import numpy as np
import scipy.sparse as sparse

class CPUBackend:
    name='cpu'
    xp=np
    sparse=sparse
    def asarray(self,a,*,dtype=None,**kwargs): return np.asarray(a,dtype=dtype)
    def matrix(self,a,**kwargs): return a.astype(np.complex128) if sparse.issparse(a) else np.asarray(a,dtype=np.complex128)
    def to_host(self,a,**kwargs): return np.asarray(a)
    def synchronize(self): pass
    def memory_info(self): return {}
