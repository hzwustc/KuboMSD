import hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from .errors import KuboMSDError

def file_record(path):
    p=Path(path).resolve(); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}

def hermitian(A,name='matrix',tol=1e-10):
    A=sp.csr_matrix(A,dtype=np.complex128); A.sum_duplicates(); A.sort_indices()
    if len(A.shape)!=2 or A.shape[0]!=A.shape[1] or not A.shape[0] or not np.isfinite(A.data).all(): raise KuboMSDError(f'{name}: invalid square finite matrix')
    n=sp.linalg.norm(A); err=sp.linalg.norm(A-A.getH())
    if err>(tol*n if n else 0): raise KuboMSDError(f'{name}: not Hermitian (relative error {err/max(n,np.finfo(float).tiny):g})')
    return A

def matrix_fingerprint(A):
    A=sp.csr_matrix(A); A.sum_duplicates(); A.sort_indices(); h=hashlib.sha256()
    for v in (np.array(A.shape),A.indptr,A.indices,A.data): h.update(v.tobytes())
    return h.hexdigest()
