import numpy as np
import scipy.sparse as sp
from ..errors import KuboMSDError

def complete_images(images,counts):
    out={}
    for key,v in images.items():
        i,j,*R=key; v=np.asarray(v,complex)
        if i<0 or j<0 or i>=len(counts) or j>=len(counts) or v.shape!=(counts[i],counts[j]) or not np.isfinite(v).all(): raise KuboMSDError(f'invalid image block {key}')
        out[key]=out.get(key,0)+v
    for (i,j,rx,ry,rz),v in list(out.items()):
        partner=(j,i,-rx,-ry,-rz)
        if partner in out:
            if not np.allclose(out[partner],v.conj().T,rtol=1e-10,atol=1e-12): raise KuboMSDError(f'inconsistent R/-R Hermitian blocks {(i,j,rx,ry,rz)}')
        else: out[partner]=v.conj().T.copy()
    return out

def assemble_images(images,counts,positions,cell):
    offsets=np.r_[0,np.cumsum(counts)]; n=offsets[-1]; rr=[]; cc=[]; values=[[] for _ in range(7)]
    for (i,j,*R),v in images.items():
        rows,cols=np.indices(v.shape); rr.extend((rows+offsets[i]).ravel()); cc.extend((cols+offsets[j]).ravel())
        ket=np.asarray(R)@cell+positions[j]; displacement=ket-positions[i]
        values[0].extend(v.ravel())
        for a in range(3): values[1+a].extend((v*displacement[a]).ravel()); values[4+a].extend((v*ket[a]).ravel())
    matrices=[sp.coo_matrix((v,(rr,cc)),shape=(n,n),dtype=complex).tocsr() for v in values]
    return matrices[0],matrices[1:4],matrices[4:7]

def assemble_gamma(images,orbital_order):
    counts=[int(v) for v in orbital_order]
    return assemble_images(complete_images(images,counts),counts,np.zeros((len(counts),3)),np.eye(3))[0]
