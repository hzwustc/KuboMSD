import numpy as np
from ..config import grid

def msd_gradient(time_fs,msd,axis=1,xp=np):
    if len(time_fs)<2: return None
    return .5*xp.gradient(msd,xp.asarray(time_fs),axis=axis,edge_order=2 if len(time_fs)>2 else 1)

def integrate_vac(time_fs,C,xp=np,axis=1):
    t=grid(time_fs,'integration time',True)
    c=xp.moveaxis(C,axis,0); M=xp.zeros_like(c); D=xp.zeros_like(c)
    for j,h in enumerate(np.diff(t)):
        M[j+1]=M[j]+2*h*D[j]+h*h/3*(2*c[j]+c[j+1])
        D[j+1]=D[j]+h/2*(c[j]+c[j+1])
    return xp.moveaxis(M,0,axis),xp.moveaxis(D,0,axis)
