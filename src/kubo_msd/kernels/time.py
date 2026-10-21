import numpy as np
from ..units import HBAR_EV_FS

def pair_kernels(delta_eV,time_fs,xp=np):
    d=xp.asarray(delta_eV); t=xp.asarray(time_fs).reshape((-1,)+(1,)*d.ndim)
    z=t*d/HBAR_EV_FS
    return t*t*xp.sinc(z/(2*xp.pi))**2,t*xp.sinc(z/xp.pi),xp.cos(z)
