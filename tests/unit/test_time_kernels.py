import numpy as np
from numpy.testing import assert_allclose

def test_degenerate_near_degenerate_zero_and_signed_D():
    from kubo_msd.kernels.time import pair_kernels
    from kubo_msd.units import HBAR_EV_FS
    de=np.array([0,1e-15,1.]); t=np.array([0,1e-9,1.,3.])
    M,D,C=pair_kernels(de,t)
    assert_allclose(M[:,0],t*t); assert_allclose(D[:,0],t); assert_allclose(C[:,0],1)
    assert_allclose(M[:,1],t*t,rtol=1e-14)
    assert D[-1,-1]<0
    assert_allclose(M[:,2],4*HBAR_EV_FS**2*np.sin(t/(2*HBAR_EV_FS))**2,atol=1e-15)
