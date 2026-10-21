import numpy as np
from numpy.testing import assert_allclose

def test_nonuniform_constant_vac():
    from kubo_msd.observables.diffusion import integrate_vac,msd_gradient
    t=np.array([0,.1,.31,1.]); C=np.full((1,4,2),2.)
    M,D=integrate_vac(t,C)
    assert_allclose(M[0,:,0],2*t*t); assert_allclose(D[0,:,0],2*t)
    assert_allclose(msd_gradient(t,M)[0,:,0],2*t,atol=1e-14)

def test_negative_vac_not_clipped():
    from kubo_msd.observables.diffusion import integrate_vac
    t=np.linspace(0,5,10001); M,D=integrate_vac(t,np.cos(t)[None,:,None])
    assert_allclose(D[0,:,0],np.sin(t),atol=3e-8)
    assert_allclose(M[0,:,0],4*np.sin(t/2)**2,atol=1e-7)
    assert D[0,-1,0]<0
