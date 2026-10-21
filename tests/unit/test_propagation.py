import numpy as np
import scipy.linalg as la
from numpy.testing import assert_allclose

def setup_ops():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.config import RunConfig
    A=np.array([[1,.1j],[.2,1.2]]); S=A.conj().T@A; H=A.conj().T@np.array([[.3,.4j],[-.4j,-.2]])@A
    V=A.conj().T@np.array([[.5,1j],[-1j,0.]])@A
    c=RunConfig(energy_eV=[-.1,.1],time_fs=[0,.2,.7],axes=('x',),kernel='jackson',spectral_bounds=(-1,1),n_moments=24,solver='stochastic-kpm',metric_rtol=1e-12,integration_dt_fs=.002,segment_fs=.3)
    return build_operators(ModelBundle(H,S,velocity={'x':V}),c),c

def test_propagation_against_expm_norm_inverse_segments():
    from kubo_msd.solvers.stepwise import propagate
    from kubo_msd.units import HBAR_EV_FS
    ops,cfg=setup_ops(); B=np.array([[1+1j,.2],[.4j,1]])
    out,diag=propagate(B,.7,ops,cfg)
    H=ops.host_H.toarray(); S=ops.host_S.toarray()
    expected=la.expm(-1j*la.solve(S,H)*.7/HBAR_EV_FS)@B
    assert_allclose(out,expected,rtol=1e-10,atol=1e-12)
    assert_allclose(propagate(out,-.7,ops,cfg)[0],B,rtol=1e-10,atol=1e-12)
    mid=propagate(B,.3,ops,cfg)[0]
    assert_allclose(propagate(mid,.4,ops,cfg)[0],out,atol=1e-12)
    assert_allclose(np.sum(B.conj()*(S@B),axis=0),np.sum(out.conj()*(S@out),axis=0),atol=1e-12)
