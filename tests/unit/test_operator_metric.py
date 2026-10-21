import numpy as np
import scipy.linalg as la
from numpy.testing import assert_allclose

def test_nonorthogonal_open_position_and_covariant_action():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.config import RunConfig
    from kubo_msd.units import HBAR_EV_FS
    A=np.array([[1,.2j],[.3,1.2]],complex); S=A.conj().T@A
    H=A.conj().T@np.diag([-.4,.8])@A; X=A.conj().T@np.array([[0,1j],[-1j,2]])@A
    model=ModelBundle(H,S,position={'x':X})
    ops=build_operators(model,RunConfig(energy_eV=[0],time_fs=[0],axes=('x',)))
    expected=1j*(H@la.solve(S,X)-X@la.solve(S,H))/HBAR_EV_FS
    B=np.array([[1+2j,0],[.4,1]],complex)
    assert_allclose(ops.apply_v_cov('x',B),expected@B,atol=1e-12)
    assert_allclose(S@ops.apply_h_action(B),H@B,atol=1e-12)

def test_composite_equals_open_commutator_and_origin_invariance():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.config import RunConfig
    H=np.array([[1,.3j],[-.3j,2]]); S=np.array([[1,.1],[.1,1.2]]); X=np.array([[.1,.4],[.4,1.5]]); r=np.array([0,1])
    cfg=RunConfig(energy_eV=[0],time_fs=[0],axes=('x',))
    ref=build_operators(ModelBundle(H,S,position={'x':X}),cfg).apply_v_cov('x',np.eye(2))
    for origin in (0,10):
        rr=r+origin; XX=X+origin*S
        comp={'x':(H*(rr[None,:]-rr[:,None]),XX-rr[:,None]*S,XX-S*rr[None,:])}
        op=build_operators(ModelBundle(H,S,composite=comp),cfg)
        assert_allclose(op.apply_v_cov('x',np.eye(2)),ref,atol=1e-12)
