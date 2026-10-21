def pytest_addoption(parser):
    parser.addoption('--require-gpu',action='store_true',help='fail instead of skipping when real GPU acceptance is requested')

import pytest

@pytest.fixture
def ill_conditioned_velocity_model():
    import numpy as np
    from kubo_msd.contracts import ModelBundle
    rng=np.random.default_rng(51);n=9
    U,_=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    S=(U*np.geomspace(.001,1,n))@U.conj().T
    A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));H=(A+A.conj().T)/2
    A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));X=(A+A.conj().T)/2
    return ModelBundle(H,S,position={"x":X})

@pytest.fixture
def built_operator_cache(tmp_path):
    import numpy as np
    from kubo_msd.operators.lcao_integrals import OrbitalSet
    from kubo_msd.operators.builder import BuilderOptions,build_operators
    from kubo_msd.operators.cache import publish_model,material_key
    r=np.linspace(0,8,801);orb=OrbitalSet('X',r,[(0,0,2/np.pi**.25*np.exp(-r*r/2))],[1.],{})
    structure=dict(cell_A=np.eye(3)*12,positions_A=np.array([[0.,0.,0.],[1.2,.3,.2]]),symbols=['X','X'])
    orbitals={'X':orb};h={(0,0,0,0,0):np.array([[-1.]]),(1,1,0,0,0):np.array([[1.]]),(0,1,0,0,0):np.array([[.2]])}
    opts=BuilderOptions();result=build_operators(structure,orbitals,h,opts)
    return publish_model(result,tmp_path/'operators',material_key(structure,orbitals,h,opts))
