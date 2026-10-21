import numpy as np
import scipy.linalg as la
import pytest
from numpy.testing import assert_allclose

def run(H,S,V,**kw):
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.config import RunConfig
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.solvers.exact import iter_exact_blocks
    cfg=RunConfig(energy_eV=[-.1,.3],time_fs=[0,.01,.4,2.],axes=('x',),sigma_eV=.7,**kw)
    m=ModelBundle(H,S,velocity={'x':V}); ops=build_operators(m,cfg)
    blocks=list(iter_exact_blocks(m,ops,cfg)); arrays={k:np.concatenate([b.arrays[k] for b in blocks],axis=1) for k in blocks[0].arrays}
    return arrays,blocks[0].static,cfg

@pytest.mark.parametrize('kernel',['gaussian','jackson'])
def test_exact_matches_expm_independent_position(kernel):
    from kubo_msd.units import HBAR_EV_FS as hb
    rng=np.random.default_rng(92); a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); H=(a+a.conj().T)/2
    a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)); X=(a+a.conj().T)/2
    V=1j*(H@X-X@H)/hb
    arr,static,cfg=run(H,np.eye(4),V,kernel=kernel,spectral_bounds=(-10,10),n_moments=32,pair_block=3,time_block=2)
    eps,Q=la.eigh(H)
    from kubo_msd.kernels.energy import energy_weights
    weights=energy_weights(cfg.energy_eV,eps,cfg)
    expected=[]
    for t in cfg.time_fs:
        U=la.expm(-1j*H*t/hb); A=U.conj().T@X@U-X
        expected.append([np.trace((Q*w)@Q.conj().T@A@A).real/sum(w) for w in weights])
    assert_allclose(arr['msd_A2/pair_analytic'][0],expected,rtol=1e-10,atol=1e-12)
    assert_allclose(arr['msd_A2/pair_analytic'][:,0],0,atol=1e-15)

@pytest.mark.parametrize('chunk',[1,3,99])
def test_nonorthogonal_change_of_basis_and_chunks(chunk):
    H=np.diag([-.5,.2,.2]); V=np.array([[1,1j,.3],[-1j,.8,.4j],[.3,-.4j,2.]])
    A=np.array([[1,.1j,.2],[.2,1,.1j],[0,.3,1.1]],complex)
    ref,_,_=run(H,np.eye(3),V)
    out,_,_=run(A.conj().T@H@A,A.conj().T@A,A.conj().T@V@A,pair_block=chunk)
    for k in ('msd_A2/pair_analytic','diffusion_A2_per_fs/pair_analytic','vac_A2_per_fs2/pair_analytic'): assert_allclose(out[k],ref[k],rtol=1e-10,atol=1e-12)
