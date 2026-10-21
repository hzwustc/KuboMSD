import numpy as np
from dataclasses import replace
from numpy.testing import assert_allclose

def test_shared_axes_and_sample_batches():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.config import RunConfig
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.solvers.stochastic import iter_stochastic_blocks
    H=np.array([[0,.4j],[-.4j,.2]]); V=np.array([[.3,1],[1,-.1]]); m=ModelBundle(H,np.eye(2),velocity={'x':V,'y':V*.2})
    c=RunConfig(energy_eV=[-.1,.2],time_fs=[0,.1,.2],axes=('x','y'),kernel='jackson',spectral_bounds=(-1,1),n_moments=8,solver='stochastic-kpm',nsamples=9,statistics_block=2,sample_batch=4,include_vac=False)
    def calc(cfg): return list(iter_stochastic_blocks(build_operators(m,cfg),cfg,'stepwise'))[0]
    a=calc(c); b=calc(replace(c,sample_batch=1)); one=calc(replace(c,axes=('x',),axes_vectors=None))
    assert_allclose(a.arrays['msd_A2/commutator'],b.arrays['msd_A2/commutator'],rtol=1e-10,atol=1e-12)
    assert_allclose(a.arrays['msd_A2/commutator'][0],one.arrays['msd_A2/commutator'][0],rtol=1e-10,atol=1e-12)
    assert_allclose(a.static['block_counts'],[2,2,2,2,1])
