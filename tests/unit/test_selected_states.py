import numpy as np
from numpy.testing import assert_allclose

def test_window_keeps_all_intermediate_states():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.config import RunConfig
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.solvers.exact import iter_exact_blocks
    H=np.diag([0.,10.]); V=np.array([[0,2.],[2.,0]])
    m=ModelBundle(H,np.eye(2),velocity={'x':V})
    c=RunConfig(energy_eV=[-.1,0,.1],time_fs=[0,.1],axes=('x',),sigma_eV=.1,initial_state_selection='window')
    b=list(iter_exact_blocks(m,build_operators(m,c),c))[0]
    assert_allclose(b.static['initial_state_indices'],[0])
    assert b.arrays['msd_A2/pair_analytic'][0,1,1]>0
    assert_allclose(b.static['retained_weight'],1)

def test_selection_expands_entire_degenerate_cluster():
    from kubo_msd.solvers.exact import select_initial_states
    assert_allclose(select_initial_states(np.array([0,1,1+1e-12,2]),(.5,1),0,1e-10),[1,2])
