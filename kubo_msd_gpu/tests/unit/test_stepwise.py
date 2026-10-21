import numpy as np
import scipy.linalg as la
from dataclasses import replace
from numpy.testing import assert_allclose
from test_propagation import setup_ops

def test_complete_metric_basis_stepwise_equals_exact():
    from kubo_msd.solvers.stepwise import stepwise_samples
    from kubo_msd.solvers.exact import iter_exact_blocks
    ops,cfg=setup_ops(); S=ops.host_S.toarray(); B=np.sqrt(2)*la.solve_triangular(la.cholesky(S,lower=True).conj().T,np.eye(2),lower=False)
    dos,nums,diag=stepwise_samples(ops,replace(cfg,include_vac=False),B)
    exact=list(iter_exact_blocks(None,ops,replace(cfg,solver='exact',backend='cpu')))[0]
    assert_allclose(dos.mean(0),exact.static['dos_states_per_eV'],rtol=1e-10,atol=1e-12)
    assert_allclose(nums['msd_A2/commutator'].mean(0),exact.arrays['numerator_msd_A2/pair_analytic'],rtol=1e-10,atol=1e-12)
