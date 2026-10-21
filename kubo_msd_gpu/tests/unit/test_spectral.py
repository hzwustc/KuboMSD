import numpy as np
import scipy.linalg as la
from dataclasses import replace
from numpy.testing import assert_allclose
from test_propagation import setup_ops

def test_spectral_complete_basis_vac_and_integral_convergence():
    from kubo_msd.solvers.spectral import spectral_samples
    from kubo_msd.solvers.exact import iter_exact_blocks
    ops,cfg=setup_ops(); S=ops.host_S.toarray(); B=np.sqrt(2)*la.solve_triangular(la.cholesky(S,lower=True).conj().T,np.eye(2),lower=False)
    dos,nums,diag=spectral_samples(ops,cfg,B)
    exact=list(iter_exact_blocks(None,ops,replace(cfg,solver='exact')))[0]
    assert_allclose(nums['vac_A2_per_fs2/vac_integral'].mean(0),exact.arrays['numerator_vac_A2_per_fs2/pair_analytic'],rtol=1e-10,atol=1e-12)
    errors=[]
    for dt in (.02,.01,.005):
        _,z,_=spectral_samples(ops,replace(cfg,integration_dt_fs=dt),B)
        errors.append(np.max(abs(z['msd_A2/vac_integral'].mean(0)-exact.arrays['numerator_msd_A2/pair_analytic'])))
    assert errors[1]<errors[0]*.3 and errors[2]<errors[1]*.3

def test_spectral_segment_partition_is_equivalent():
    from kubo_msd.solvers.spectral import spectral_samples
    ops,cfg=setup_ops(); B=np.array([[1,1j],[1j,1]])
    d,a,_=spectral_samples(ops,cfg,B); e,b,_=spectral_samples(ops,replace(cfg,segment_fs=.23),B)
    for key in a: assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-12)
