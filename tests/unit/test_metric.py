import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pytest
from numpy.testing import assert_allclose

@pytest.mark.parametrize('pre',['diagonal','block-jacobi'])
def test_batch_cg_complex_scale_zero_and_true_residual(pre):
    from kubo_msd.metric.solve import MetricSolver
    from kubo_msd.config import RunConfig
    rng=np.random.default_rng(7); A=rng.normal(size=(8,8))+1j*rng.normal(size=(8,8)); S=A.conj().T@A+np.eye(8)
    B=rng.normal(size=(8,4))+1j*rng.normal(size=(8,4)); B*=np.array([1,0,1e-180,1e180])
    cfg=RunConfig(energy_eV=[0],time_fs=[0],preconditioner=pre,metric_rtol=1e-12)
    ms=MetricSolver(sp.csr_matrix(S),cfg,atom_slices=[(0,3),(3,8)])
    Y,r=ms.solve_s(B)
    for j in [0,2,3]: assert_allclose(Y[:,j]/np.max(np.abs(B[:,j])),la.solve(S,B[:,j]/np.max(np.abs(B[:,j]))),rtol=1e-10,atol=1e-12)
    assert np.all(r.converged); assert np.all(r.relative_residual<=1e-12)
    assert_allclose(Y[:,1],0)

def test_metric_non_spd_and_maxiter_fail():
    from kubo_msd.metric.solve import MetricSolver
    from kubo_msd.config import RunConfig
    cfg=RunConfig(energy_eV=[0],time_fs=[0],metric_maxiter=1)
    with pytest.raises(ValueError): MetricSolver(np.array([[1,2],[2,1]]),cfg)
    s=MetricSolver(np.array([[1,.2],[.2,2]]),cfg)
    with pytest.raises(ValueError): s.solve_s(np.ones((2,1)))

def test_cache_is_content_specific():
    from kubo_msd.metric.solve import MetricSolver
    from kubo_msd.config import RunConfig
    c=RunConfig(energy_eV=[0],time_fs=[0]); a=MetricSolver(np.eye(2),c); b=MetricSolver(np.diag([2.,3]),c)
    assert a.cache_key!=b.cache_key
    assert_allclose(b.solve_s(np.ones((2,1)))[0].ravel(),[.5,1/3])

def test_failed_metric_solve_carries_per_column_diagnostics():
    from kubo_msd.metric.solve import MetricSolver
    from kubo_msd.config import RunConfig
    cfg=RunConfig(energy_eV=[0],time_fs=[0],metric_maxiter=1)
    s=MetricSolver(np.array([[1,.2],[.2,2.]]),cfg)
    with pytest.raises(ValueError) as failure: s.solve_s(np.array([[0,1],[0,1.]]))
    detail=failure.value.diagnostics
    assert detail['converged']==[True,False]
    assert detail['iterations']==[0,1]
    assert len(detail['true_residual'])==2
