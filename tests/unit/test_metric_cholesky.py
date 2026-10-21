import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


def config(**kw):
    return RunConfig([0], [0], solver="stochastic-kpm", kernel="jackson",
                     metric_solver="cholesky", metric_rtol=1e-12, **kw)


@pytest.mark.parametrize("kernel", ["reference", "cpp"])
def test_direct_factor_once_original_equation_extremes(monkeypatch, kernel):
    calls = []
    old = la.cholesky
    def record(*a, **kw):
        calls.append(1)
        return old(*a, **kw)
    monkeypatch.setattr(la, "cholesky", record)
    S = np.array([[2, .3j], [-.3j, .7]])
    original = S.copy()
    m = MetricSolver(S, config(cpu_kernel=kernel))
    for k in range(3):
        B = np.array([[1+1j, 0, 2e-180j, 1e180], [2j, 0, 1e-180, 2e180j]])
        Y, r = m.solve_s(B)
        np.testing.assert_allclose(Y/B.max(axis=0, initial=1), np.linalg.solve(S,B)/B.max(axis=0, initial=1),
                                   rtol=1e-12, atol=1e-12)
        assert np.all(r.iterations == 0)
        assert np.all(r.relative_residual <= 1e-12)
    np.testing.assert_array_equal(S, original)
    assert len(calls) == 1
    assert m.diagnostics()["factorizations"] == 1
    assert m.diagnostics()["solver"] == "cholesky"
    if kernel == "cpp":
        assert m.native.diagnostics()["calls"] == 3


def test_direct_memory_guard_precedes_dense_allocation(monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("allocation happened before memory guard")
    monkeypatch.setattr(sp.csr_matrix, "toarray", forbidden)
    with pytest.raises(ValueError, match="memory"):
        MetricSolver(sp.eye(200), config(memory_limit_gb=1e-6))


@pytest.mark.parametrize("options", [
    {"metric_residual_policy": "adaptive"}, {"metric_check_interval": 4},
    {"metric_gap_ratio": .2}, {"preconditioner": "block-jacobi"},
    {"metric_refine_maxiter": -1}])
def test_direct_rejects_ignored_or_invalid_options(options):
    with pytest.raises(ValueError):
        config(**options)


def test_direct_indefinite_and_unrepresentable_fail():
    with pytest.raises(ValueError, match="positive definite"):
        MetricSolver([[1, 2], [2, 1]], config())
    m = MetricSolver(np.eye(2)*.5, config())
    with pytest.raises(ValueError, match="representable"):
        m.solve_s(np.full((2,1), 1.5e308))


def test_direct_refines_against_original_equation(monkeypatch):
    m = MetricSolver([[2, .1j], [-.1j, 1]], config(cpu_kernel="reference"))
    solve = m.cholesky.solve_scaled
    calls = []
    def first_inexact(B):
        calls.append(1)
        return solve(B)*(1+1e-6 if len(calls)==1 else 1)
    monkeypatch.setattr(m.cholesky, "solve_scaled", first_inexact)
    y, report = m.solve_s(np.ones((2,2)))
    assert np.all(report.converged)
    assert len(calls) == 2
    assert m.diagnostics()["refinement_steps"] == 1


def test_native_direct_parallelizes_independent_rhs_without_changing_arithmetic():
    rng=np.random.default_rng(7201);A=rng.normal(size=(64,64))+1j*rng.normal(size=(64,64))
    S=A@A.conj().T+np.eye(64)*20
    B=rng.normal(size=(64,64))+1j*rng.normal(size=(64,64))
    B[:,0]=0;B[:,1]*=1e-100;B[:,2]*=1e100
    one=MetricSolver(S,config(cpu_kernel="cpp",omp_threads=1))
    many=MetricSolver(S,config(cpu_kernel="cpp",omp_threads=4))
    Y1,r1=one.solve_s(B);Y4,r4=many.solve_s(B)
    np.testing.assert_array_equal(Y1,Y4)
    np.testing.assert_array_equal(r1.relative_residual,r4.relative_residual)
    assert np.all(r4.converged)
    assert many.diagnostics()["direct_solve_workers_observed"]==4
    assert one.diagnostics()["direct_solve_workers_observed"]==1
