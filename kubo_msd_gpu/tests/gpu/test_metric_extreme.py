import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver

pytestmark = pytest.mark.gpu


def test_gpu_finite_complex_rhs_with_overflowing_modulus(gpu):
    cfg = RunConfig([0.], [0.], backend="gpu")
    rhs = np.array([[1.3e308+1.3e308j], [1.+0j]])
    solved, report = MetricSolver(np.eye(2), cfg, gpu).solve_s(gpu.xp.asarray(rhs))
    actual = gpu.to_host(solved)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.real/1.3e308, rhs.real/1.3e308, atol=1e-15)
    np.testing.assert_allclose(actual.imag/1.3e308, rhs.imag/1.3e308, atol=1e-15)
    assert bool(gpu.xp.all(report.converged))


def test_gpu_unrepresentable_solution_is_not_success(gpu):
    cfg = RunConfig([0.], [0.], backend="gpu")
    with pytest.raises(ValueError, match="representable") as exc:
        MetricSolver(.25*np.eye(2), cfg, gpu).solve_s(gpu.xp.asarray([[1e308], [1.]], dtype=complex))
    assert exc.value.diagnostics["converged"] == [False]


def test_gpu_near_threshold_residual_uses_remaining_iteration_budget(gpu):
    import scipy.linalg as la
    import scipy.sparse as sp
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
    s = (q*np.geomspace(1., 1000., 32))@q.T
    b = rng.normal(size=(32, 8))+1j*rng.normal(size=(32, 8))
    cfg = RunConfig([0.], [0.], backend="gpu", metric_rtol=1e-13, metric_maxiter=4000)
    x, report = MetricSolver(sp.csr_matrix(s), cfg, gpu).solve_s(gpu.xp.asarray(b))
    assert bool(gpu.xp.all(report.converged))
    assert bool(gpu.xp.all(report.relative_residual <= cfg.metric_rtol))
    assert int(gpu.xp.max(report.iterations)) < cfg.metric_maxiter
    np.testing.assert_allclose(gpu.to_host(x), la.solve(s, b), rtol=1e-10, atol=1e-12)
