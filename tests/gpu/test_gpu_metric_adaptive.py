import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


@pytest.mark.gpu
def test_gpu_adaptive_true_residual(gpu):
    gpu_backend = gpu
    S = np.array([[2, .1j], [-.1j, 1.]])
    cfg = RunConfig(energy_eV=[0], time_fs=[0], solver="stochastic-kpm",
                    kernel="jackson", backend="gpu", metric_residual_policy="adaptive",
                    metric_check_interval=1, metric_rtol=1e-13)
    m = MetricSolver(S, cfg, gpu_backend)
    b = gpu_backend.asarray(np.array([[0, 1+2j], [0, 3j]]))
    y, r = m.solve_s(b)
    assert bool(gpu_backend.xp.all(r.converged))
    np.testing.assert_allclose(gpu_backend.to_host(y), np.linalg.solve(S, b.get()),
                               atol=1e-12, rtol=1e-12)


@pytest.mark.gpu
def test_gpu_actual_false_convergence_restarts(gpu):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(0)
        U, _ = np.linalg.qr(rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32)))
    S = (U*np.geomspace(1/30000, 1, 32))@U.conj().T
    B = rng.normal(size=(32, 3))+1j*rng.normal(size=(32, 3))
    B[:, 0] = 0
    cfg = RunConfig([0], [0], solver="stochastic-kpm", kernel="jackson", backend="gpu",
                    metric_residual_policy="adaptive", metric_check_interval=10000,
                    metric_maxiter=500, metric_rtol=1e-12, metric_gpu_check_batch=16)
    m = MetricSolver(S, cfg, gpu)
    Y, r = m.solve_s(B)
    assert m.diagnostics()["residual_restarts"] > 0
    assert bool(gpu.xp.all(r.converged)) and int(r.iterations[0]) == 0
    np.testing.assert_array_equal(Y.get()[:, 0], 0)
    for j in (1, 2):
        assert np.linalg.norm(B[:, j]-S@Y.get()[:, j])/np.linalg.norm(B[:, j]) <= 1e-12


@pytest.mark.gpu
@pytest.mark.parametrize("solver", ["pcg", "cholesky"])
def test_gpu_counts_final_true_residual(gpu, solver):
    cfg = RunConfig([0], [0], solver="stochastic-kpm", kernel="jackson", backend="gpu",
                    metric_solver=solver, metric_refine_maxiter=0)
    m = MetricSolver(np.eye(2), cfg, gpu)
    m.solve_s(np.ones((2, 2)))
    assert m.diagnostics()["true_residual_checks"] == 1
