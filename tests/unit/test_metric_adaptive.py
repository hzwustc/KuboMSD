from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


def config(**kw):
    return RunConfig(energy_eV=[0], time_fs=[0], solver="stochastic-kpm",
                     kernel="jackson", **kw)


def test_false_convergence_reactivates_only_failed_column():
    from kubo_msd.metric.residual import assess_residual
    # Column 0 falsely converges; 1 small gap keeps conjugacy; 2 truly converges;
    # 3 is already frozen and must never restart.
    rec = np.array([[.01, 2., .01, .2]])
    true = np.array([[.11, 2.01, .02, 9.]])
    conv, restart = assess_residual(rec, true, np.full(4, .1),
                                    np.array([1, 1, 1, 0], bool), .1, xp=np)
    np.testing.assert_array_equal(conv, [False, False, True, False])
    np.testing.assert_array_equal(restart, [True, False, False, False])


@pytest.mark.parametrize("kw", [{"metric_check_interval": 0}, {"metric_gap_ratio": 1},
                               {"metric_gap_ratio": 0}, {"metric_residual_policy": "typo"}])
def test_invalid_adaptive_controls_rejected(kw):
    with pytest.raises(ValueError):
        config(**kw)


def test_adaptive_cli_and_config_roundtrip():
    from kubo_msd.cli import parser
    a = parser().parse_args(["run", "--input", "x", "--out", "y",
                            "--metric-residual-policy", "adaptive",
                            "--metric-check-interval", "7", "--metric-gap-ratio", ".2"])
    c = config(metric_residual_policy=a.metric_residual_policy,
               metric_check_interval=a.metric_check_interval, metric_gap_ratio=a.metric_gap_ratio)
    assert RunConfig(**c.as_dict()).metric_check_interval == 7
    assert config().metric_residual_policy == "fixed"
    assert config().metric_check_interval == 32


@pytest.mark.parametrize("kernel", ["reference", "cpp"])
def test_adaptive_complex_batch_true_residual_extremes_and_override(kernel):
    rng = np.random.default_rng(4)
    U, _ = np.linalg.qr(rng.normal(size=(48, 48))+1j*rng.normal(size=(48, 48)))
    S = (U * np.geomspace(.01, 2, 48)) @ U.conj().T
    B = (rng.normal(size=(48, 8))+1j*rng.normal(size=(48, 8)))[:, ::2]
    B *= [0, 1, 1e-180, 1e180]
    c = config(cpu_kernel=kernel, metric_residual_policy="adaptive",
               metric_check_interval=4, metric_rtol=1e-12, metric_maxiter=400)
    m = MetricSolver(S, c)
    y, r = m.solve_s(B, SimpleNamespace(**{**c.as_dict(), "rtol": 2e-13}))
    assert np.all(r.relative_residual <= 2e-13)
    np.testing.assert_array_equal(y[:, 0], 0)
    for j in (1, 2, 3):
        scale = np.max(abs(B[:, j]))
        np.testing.assert_allclose(y[:, j]/scale, np.linalg.solve(S, B[:, j]/scale),
                                   rtol=3e-11, atol=3e-11)
    assert m.diagnostics()["true_residual_checks"] > 0
    if kernel == "cpp":
        assert m.native is not None and m.native.diagnostics()["calls"] == 1


@pytest.mark.parametrize("kernel", ["reference", "cpp"])
def test_actual_false_convergence_restarts_without_periodic_refresh(kernel):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(8)
        U, _ = np.linalg.qr(rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32)))
        S = (U*np.geomspace(1e-4, 1, 32))@U.conj().T
        B = rng.normal(size=(32, 3))+1j*rng.normal(size=(32, 3))
        B[:, 0] = 0
        c = config(cpu_kernel=kernel, metric_residual_policy="adaptive",
                   metric_check_interval=10000, metric_maxiter=1000, metric_rtol=1e-12)
        m = MetricSolver(S, c)
        Y, r = m.solve_s(B)
    # No periodic refresh is possible: a restart proves the actual candidate
    # false-convergence branch reached the solver, not just its controller.
    assert m.diagnostics()["residual_restarts"] > 0
    assert np.all(r.converged) and r.iterations[0] == 0
    np.testing.assert_array_equal(Y[:, 0], 0)
    for j in (1, 2):
        assert np.linalg.norm(B[:, j]-S@Y[:, j])/np.linalg.norm(B[:, j]) <= 1e-12


@pytest.mark.parametrize("kernel", ["reference", "cpp"])
@pytest.mark.parametrize("solver", ["pcg", "cholesky"])
def test_final_true_residual_acceptance_is_counted(kernel, solver):
    m = MetricSolver(np.eye(2), config(cpu_kernel=kernel, metric_solver=solver,
                                      metric_refine_maxiter=0))
    m.solve_s(np.ones((2, 2)))
    assert m.diagnostics()["true_residual_checks"] == 1
    if kernel == "cpp":
        assert m.diagnostics()["true_residual_refreshes"] == 0
