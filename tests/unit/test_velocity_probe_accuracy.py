import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.metric.solve import MetricSolver
from kubo_msd.operators.velocity import build_operators


@pytest.mark.parametrize("kernel",["reference","cpp"])
@pytest.mark.parametrize("representation",["position","composite"])
def test_strict_velocity_probe_retries_without_changing_transport_tolerance(
        ill_conditioned_velocity_model,kernel,representation):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",axes=("x",),
                  cpu_kernel=kernel,metric_rtol=1e-6,metric_residual_policy="adaptive")
    model=ill_conditioned_velocity_model
    if representation=="composite":
        X=model.position["x"]
        model=ModelBundle(model.H_cov,model.S,composite={"x":(np.zeros_like(X),X,X)})
    ops=build_operators(model,cfg)
    report=ops.velocity_hermiticity["x"]
    assert report["attempts"][0]["relative_error"]>1e-10
    assert report["relative_error"]<=1e-10
    assert report["attempts"][-1]["metric_rtol"]<cfg.metric_rtol
    assert ops.config.metric_rtol==1e-6
    B=np.arange(18).reshape(9,2)*(1+.2j)+1
    expected=MetricSolver(ops.host_S,cfg).solve_s(ops.apply_h_cov(B))[0]
    # Detect a validation-only tolerance leaking into the native engine.
    np.testing.assert_allclose(ops.apply_h_action(B),expected,rtol=5e-12,atol=5e-12)


@pytest.mark.parametrize("kernel",["reference","cpp"])
def test_probe_retry_cannot_accept_a_nonhermitian_physical_velocity(kernel):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",axes=("x",),
                  cpu_kernel=kernel,metric_rtol=1e-6)
    model=ModelBundle(np.diag([1.,2]),np.eye(2),
        composite={"x":(np.eye(2),np.zeros((2,2)),np.zeros((2,2)))})
    with pytest.raises(ValueError,match="not Hermitian"):
        build_operators(model,cfg)


@pytest.mark.parametrize("kernel",["reference","cpp"])
def test_failed_stricter_solve_retains_probe_and_column_diagnostics(
        ill_conditioned_velocity_model,kernel):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",axes=("x",),
                  cpu_kernel=kernel,metric_rtol=1e-6,metric_residual_policy="adaptive",
                  metric_maxiter=10)
    with pytest.raises(ValueError) as caught:
        build_operators(ill_conditioned_velocity_model,cfg)
    details=caught.value.diagnostics
    assert "relative_residual" in details and not all(details["converged"])
    probe=details["velocity_probe"]
    assert probe["axis"]=="x" and probe["threshold"]==1e-10
    assert probe["attempts"][0]["relative_error"]>1e-10
    assert probe["attempts"][-1]["status"]=="S_SOLVE_FAILED"
    assert probe["attempts"][-1]["metric_rtol"]==1e-12
