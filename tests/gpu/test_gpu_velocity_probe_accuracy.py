import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.metric.solve import MetricSolver
from kubo_msd.operators.velocity import OperatorBundle


@pytest.mark.gpu
@pytest.mark.parametrize("representation",["position","composite"])
def test_gpu_strict_probe_retry_keeps_run_controls(gpu,ill_conditioned_velocity_model,representation):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  axes=("x",),metric_rtol=1e-6,metric_residual_policy="adaptive")
    model=ill_conditioned_velocity_model
    if representation=="composite":
        X=model.position["x"]
        model=ModelBundle(model.H_cov,model.S,composite={"x":(np.zeros_like(X),X,X)})
    ops=OperatorBundle(model,cfg,gpu)
    report=ops.velocity_hermiticity["x"]
    assert report["attempts"][0]["relative_error"]>1e-10
    assert report["relative_error"]<=1e-10
    B=gpu.asarray(np.arange(18).reshape(9,2)*(1+.2j)+1)
    expected=MetricSolver(ops.host_S,cfg,gpu).solve_s(ops.apply_h_cov(B))[0]
    np.testing.assert_allclose(ops.apply_h_action(B).get(),expected.get(),rtol=5e-12,atol=5e-12)
    assert ops.config.metric_rtol==1e-6


@pytest.mark.gpu
def test_gpu_failed_probe_solve_retains_diagnostics(gpu,ill_conditioned_velocity_model):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  axes=("x",),metric_rtol=1e-6,metric_residual_policy="adaptive",
                  metric_maxiter=10)
    with pytest.raises(ValueError) as caught:
        OperatorBundle(ill_conditioned_velocity_model,cfg,gpu)
    report=caught.value.diagnostics
    assert not all(report["converged"])
    assert report["velocity_probe"]["attempts"][0]["relative_error"]>1e-10
    assert report["velocity_probe"]["attempts"][-1]["status"]=="S_SOLVE_FAILED"
