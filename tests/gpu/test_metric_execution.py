from dataclasses import replace
import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


@pytest.mark.gpu
@pytest.mark.parametrize("policy",["fixed","adaptive"])
def test_gpu_batch_masks_early_convergence_and_remainder(gpu,policy):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  metric_residual_policy=policy,metric_gpu_check_batch=1,
                  metric_maxiter=19,metric_rtol=1e-12,metric_check_interval=4)
    S=np.diag([1.,2,3,4])+np.diag([.01,.02,.03],1)+np.diag([.01,.02,.03],-1)
    B=np.column_stack([np.zeros(4),[1,0,0,0],np.ones(4)])
    a=MetricSolver(S,cfg,gpu);b=MetricSolver(S,replace(cfg,metric_gpu_check_batch=16),gpu)
    x,r=a.solve_s(B);y,s=b.solve_s(B)
    np.testing.assert_allclose(x.get(),y.get(),rtol=1e-12,atol=1e-13)
    np.testing.assert_array_equal(r.iterations.get(),s.iterations.get())
    assert int(s.iterations[0])==0
    assert b.diagnostics()["control_status_reads"]<a.diagnostics()["control_status_reads"]


@pytest.mark.gpu
def test_gpu_batch_latches_curvature_failure_and_preserves_zero_column(gpu):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  metric_gpu_check_batch=16)
    m=MetricSolver(np.eye(2),cfg,gpu)
    # Corrupt only the device action after physical SPD validation to exercise
    # the runtime breakdown path, not construction rejection.
    m.S=-m.S
    with pytest.raises(ValueError) as caught:
        m.solve_s(gpu.asarray([[0,1],[0,1]],dtype=complex))
    assert caught.value.diagnostics["breakdown"]==[False,True]
    assert caught.value.diagnostics["iterations"]==[0,0]


@pytest.mark.gpu
def test_gpu_nonfinite_curvature_latches_before_batched_host_poll(gpu):
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  metric_gpu_check_batch=16)
    m=MetricSolver(np.eye(2),cfg,gpu)
    m.S.data[:] = gpu.xp.nan
    with pytest.raises(ValueError) as caught:
        m.solve_s(gpu.asarray([[0,1],[0,1]],dtype=complex))
    assert caught.value.diagnostics["breakdown"] == [False,True]
    assert caught.value.diagnostics["iterations"] == [0,0]
