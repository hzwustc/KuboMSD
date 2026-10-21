"""Real-device checks: complex metric, near degeneracy, sharing and batching."""
from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg as la
from numpy.testing import assert_allclose

from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.metric.sampling import MetricSampler, sample_metric
from kubo_msd.metric.solve import MetricSolver
from kubo_msd.operators.velocity import build_operators
from kubo_msd.solvers.exact import iter_exact_blocks, solve_generalized
from kubo_msd.solvers.spectral import spectral_samples
from kubo_msd.solvers.stepwise import stepwise_samples
from kubo_msd.solvers.stochastic import iter_stochastic_blocks


def model_config():
    rng = np.random.default_rng(5901)
    n = 7
    A = np.eye(n) + .04*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    Q = la.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
    h = Q@np.diag([-.7,-.3,-.3+1e-13,0,.2,.5,.8])@Q.conj().T
    velocities = {}
    for axis in 'xyz':
        v = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        velocities[axis] = A.conj().T@(.1*(v+v.conj().T))@A
    model = ModelBundle(A.conj().T@h@A, A.conj().T@A,
                        velocity=velocities, atom_slices=[(0,3),(3,7)])
    cfg = RunConfig(energy_eV=[-.31,0,.23], time_fs=[0,.03,.11,.2],
                    axes=('x','y','z'), kernel='jackson', n_moments=16,
                    spectral_bounds=(-1,1), metric_rtol=1e-12,
                    sampling_tol=1e-12, pair_block=2, time_block=2,
                    energy_block=2, integration_dt_fs=.01, segment_fs=.07)
    return model, cfg


@pytest.mark.parametrize('backend', ['hybrid','gpu'])
@pytest.mark.parametrize('kernel', ['gaussian','jackson'])
@pytest.mark.parametrize('selection', ['all','window'])
def test_complex_near_degenerate_exact_all_blocks(gpu,backend,kernel,selection):
    model,cfg = model_config()
    cfg = replace(cfg,kernel=kernel,initial_state_selection=selection,buffer_eV=.01)
    cpu = list(iter_exact_blocks(model,build_operators(model,cfg),cfg))
    device_cfg = replace(cfg,backend=backend)
    actual = list(iter_exact_blocks(model,build_operators(model,device_cfg),device_cfg))
    assert len(actual)==len(cpu)==2
    for expected,block in zip(cpu,actual):
        for key in expected.arrays:
            assert_allclose(block.arrays[key],expected.arrays[key],rtol=1e-10,atol=1e-12)
    for key in cpu[0].static:
        assert_allclose(actual[0].static[key],cpu[0].static[key],rtol=1e-10,atol=1e-12)
    diag=actual[0].metadata['eigensolve']
    assert diag['max_normalized_residual']<1e-12
    assert diag['metric_orthogonality']<1e-12


@pytest.mark.parametrize('preconditioner',['diagonal','block-jacobi'])
def test_gpu_metric_true_residual_extreme_columns(gpu,preconditioner):
    model,cfg=model_config()
    cfg=replace(cfg,preconditioner=preconditioner)
    metric=MetricSolver(model.S,cfg,gpu,model.atom_slices)
    rng=np.random.default_rng(613)
    B=rng.normal(size=(7,4))+1j*rng.normal(size=(7,4))
    scales=np.array([1.,0.,1e-180,1e180]); B*=scales
    Y,report=metric.solve_s(gpu.xp.asarray(B))
    actual=gpu.to_host(Y)
    assert isinstance(Y,gpu.xp.ndarray)
    assert gpu.xp.all(report.converged)
    assert float(gpu.xp.max(report.relative_residual))<=1e-12
    assert_allclose(actual[:,1],0,atol=0)
    for j in (0,2,3):
        assert_allclose(actual[:,j]/scales[j],la.solve(model.S,B[:,j]/scales[j]),rtol=1e-10,atol=1e-12)


def test_gpu_metric_rejects_failed_columns(gpu):
    model,cfg=model_config()
    metric=MetricSolver(model.S,replace(cfg,metric_maxiter=1),gpu)
    with pytest.raises(ValueError) as caught:
        metric.solve_s(gpu.xp.asarray(np.column_stack([np.zeros(7),np.ones(7)])))
    assert caught.value.diagnostics['converged']==[True,False]
    assert caught.value.diagnostics['iterations']==[0,1]


def test_gpu_sampler_complete_covariance_and_same_sample_ids(gpu):
    model,cfg=model_config()
    cpu=build_operators(model,cfg)
    device=build_operators(model,replace(cfg,backend='gpu'))
    sampler=MetricSampler(device.metric,cfg)
    C=sampler.transform(gpu.xp.eye(7,dtype=complex))
    assert isinstance(C,gpu.xp.ndarray)
    assert_allclose(gpu.to_host(C@C.conj().T),la.inv(model.S),rtol=1e-11,atol=1e-12)
    whole=sample_metric(301,[0,3,7,11],device,cfg)
    parts=gpu.xp.column_stack([sample_metric(301,[j],device,cfg) for j in [0,3,7,11]])
    assert_allclose(gpu.to_host(whole),sample_metric(301,[0,3,7,11],cpu,cfg),rtol=1e-12,atol=1e-12)
    assert_allclose(gpu.to_host(whole),gpu.to_host(parts),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('method',['stepwise','spectral'])
def test_gpu_complete_metric_basis_trace_limit_and_residency(gpu,method):
    model,cfg=model_config()
    cfg=replace(cfg,time_block=8,solver='stochastic-kpm')
    cpu=build_operators(model,cfg)
    ref=list(iter_exact_blocks(model,cpu,replace(cfg,solver='exact')))[0]
    # Independent complete metric basis: (1/N) B B^dagger = S^-1.
    B=np.sqrt(7)*la.solve_triangular(la.cholesky(model.S,lower=True).conj().T,np.eye(7),lower=False)
    fn=stepwise_samples if method=='stepwise' else spectral_samples
    expected_dos,expected_arrays,_=fn(cpu,cfg,B)
    device=build_operators(model,replace(cfg,backend='gpu'))
    before=device.backend.download_bytes
    dos,arrays,_=fn(device,cfg,gpu.xp.asarray(B))
    assert device.backend.download_bytes==before
    assert isinstance(dos,gpu.xp.ndarray)
    assert_allclose(gpu.to_host(dos),expected_dos,rtol=1e-10,atol=1e-12)
    assert_allclose(gpu.to_host(dos.mean(0)),ref.static['dos_states_per_eV'],rtol=1e-10,atol=1e-12)
    for key in arrays:
        assert isinstance(arrays[key],gpu.xp.ndarray)
        assert_allclose(gpu.to_host(arrays[key]),expected_arrays[key],rtol=1e-10,atol=1e-12)
    key='msd_A2/commutator' if method=='stepwise' else 'vac_A2_per_fs2/vac_integral'
    reference='numerator_msd_A2/pair_analytic' if method=='stepwise' else 'numerator_vac_A2_per_fs2/pair_analytic'
    assert_allclose(gpu.to_host(arrays[key].mean(0)),ref.arrays[reference],rtol=1e-9,atol=1e-11)


@pytest.mark.parametrize('method',['stepwise','spectral'])
def test_gpu_shared_axes_batch_and_segment_invariance(gpu,method):
    model,cfg=model_config()
    cfg=replace(cfg,backend='gpu',solver='stochastic-kpm',nsamples=9,
                statistics_block=2,sample_batch=4,time_block=8,include_vac=False)
    def calc(c): return list(iter_stochastic_blocks(build_operators(model,c),c,method))[0]
    whole=calc(cfg)
    split=calc(replace(cfg,sample_batch=1,segment_fs=.13))
    one=calc(replace(cfg,axes=('x',),axes_vectors=None))
    estimator='commutator' if method=='stepwise' else 'vac_integral'
    key='msd_A2/'+estimator
    assert_allclose(whole.arrays[key],split.arrays[key],rtol=1e-9,atol=1e-11)
    assert_allclose(whole.arrays[key][0],one.arrays[key][0],rtol=1e-9,atol=1e-11)
    assert_allclose(whole.static['block_counts'],[2,2,2,2,1],rtol=0,atol=0)
