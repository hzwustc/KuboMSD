from dataclasses import replace
import numpy as np
import pytest
from numpy.testing import assert_allclose

def data():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.config import RunConfig
    A=np.array([[1,.2j],[.1,1.2]]); H=A.conj().T@np.diag([-.3,.4])@A; S=A.conj().T@A; V=A.conj().T@np.array([[.4,1j],[-1j,.3]])@A
    return ModelBundle(H,S,velocity={'x':V}),RunConfig(energy_eV=[-.1,.2],time_fs=[0,.1,.3],axes=('x',),kernel='jackson',n_moments=16,spectral_bounds=(-1,1),metric_rtol=1e-12)

@pytest.mark.parametrize('backend',['hybrid','gpu'])
def test_exact_cpu_gpu_equivalence(gpu,backend):
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.solvers.exact import iter_exact_blocks
    m,c=data(); ref=list(iter_exact_blocks(m,build_operators(m,c),c))[0]
    c=replace(c,backend=backend); actual=list(iter_exact_blocks(m,build_operators(m,c),c))[0]
    for key in ref.arrays: assert_allclose(actual.arrays[key],ref.arrays[key],rtol=1e-10,atol=1e-12)

def test_metric_gpu_residual_and_residency(gpu):
    from kubo_msd.metric.solve import MetricSolver
    m,c=data(); ms=MetricSolver(m.S,c,gpu); B=gpu.xp.asarray(np.array([[1,0,1e-150],[1j,0,1e150]],complex))
    Y,report=ms.solve_s(B)
    assert isinstance(Y,gpu.xp.ndarray); assert bool(gpu.xp.all(report.converged)); assert float(gpu.xp.max(report.relative_residual))<=1e-12

@pytest.mark.parametrize('method',['stepwise','spectral'])
def test_stochastic_same_initial_vectors_and_gpu_residency(gpu,method):
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.solvers.stepwise import stepwise_samples
    from kubo_msd.solvers.spectral import spectral_samples
    m,c=data(); c=replace(c,solver='stochastic-kpm'); fn=stepwise_samples if method=='stepwise' else spectral_samples; B=np.array([[1,1j],[1j,1]],complex)
    D,A,_=fn(build_operators(m,c),c,B); c=replace(c,backend='gpu'); ops=build_operators(m,c)
    d,a,_=fn(ops,c,gpu.xp.asarray(B))
    assert isinstance(d,gpu.xp.ndarray); assert_allclose(gpu.to_host(d),D,rtol=1e-10,atol=1e-12)
    for key in A: assert isinstance(a[key],gpu.xp.ndarray); assert_allclose(gpu.to_host(a[key]),A[key],rtol=1e-10,atol=1e-12)
