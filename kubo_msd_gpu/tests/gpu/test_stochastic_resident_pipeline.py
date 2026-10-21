import importlib
from dataclasses import replace
import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.operators.velocity import build_operators
from kubo_msd.solvers.stochastic import iter_stochastic_blocks
from kubo_msd.observables.statistics import estimate_ratio
from test_device_metric_setup import setup_model,runtime


def test_tail_blocks_and_bootstrap_match():
    ctx,_,_=setup_model(3)
    assert importlib.util.find_spec('kubo_msd.observables.statistics_gpu'), 'missing device bootstrap'
    from kubo_msd.observables.statistics_gpu import estimate_ratio_gpu
    rng=np.random.default_rng(92); A=rng.normal(size=(3,2,4,7)); D=rng.random((3,7)); counts=[3,3,1]
    value,se,status=estimate_ratio(A,D,counts)
    g,e,s=estimate_ratio_gpu(ctx.asarray(A),ctx.asarray(D),counts,ctx)
    np.testing.assert_allclose(ctx.to_host(g),value,rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(ctx.to_host(e),se,rtol=1e-12,atol=1e-13)
    assert s==status

@pytest.mark.parametrize('method',['stepwise','spectral'])
def test_stepwise_spectral_same_sample_ids(monkeypatch,method):
    ctx,model,host=setup_model(3)
    cfg=RunConfig(np.linspace(-.4,.4,5),[0,.03,.08],solver='stochastic-kpm',kernel='jackson',
        spectral_bounds=(-2,2),n_moments=12,axes=('x',),nsamples=5,statistics_block=2,
        sample_batch=2,cpu_kernel='reference',include_vac=True,integration_dt_fs=.02,
        metric_rtol=1e-12,sampling_tol=1e-12)
    ref=list(iter_stochastic_blocks(build_operators(host,cfg),cfg,method))
    gcfg=replace(cfg,backend='gpu'); ops=runtime()(model,gcfg,ctx)
    import cupy as cp
    old=cp.asnumpy
    def guard(a,*args,**kw):
        assert not (a.ndim==2 and a.shape[0]==3 and a.shape[1] in (1,2,3)), 'matrix/state roundtrip'
        return old(a,*args,**kw)
    monkeypatch.setattr(cp,'asnumpy',guard)
    got=list(iter_stochastic_blocks(ops,gcfg,method))
    for a,b in zip(ref,got):
        for key in a.arrays:
            np.testing.assert_allclose(b.arrays[key],a.arrays[key],rtol=1e-9,atol=1e-11,equal_nan=True)
    np.testing.assert_array_equal(got[0].static['sample_ids'],np.arange(5))
    np.testing.assert_array_equal(got[0].static['block_counts'],[2,2,1])
    assert ops.S is ops.metric.S is model.S


def test_complete_metric_basis_trace_matches_exact():
    from kubo_msd.solvers.exact import iter_exact_blocks
    ctx,model,_=setup_model(3)
    cfg=RunConfig(np.linspace(-.4,.4,5),[0,.03,.08],backend='gpu',
        kernel='jackson',spectral_bounds=(-2,2),n_moments=12,axes=('x',),
        include_vac=False,metric_rtol=1e-12)
    exact=list(iter_exact_blocks(model,runtime()(model,cfg,ctx),cfg))
    values,vectors=ctx.xp.linalg.eigh(model.S.toarray())
    samples=(vectors/ctx.xp.sqrt(values))@vectors.conj().T*ctx.xp.sqrt(3.)
    stochastic=replace(cfg,solver='stochastic-kpm',nsamples=3,statistics_block=1,sample_batch=1)
    trace=list(iter_stochastic_blocks(runtime()(model,stochastic,ctx),stochastic,'stepwise',samples))
    np.testing.assert_allclose(trace[0].static['dos_states_per_eV'],exact[0].static['dos_states_per_eV'],
                               rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(trace[0].arrays['numerator_msd_A2/commutator'],
                               exact[0].arrays['numerator_msd_A2/pair_analytic'],rtol=1e-9,atol=1e-11)
