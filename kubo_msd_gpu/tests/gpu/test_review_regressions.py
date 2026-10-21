from dataclasses import replace
import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.errors import NumericalError
from test_device_metric_setup import setup_model, runtime


def test_spectral_actual_order_respects_device_limit(monkeypatch):
    import kubo_msd.solvers.spectral as spectral
    ctx, model, _ = setup_model(128)
    cfg = RunConfig([0], [0, 100], backend='gpu', solver='stochastic-kpm',
        kernel='jackson', n_moments=16, axes=('x',), spectral_bounds=(-10, 10),
        segment_fs=100, integration_dt_fs=100)
    ops = runtime()(model, cfg, ctx)
    samples = ctx.asarray(np.ones((128, 1), complex))
    ctx.memory_limit_bytes = 32 * 1024**2
    monkeypatch.setattr(spectral, 'chain', lambda *a, **k: pytest.fail('chain allocated before resource gate'))
    with pytest.raises(NumericalError, match='RESOURCE_BLOCKED'):
        spectral.spectral_samples(ops, cfg, samples)


def test_exact_uses_pool_aware_budget(monkeypatch):
    from kubo_msd.solvers.exact import iter_exact_blocks
    ctx, model, _ = setup_model(3)
    cfg = RunConfig([0], [0, .01], backend='gpu', axes=('x',))
    ops = runtime()(model, cfg, ctx)
    original = ctx.memory_info
    monkeypatch.setattr(ctx, 'memory_info', lambda: {**original(), 'free_bytes': 1})
    # This only emulates the legacy raw-free report. The real context budget gate
    # and the actual device eigensolver still run.
    assert list(iter_exact_blocks(model, ops, cfg))


def test_sample_batch_reduced_without_changing_samples(monkeypatch):
    from kubo_msd.solvers.stochastic import iter_stochastic_blocks
    ctx, model, _ = setup_model(3)
    cfg = RunConfig(np.linspace(-.4, .4, 3), [0, .02], backend='gpu',
        solver='stochastic-kpm', kernel='jackson', n_moments=8,
        spectral_bounds=(-2, 2), axes=('x',), nsamples=5, statistics_block=3,
        sample_batch=3, include_vac=False)
    ref = list(iter_stochastic_blocks(runtime()(model, cfg, ctx), cfg, 'stepwise'))
    original = ctx.require_bytes
    def bounded(additional, purpose, logical_limit=None):
        if purpose == 'stochastic sample workspace' and additional > 20000:
            raise NumericalError('RESOURCE_BLOCKED: bounded regression budget')
        return original(additional, purpose, logical_limit)
    monkeypatch.setattr(ctx, 'require_bytes', bounded)
    got = list(iter_stochastic_blocks(runtime()(model, cfg, ctx), cfg, 'stepwise'))
    assert got[0].metadata['resource_batch']['effective'] < 3
    for key in ref[0].arrays:
        np.testing.assert_allclose(got[0].arrays[key], ref[0].arrays[key],
                                   rtol=1e-9, atol=1e-11, equal_nan=True)
    np.testing.assert_array_equal(got[0].static['sample_ids'], np.arange(5))
