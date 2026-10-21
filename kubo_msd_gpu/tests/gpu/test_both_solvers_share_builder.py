from dataclasses import replace
from pathlib import Path
import numpy as np
from kubo_msd.config import RunConfig
from kubo_msd.backends.gpu_context import create_gpu_context
from kubo_msd.operators.gpu.contracts import GpuBuildOptions
from kubo_msd.solvers.exact import iter_exact_blocks
from kubo_msd.solvers.stochastic import iter_stochastic_blocks


def test_build_once_run_exact_then_stochastic(tmp_path,monkeypatch):
    from kubo_msd.operators.gpu.input import parse_build_input
    from kubo_msd.operators.gpu.builder import build_gpu_operators
    from kubo_msd.operators.gpu.runtime import make_device_operators
    from test_shared_builder_cache import matrices
    ctx=create_gpu_context(0,1024**3)
    p=parse_build_input(Path(__file__).parents[2]/'examples/raw_lcao/input.json')
    result=build_gpu_operators(p,GpuBuildOptions(table_cache_dir=tmp_path/'tables'),ctx)
    m=result.model;before=[ctx.to_host(x.data) for x in matrices(m)]
    cfg=RunConfig(np.linspace(-.2,.2,5),[0,.03,.09],backend='gpu',kernel='jackson',spectral_bounds=(-3,3),
        n_moments=12,nsamples=3,statistics_block=2,sample_batch=2,include_vac=False,axes=('x',),cpu_kernel='reference')
    ops=make_device_operators(m,cfg,ctx)
    kernels=len(ctx.kernel_calls)
    exact=list(iter_exact_blocks(m,ops,cfg))
    stochastic=list(iter_stochastic_blocks(ops,replace(cfg,solver='stochastic-kpm'),'stepwise'))
    assert exact and stochastic and ops.model is m and ops.S is m.S
    assert len(ctx.kernel_calls)==kernels, 'solver rebuilt operator kernels'
    for v,w in zip(before,matrices(m)):np.testing.assert_array_equal(v,ctx.to_host(w.data))
