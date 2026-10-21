import importlib
from pathlib import Path
import numpy as np
from kubo_msd.backends.gpu_context import create_gpu_context
from kubo_msd.operators.gpu.contracts import GpuBuildOptions
from gpu_fixtures import parsed


def test_gpu_cold_tables_match_cpu_channels():
    assert importlib.util.find_spec('kubo_msd.operators.gpu.tables'), 'missing GPU tables'
    from kubo_msd.operators.gpu.tables import build_tables_gpu
    ctx=create_gpu_context(0,1024**3); p=parsed(); options=GpuBuildOptions()
    tables=build_tables_gpu(p,options,ctx)
    ref=np.load(Path(__file__).parents[1]/'data/cpu_spdf_tables.npz')
    np.testing.assert_array_equal(tables.metadata['channel_keys'],ref['channel_keys'])
    np.testing.assert_allclose(ctx.to_host(tables.arrays['values']),ref['values'],rtol=1e-10,atol=1e-12)
    assert any(k['kernel']=='radial_transform' for k in ctx.kernel_calls)
    assert tables.metadata['table_source']=='gpu_cold'


def test_hot_table_invalidates_orbital_or_tolerance_change(tmp_path):
    from kubo_msd.operators.gpu.tables import build_tables_gpu
    ctx=create_gpu_context(0,1024**3); p=parsed()
    a=build_tables_gpu(p,GpuBuildOptions(table_cache_dir=tmp_path),ctx)
    calls=len(ctx.kernel_calls)
    b=build_tables_gpu(p,GpuBuildOptions(table_cache_dir=tmp_path),ctx)
    assert a.identity==b.identity and b.metadata['table_source']=='validated_cache'
    assert len(ctx.kernel_calls)==calls
    c=build_tables_gpu(p,GpuBuildOptions(integral_tolerance=1e-6,table_cache_dir=tmp_path),ctx)
    assert c.identity!=b.identity and c.metadata['settings']!=b.metadata['settings']


def test_gaunt_uploaded_once_and_corrupt_metadata_rejected(tmp_path):
    from kubo_msd.operators.gpu.tables import build_tables_gpu
    import json,pytest
    ctx=create_gpu_context(0,1024**3);p=parsed();cfg=GpuBuildOptions(table_cache_dir=tmp_path)
    build_tables_gpu(p,cfg,ctx);build_tables_gpu(p,cfg,ctx)
    assert sum(t['object']=='term_coefficients' and t['direction']=='H2D' for t in ctx.transfers)==1
    path=next(tmp_path.glob('*.npz'))
    with np.load(path,allow_pickle=False) as f: values=f['values'];meta=json.loads(str(f['metadata']))
    meta['step']*=2
    np.savez_compressed(path,values=values,metadata=json.dumps(meta))
    with pytest.raises(ValueError,match='cache'):build_tables_gpu(p,cfg,ctx)
