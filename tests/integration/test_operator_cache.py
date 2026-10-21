import json
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from test_fast_operator_assembly import fixture


def test_physical_keys_ignore_resources_but_track_physics(tmp_path):
    from kubo_msd.operators.cache import table_key,material_key
    from kubo_msd.operators.builder import BuilderOptions
    s,o,h=fixture();a=BuilderOptions();b=BuilderOptions(operator_builder='cpp-omp',threads=2,build_distribution='mpi')
    assert table_key(o,a)==table_key(o,b)
    assert material_key(s,o,h,a)==material_key(s,o,h,b)
    assert table_key(o,a)!=table_key(o,BuilderOptions(tolerance=1e-6))
    s['positions_A'][1,0]+=.1
    assert material_key(s,o,h,a)!=material_key(fixture()[0],o,h,a)
    h[(0,0,0,0,0)]*=2
    assert material_key(s,o,h,a)!=material_key(s,o,fixture()[2],a)


def test_table_and_complete_cache_roundtrip_reject_corruption(tmp_path):
    from kubo_msd.operators.cache import publish_model,material_key
    from kubo_msd.operators.builder import BuilderOptions,build_operators
    from kubo_msd.io.model import load_model
    s,o,h=fixture();opts=BuilderOptions(table_cache_dir=tmp_path/'tables')
    a=build_operators(s,o,h,opts);b=build_operators(s,o,h,opts)
    assert a.diagnostics['details']['table']['cache']=='cold'
    assert b.diagnostics['details']['table']['cache']=='hot'
    assert_array_equal(a.S.data,b.S.data)
    p=publish_model(a,tmp_path/'model',material_key(s,o,h,opts))
    assert (p.parent/'CACHE_READY').is_file()
    c=load_model(p)
    for x,y in [(a.S,c.S),(a.H_cov,c.H_cov)]+[(x,y) for q in 'xyz' for x,y in zip(a.composite[q],c.composite[q])]:
        assert_array_equal(x.data,y.data);assert_array_equal(x.indices,y.indices);assert_array_equal(x.indptr,y.indptr)
    (p.parent/'CACHE_READY').unlink()
    with pytest.raises(ValueError,match='READY'):load_model(p)


def test_no_ready_on_write_failure(tmp_path,monkeypatch):
    from kubo_msd.operators import cache
    from kubo_msd.operators.builder import BuilderOptions,build_operators
    s,o,h=fixture();a=build_operators(s,o,h,BuilderOptions())
    def fail(*args,**kwargs):raise OSError('injected disk error')
    monkeypatch.setattr(cache,'_write_model_files',fail)
    with pytest.raises(OSError):cache.publish_model(a,tmp_path/'failed','key')
    assert not list(tmp_path.rglob('CACHE_READY'))
