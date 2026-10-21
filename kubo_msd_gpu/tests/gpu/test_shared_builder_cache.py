from dataclasses import replace
from pathlib import Path
import importlib
import numpy as np
import pytest
from kubo_msd.backends.gpu_context import create_gpu_context
from kubo_msd.operators.gpu.contracts import GpuBuildOptions
from gpu_fixtures import parsed,gaussian


def test_periodic_support_exceeds_h_graph():
    assert importlib.util.find_spec('kubo_msd.operators.gpu.plan'), 'missing pair plan'
    from kubo_msd.operators.gpu.plan import make_pair_plan
    p=parsed({'X':gaussian(lmax=0)})
    p=replace(p,structure={'cell_A':np.array([[3.,0,0],[.6,3.3,0],[.2,.3,4.]]),'positions_A':np.array([[0,0,0],[6.4,.2,.1]]),'symbols':['X','X']},orbital_order=('a','b'),atom_slices=((0,1),(1,2)))
    plan=make_pair_plan(p)
    assert any(tuple(k[:2])==(0,0) and any(k[2:]) for k in plan.keys)
    expected=set();cut=2*p.orbitals['X'].cutoff_A
    from itertools import product
    for i in range(2):
        for j in range(i,2):
            for R in product(range(-7,8),repeat=3):
                if i==j and R<(0,0,0):continue
                d=p.structure['positions_A'][j]-p.structure['positions_A'][i]+np.array(R)@p.structure['cell_A']
                if np.linalg.norm(d)<cut:expected.add((i,j,*R))
    assert set(map(tuple,plan.keys))==expected
    assert len(plan.pair_ids)==len(set(map(tuple,plan.keys)))


def test_stable_csr_across_batches_and_cache(tmp_path):
    from kubo_msd.operators.gpu.input import parse_build_input
    from kubo_msd.operators.gpu.builder import build_gpu_operators
    from kubo_msd.operators.gpu.cache import save_operator_cache,load_operator_cache_gpu
    ctx=create_gpu_context(0,1024**3)
    inp=Path(__file__).parents[2]/'examples/raw_lcao/input.json'
    p=parse_build_input(inp)
    a=build_gpu_operators(p,GpuBuildOptions(pair_batch_size=1,table_cache_dir=tmp_path/'tables'),ctx)
    b=build_gpu_operators(p,GpuBuildOptions(pair_batch_size=5,table_cache_dir=tmp_path/'tables'),ctx)
    assert a.model.cache_identity==b.model.cache_identity
    for m,n in zip(matrices(a.model),matrices(b.model)):
        for field in ('data','indices','indptr'):
            np.testing.assert_array_equal(ctx.to_host(getattr(m,field)),ctx.to_host(getattr(n,field)))
    save_operator_cache(a.model,tmp_path/'model',ctx)
    c=load_operator_cache_gpu(tmp_path/'model/model.json',ctx)
    for m,n in zip(matrices(a.model),matrices(c)):
        for field in ('data','indices','indptr'):
            np.testing.assert_array_equal(ctx.to_host(getattr(m,field)),ctx.to_host(getattr(n,field)))
    assert (tmp_path/'model/CACHE_READY').is_file()
    with pytest.raises(FileExistsError): save_operator_cache(a.model,tmp_path/'model',ctx)


def matrices(m):
    return [m.H_cov,m.S,*m.velocity.values(),*m.position.values(),*[x for parts in m.composite.values() for x in parts]]

@pytest.mark.parametrize('example',['two_level','nonorthogonal'])
def test_existing_velocity_and_position_cache_on_device(example):
    from kubo_msd.operators.gpu.cache import load_operator_cache_gpu
    ctx=create_gpu_context(0,512*1024**2)
    p=Path(__file__).parents[2]/'examples'/example/'model.json'
    m=load_operator_cache_gpu(p,ctx)
    assert m.device_id==0 and m.position
    assert all(x.data.device.id==0 for x in matrices(m))


def test_explicit_velocity_roundtrip_and_mixed_source_rejection(tmp_path):
    from test_device_metric_setup import setup_model
    from kubo_msd.operators.gpu.cache import save_operator_cache,load_operator_cache_gpu
    ctx,model,_=setup_model(3)
    velocity=dict(model.position)
    explicit=replace(model,position={},velocity=velocity)
    save_operator_cache(explicit,tmp_path/'velocity',ctx)
    loaded=load_operator_cache_gpu(tmp_path/'velocity',ctx)
    assert loaded.velocity and not loaded.position
    for a,m in loaded.velocity.items():
        np.testing.assert_array_equal(ctx.to_host(m.data),ctx.to_host(velocity[a].data))
    with pytest.raises(ValueError,match='one velocity source'):
        save_operator_cache(replace(model,velocity=velocity),tmp_path/'mixed',ctx)
    assert not (tmp_path/'mixed/CACHE_READY').exists()
