import importlib
import numpy as np
import pytest


def context():
    assert importlib.util.find_spec('kubo_msd.backends.gpu_context'), 'missing GPUContext'
    from kubo_msd.backends.gpu_context import create_gpu_context
    return create_gpu_context(0,256*1024**2)


def test_no_a100_or_80gb_gate():
    ctx=context(); assert ctx.capabilities['float64_complex128_smoke']=='PASS'
    assert ctx.capabilities['compute_capability']
    assert ctx.memory_limit_bytes<=256*1024**2


def test_budget_rejects_before_allocation():
    ctx=context(); before=ctx.xp.get_default_memory_pool().used_bytes()
    with pytest.raises(ValueError,match='RESOURCE_BLOCKED'): ctx.require_bytes(512*1024**2,'test')
    assert ctx.xp.get_default_memory_pool().used_bytes()==before


def test_strides_packed_or_rejected():
    ctx=context(); a=ctx.asarray(np.arange(12.).reshape(3,4),'input')
    b=ctx.contiguous(a.T,'transposed')
    assert b.flags.c_contiguous
    np.testing.assert_array_equal(ctx.to_host(b),np.arange(12.).reshape(3,4).T)
    assert any(t['direction']=='D2D' for t in ctx.transfers)
    assert any(t['direction']=='H2D' for t in ctx.transfers)
