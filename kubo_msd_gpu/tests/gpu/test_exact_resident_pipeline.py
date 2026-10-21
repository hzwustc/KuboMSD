from dataclasses import replace
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.operators.velocity import build_operators
from kubo_msd.solvers.exact import iter_exact_blocks
from test_device_metric_setup import setup_model,runtime

@pytest.mark.parametrize('kernel',['gaussian','jackson'])
@pytest.mark.parametrize('selection',['all','window'])
def test_exact_without_host_dense_or_scipy_eigh(monkeypatch,kernel,selection):
    ctx,model,host=setup_model()
    cfg=RunConfig(np.linspace(-.3,.3,11),[0,.07,.2],axes=('x','y'),kernel=kernel,
        spectral_bounds=(-2,2),n_moments=24,cpu_kernel='reference',metric_rtol=1e-12,
        initial_state_selection=selection,pair_block=3,time_block=2)
    ref=list(iter_exact_blocks(host,build_operators(host,cfg),cfg))
    gpu_cfg=replace(cfg,backend='gpu'); ops=runtime()(model,gpu_cfg,ctx)
    monkeypatch.setattr(la,'eigh',lambda *a,**k: pytest.fail('CPU eigensolver called'))
    monkeypatch.setattr(sp.csr_matrix,'toarray',lambda *a,**k: pytest.fail('host dense called'))
    import cupy as cp
    old=cp.asnumpy
    def guard(a,*args,**kw):
        assert a.shape!=(8,8), 'host C/H/S download'
        return old(a,*args,**kw)
    monkeypatch.setattr(cp,'asnumpy',guard)
    got=list(iter_exact_blocks(model,ops,gpu_cfg))
    for x,y in zip(ref,got):
        for key in x.arrays:
            np.testing.assert_allclose(y.arrays[key],x.arrays[key],rtol=1e-9,atol=1e-11,equal_nan=True)
    assert got[0].metadata['eigensolve']['max_normalized_residual']<1e-10
    assert ops.metric.S is model.S


def test_degenerate_subspace_and_window():
    ctx,model,host=setup_model(4)
    from kubo_msd.solvers.exact import solve_generalized_gpu
    from kubo_msd.operators.gpu.contracts import DeviceModelBundle
    S=host.S.toarray(); L=la.cholesky(S,lower=True); H=L@np.diag([1.,1.,2.,3.])@L.conj().T
    m=DeviceModelBundle(ctx.matrix(sp.csr_matrix(H)),model.S,0)
    e,C,report=solve_generalized_gpu(m,ctx)
    np.testing.assert_allclose(ctx.to_host(e),[1,1,2,3],atol=1e-12)
    assert report['metric_orthogonality']<1e-10
