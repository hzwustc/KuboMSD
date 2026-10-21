import importlib
from dataclasses import replace
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle


def setup_model(n=8):
    from kubo_msd.backends.gpu_context import create_gpu_context
    from kubo_msd.operators.gpu.contracts import DeviceModelBundle
    ctx=create_gpu_context(0,512*1024**2)
    rng=np.random.default_rng(72); a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    S=np.eye(n)+.1*a.conj().T@a/n
    h=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)); H=(h+h.conj().T)/n
    x=np.diag(np.arange(n).astype(complex))
    host=ModelBundle(sp.csr_matrix(H),sp.csr_matrix(S),position={a:sp.csr_matrix(x) for a in 'xyz'},atom_slices=[(0,n)])
    model=DeviceModelBundle(ctx.matrix(host.H_cov,'H'),ctx.matrix(host.S,'S'),0,
        position={a:ctx.matrix(v,'X'+a) for a,v in host.position.items()},atom_slices=((0,n),))
    return ctx,model,host


def runtime():
    assert importlib.util.find_spec('kubo_msd.operators.gpu.runtime'), 'missing device operators'
    from kubo_msd.operators.gpu.runtime import make_device_operators
    return make_device_operators

@pytest.mark.parametrize('preconditioner',['diagonal','block-jacobi'])
def test_no_full_s_download_or_duplicate_upload(monkeypatch,preconditioner):
    ctx,model,host=setup_model(); before=len(ctx.transfers)
    import cupy as cp
    original=cp.asnumpy
    def guard(a,*args,**kwargs):
        assert a.shape!=(8,8), 'full matrix download'
        return original(a,*args,**kwargs)
    monkeypatch.setattr(cp,'asnumpy',guard)
    monkeypatch.setattr(sp.csr_matrix,'toarray',lambda *a,**k: pytest.fail('host densification'))
    cfg=RunConfig([-1,0,1],[0,.1],backend='gpu',preconditioner=preconditioner)
    ops=runtime()(model,cfg,ctx)
    assert ops.S is model.S and ops.metric.S is model.S
    rhs=ctx.asarray(np.ones((8,2),complex)*1e200)
    sol,rep=ops.solve_s(rhs)
    assert bool(cp.all(rep.converged))
    np.testing.assert_allclose(ctx.to_host(model.S@sol/rhs),1,rtol=1e-10)
    assert not any(t['object'].startswith('S') and t['direction']=='H2D' for t in ctx.transfers[before:])


def test_nondiagonal_complex_metric_and_extrema():
    ctx,model,host=setup_model(); cfg=RunConfig([-1,0,1],[0,.1],backend='gpu')
    ops=runtime()(model,cfg,ctx)
    from kubo_msd.metric.gpu_bounds import estimate_spectral_bounds_gpu
    report=estimate_spectral_bounds_gpu(ops,cfg)
    ev=la.eigvalsh(host.H_cov.toarray(),host.S.toarray())
    np.testing.assert_allclose(report['extrema'],ev[[0,-1]],rtol=1e-9,atol=1e-11)
    assert not report['certified']
    with pytest.raises(ValueError,match='bounds'):
        estimate_spectral_bounds_gpu(ops,replace(cfg,spectral_bounds=(-.0001,.0001)))


def test_bad_bounds_and_indefinite_s_rejected():
    ctx,model,host=setup_model()
    from kubo_msd.metric.gpu_setup import spd_bounds_gpu
    with pytest.raises(ValueError,match='positive definite'):
        spd_bounds_gpu(ctx.matrix(sp.diags([1.,-1.])) ,ctx)


def test_pcg_extreme_rhs_preserved():
    ctx,model,host=setup_model(2); cfg=RunConfig([-1,1],[0],backend='gpu')
    ops=runtime()(model,cfg,ctx)
    B=np.array([[1e300+1e300j,1e-300],[2e300,2e-300]],complex)
    X,r=ops.solve_s(ctx.asarray(B))
    np.testing.assert_allclose(ctx.to_host(X)/B,la.solve(host.S.toarray(),B)/B,rtol=1e-10)


@pytest.mark.parametrize('complex_matrix', [False, True])
def test_large_nondiagonally_dominant_metric_ritz(complex_matrix):
    """Catch an incompatible real starting buffer for complex sparse Lanczos."""
    from kubo_msd.backends.gpu_context import create_gpu_context
    from kubo_msd.metric.gpu_setup import spd_bounds_gpu
    rng = np.random.default_rng(9022)
    a = rng.normal(size=(160, 160))
    if complex_matrix:
        a = a + 1j*rng.normal(size=a.shape)
    s = .3*np.eye(160) + a.conj().T@a/160
    assert np.min(s.diagonal().real - (np.abs(s).sum(axis=1)
                                      - np.abs(s.diagonal()))) < 0
    expected = la.eigvalsh(s)[0]
    ctx = create_gpu_context(0, 512*1024**2)
    with ctx.activate():
        lo, hi, report = spd_bounds_gpu(ctx.matrix(sp.csr_matrix(s), 'S'), ctx)
    assert 0 < lo <= expected*(1+1e-11)
    np.testing.assert_allclose(lo, expected, rtol=2e-9)
    assert hi >= la.eigvalsh(s)[-1]
    assert report['residual'] < 1e-10
