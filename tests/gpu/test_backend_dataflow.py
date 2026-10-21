import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.backends.gpu import GPUBackend
from kubo_msd.operators.velocity import OperatorBundle

pytestmark=pytest.mark.gpu


def fixture(preconditioner='diagonal'):
    n=4;S=sp.csr_matrix(np.eye(n)+.02*np.ones((n,n)),dtype=complex);H=sp.diags([-1.,-.2,.5,1.1],dtype=complex)
    cfg=RunConfig([-1,0,1],[0,.1],backend='gpu',axes=('x',),preconditioner=preconditioner)
    model=ModelBundle(H,S,position={'x':sp.diags(np.arange(n),dtype=complex)},atom_slices=[(0,2),(2,4)])
    return model,cfg


def test_gpu_exact_uses_device_csr(monkeypatch):
    from kubo_msd.solvers.exact import solve_generalized
    m,c=fixture();ops=OperatorBundle(m,c,GPUBackend());device=ops.backend
    uploads=device.upload_bytes
    def forbidden(*a,**kw):raise AssertionError('host dense/eigh during GPU eigensolve')
    import scipy.linalg as la
    monkeypatch.setattr(sp.csr_matrix,'toarray',forbidden);monkeypatch.setattr(la,'eigh',forbidden)
    eps,C,report=solve_generalized(ops.H,ops.S,'gpu',device_backend=device)
    assert device.upload_bytes==uploads
    assert report['metric_orthogonality']<1e-10
    assert report['max_normalized_residual']<1e-10
    assert report['device_dense_seconds']>=0


@pytest.mark.parametrize('preconditioner',['diagonal','block-jacobi'])
def test_shared_s_without_repeat_upload_and_sampler_without_get(monkeypatch,preconditioner):
    from kubo_msd.metric.sampling import MetricSampler
    from cupyx.scipy.sparse import csr_matrix
    m,c=fixture(preconditioner);ops=OperatorBundle(m,c,GPUBackend())
    for part in ('data','indices','indptr'):
        assert getattr(ops.S,part).data.ptr==getattr(ops.metric.S,part).data.ptr
    def forbidden(*args,**kwargs):raise AssertionError('entire S downloaded')
    monkeypatch.setattr(csr_matrix,'get',forbidden)
    sampler=MetricSampler(ops.metric,c)
    sampler.transform(np.eye(4,dtype=complex))
    uploads=[r for r in ops.backend.transfers if r['object']=='S' and r['direction']=='H2D']
    assert len(uploads)==1
    assert sum(r['bytes'] for r in ops.backend.transfers if r['direction']=='H2D')==ops.backend.upload_bytes


def test_host_setup_invalidated_by_matrix_or_config():
    from kubo_msd.metric.setup import prepare_host_metric
    from kubo_msd.metric.solve import MetricSolver
    from dataclasses import replace
    m,c=fixture();setup=prepare_host_metric(m.S,c,m.atom_slices)
    m.S.data[0]+=.2
    with pytest.raises(ValueError,match='setup'):MetricSolver(m.S,c,host_setup=setup,atom_slices=m.atom_slices)
    m,c=fixture()
    with pytest.raises(ValueError,match='setup'):MetricSolver(m.S,replace(c,sampling_tol=c.sampling_tol/10),host_setup=setup,atom_slices=m.atom_slices)


def test_cpu_hybrid_keep_scipy_eigh(monkeypatch):
    from kubo_msd.solvers.exact import solve_generalized
    import scipy.linalg as la
    m,c=fixture();original=la.eigh;calls=[]
    def spy(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(la,'eigh',spy)
    for backend in ('cpu','hybrid'):solve_generalized(m.H_cov,m.S,backend)
    assert len(calls)==2


def test_zero_copy_reference_neither_transfers_nor_synchronizes():
    b=GPUBackend();x=b.asarray(np.arange(8,dtype=float));before=(b.upload_bytes,b.device_copy_bytes,b.sync_count)
    assert b.asarray(x) is x
    assert (b.upload_bytes,b.device_copy_bytes,b.sync_count)==before
