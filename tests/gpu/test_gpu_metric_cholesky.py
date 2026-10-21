import numpy as np
import pytest
import scipy.sparse as sp
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


@pytest.mark.gpu
def test_device_cholesky_factor_once_no_dense_upload_or_rhs_download(gpu, monkeypatch):
    from cupyx.scipy.sparse import csr_matrix
    original=np.array([[10, 2j], [-2j, 1]],complex)
    S = sp.csr_matrix(original)
    cfg = RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                    metric_solver="cholesky",metric_rtol=1e-13)
    calls=[]; chol=gpu.xp.linalg.cholesky
    def record(a):
        assert isinstance(a,gpu.xp.ndarray)
        calls.append(1)
        return chol(a)
    monkeypatch.setattr(gpu.xp.linalg,"cholesky",record)
    def no_host_dense(*a,**kw):raise AssertionError("GPU direct route made a host dense S")
    monkeypatch.setattr(sp.csr_matrix,"toarray",no_host_dense)
    m=MetricSolver(S,cfg,gpu)
    pointer=m.S.data.data.ptr
    before=gpu.download_bytes
    B=gpu.asarray(np.ones((2,2),complex))
    for _ in range(3):
        Y,r=m.solve_s(B)
        assert isinstance(Y,gpu.xp.ndarray)
        assert bool(gpu.xp.all(r.converged))
    assert gpu.download_bytes==before
    assert m.S.data.data.ptr==pointer and len(calls)==1
    assert not any(t["direction"]=="H2D" and t["shape"]==[2,2] and "factor" in t["object"] for t in gpu.transfers)
    np.testing.assert_allclose(Y.get(),np.linalg.solve(original,np.ones((2,2))),atol=1e-12)
