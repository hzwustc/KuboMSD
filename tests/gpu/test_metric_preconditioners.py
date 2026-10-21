import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


@pytest.mark.gpu
@pytest.mark.parametrize("pre",["block-jacobi","ic0"])
def test_gpu_fixed_preconditioner_triangular_reuse(gpu,monkeypatch,pre):
    S=np.array([[2,.2j,0],[-.2j,1,.1j],[0,-.1j,1.5]])
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  preconditioner=pre,metric_rtol=1e-13)
    def forbidden(*a,**kw):raise AssertionError("generic solve refactorizes triangular factor")
    monkeypatch.setattr(gpu.xp.linalg,"solve",forbidden)
    m=MetricSolver(S,cfg,gpu,atom_slices=[(0,2),(2,3)])
    B=gpu.asarray(np.ones((3,2),complex))
    before=(gpu.upload_bytes,gpu.download_bytes)
    for _ in range(3):
        Y,r=m.solve_s(B)
        assert bool(gpu.xp.all(r.converged))
    assert (gpu.upload_bytes,gpu.download_bytes)==before
    np.testing.assert_allclose(Y.get(),np.linalg.solve(S,np.ones((3,2))),rtol=1e-12,atol=1e-12)


@pytest.mark.gpu
def test_gpu_noncontiguous_spatial_cluster_and_extreme_rhs(gpu,tmp_path,monkeypatch):
    import json
    import scipy.sparse as sp
    from kubo_msd.metric.partition import make_spatial_partition
    from kubo_msd.validation import matrix_fingerprint
    S=sp.csr_matrix([[2,.1j,.4],[-.1j,1,.1j],[.4,-.1j,1.5]])
    slices=[(0,1),(1,2),(2,3)]
    identity=dict(physical_key="p",orbital_order_hash="o")
    sidecar=make_spatial_partition(
        dict(cell_A=np.eye(3)*10,positions_A=[[0,0,0],[5,0,0],[.2,0,0]],symbols=["A"]*3),
        slices,max_orbitals=2,s_fingerprint=matrix_fingerprint(S),**identity)
    assert sidecar["blocks"][0]["orbitals"] == [0,2]
    path=tmp_path/"partition.json";path.write_text(json.dumps(sidecar))
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  preconditioner="cluster-jacobi",metric_partition_file=str(path),
                  metric_rtol=1e-13,metric_gpu_check_batch=16)
    def forbidden(*a,**kw):raise AssertionError("generic solve is not triangular reuse")
    monkeypatch.setattr(gpu.xp.linalg,"solve",forbidden)
    m=MetricSolver(S,cfg,gpu,atom_slices=slices,metric_identity=identity)
    B=np.array([[1j],[2],[3j]])*np.array([0,1,1e-180,1e180])
    device=gpu.asarray(B);before=(gpu.upload_bytes,gpu.download_bytes)
    Y,r=m.solve_s(device)
    assert (gpu.upload_bytes,gpu.download_bytes)==before
    assert bool(gpu.xp.all(r.converged))
    actual=Y.get()
    for j in (1,2,3):
        scale=np.max(abs(B[:,j]))
        np.testing.assert_allclose(actual[:,j]/scale,
            np.linalg.solve(S.toarray(),B[:,j]/scale),rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(actual[:,0],0)


@pytest.mark.gpu
def test_gpu_block_preconditioner_batches_equal_sizes_without_per_block_calls(gpu,monkeypatch):
    import scipy.linalg as la
    import cupyx.scipy.linalg as device_la
    rng=np.random.default_rng(318)
    sizes=[2,3,2,3,2,3,2,3,5]
    n=sum(sizes)
    A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    S=A@A.conj().T+np.eye(n)
    edges=np.r_[0,np.cumsum(sizes)]
    slices=list(zip(edges[:-1],edges[1:]))
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
                  preconditioner="block-jacobi",metric_rtol=1e-13)
    m=MetricSolver(S,cfg,gpu,atom_slices=slices)
    B=(rng.normal(size=(n,8))+1j*rng.normal(size=(n,8)))
    device=gpu.asarray(B)[:,::2]
    assert not device.flags.c_contiguous
    def forbidden(*a,**kw):
        raise AssertionError("per-block triangular submissions violate grouped execution")
    monkeypatch.setattr(device_la,"solve_triangular",forbidden)
    before=(gpu.upload_bytes,gpu.download_bytes)
    first=m.precondition(device)
    second=m.precondition(-device)
    expected=np.empty_like(B[:,::2])
    for a,b in slices:
        expected[a:b]=la.solve(S[a:b,a:b],B[a:b,::2],assume_a="pos")
    np.testing.assert_allclose(first.get(),expected,rtol=3e-13,atol=2e-14)
    np.testing.assert_allclose(second.get(),-expected,rtol=3e-13,atol=2e-14)
    assert (gpu.upload_bytes,gpu.download_bytes)==before
    report=m.diagnostics()["grouped_block_triangular"]
    assert report["groups"]=={"2":4,"3":4,"5":1}
    assert report["applications"]==2
    assert report["kernel_launches"]==6


@pytest.mark.gpu
def test_grouped_triangular_strided_rows_indices_and_workspace(gpu):
    import scipy.linalg as la
    from kubo_msd.metric.gpu_kernels import DeviceBlocks
    rng=np.random.default_rng(731)
    sizes=[1,13,25,64,128,256,257]*2
    n=sum(sizes);order=rng.permutation(n);entries=[];start=0
    for size in sizes:
        A=rng.normal(size=(size,size))+1j*rng.normal(size=(size,size))
        factor=la.cholesky(A@A.conj().T+size*np.eye(size),lower=True)
        entries.append((order[start:start+size],factor));start+=size
    device=DeviceBlocks(entries,gpu)
    for width in (4,1,7):
        B=rng.normal(size=(n,width))+1j*rng.normal(size=(n,width))
        scales=np.resize(np.array([0,1e-180,1e180,1.]),width)
        B*=scales
        expected=np.empty_like(B)
        for indices,L in entries:
            expected[indices]=la.cho_solve((L,True),B[indices])
        rhs=gpu.asarray(B);workspace=gpu.xp.empty_like(rhs)
        before=(gpu.upload_bytes,gpu.download_bytes)
        Y=device.apply(rhs,out=workspace)
        assert Y is workspace
        actual=Y.get()
        np.testing.assert_allclose(actual/np.where(scales==0,1,scales),
                                   expected/np.where(scales==0,1,scales),
                                   rtol=3e-11,atol=2e-12)
        np.testing.assert_array_equal(actual[:,scales==0],0)
        assert (gpu.upload_bytes,gpu.download_bytes)==before
    assert device.diagnostics()["kernel_launches"]==21


@pytest.mark.gpu
def test_gpu_multiple_noncontiguous_clusters_use_grouped_solve(gpu,tmp_path,monkeypatch):
    import json
    import cupyx.scipy.linalg as device_la
    from kubo_msd.metric.partition import make_spatial_partition
    from kubo_msd.validation import matrix_fingerprint
    rng=np.random.default_rng(512);n=6
    A=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
    S=A@A.conj().T+10*np.eye(n)
    slices=[(i,i+1) for i in range(n)]
    identity=dict(physical_key="grouped",orbital_order_hash="six")
    sidecar=make_spatial_partition(dict(cell_A=np.eye(3)*30,
        positions_A=[[0,0,0],[10,0,0],[20,0,0],[.1,0,0],[10.1,0,0],[20.1,0,0]],
        symbols=["A"]*n),slices,max_orbitals=2,
        s_fingerprint=matrix_fingerprint(S),**identity)
    assert [b["orbitals"] for b in sidecar["blocks"]]==[[0,3],[1,4],[2,5]]
    path=tmp_path/"groups.json";path.write_text(json.dumps(sidecar))
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",backend="gpu",
        preconditioner="cluster-jacobi",metric_partition_file=str(path),
        metric_rtol=1e-13,metric_residual_policy="adaptive",metric_gpu_check_batch=16)
    m=MetricSolver(S,cfg,gpu,atom_slices=slices,metric_identity=identity)
    def forbidden(*a,**kw):raise AssertionError("per-block triangular call")
    monkeypatch.setattr(device_la,"solve_triangular",forbidden)
    B=rng.normal(size=(n,3))+1j*rng.normal(size=(n,3));B[:,0]=0
    Y,report=m.solve_s(gpu.asarray(B))
    np.testing.assert_allclose(Y.get(),np.linalg.solve(S,B),rtol=3e-11,atol=2e-12)
    assert bool(gpu.xp.all(report.converged))
    diag=m.diagnostics()["grouped_block_triangular"]
    assert diag["groups"]=={"2":3}
    assert diag["kernel_launches"]==diag["applications"]


@pytest.mark.gpu
def test_grouped_triangular_rejects_overlapping_output(gpu):
    from kubo_msd.metric.gpu_kernels import DeviceBlocks
    device=DeviceBlocks([(np.arange(257),np.eye(257,dtype=complex))],gpu)
    base=gpu.asarray(np.arange(258*2).reshape(258,2),dtype=gpu.xp.complex128)
    with pytest.raises(ValueError,match="overlap"):
        device.apply(base[:-1],out=base[1:])
    with pytest.raises(ValueError,match="overlap"):
        device.apply(base[:-1],out=base[:-1])


@pytest.mark.gpu
@pytest.mark.parametrize("shape",[(2,2),(4,2),(3,),(3,0),(3,2,1)])
def test_grouped_triangular_rejects_bad_rhs_before_kernel(gpu,monkeypatch,shape):
    from kubo_msd.metric.gpu_kernels import DeviceBlocks
    device=DeviceBlocks([(np.arange(3),np.eye(3,dtype=complex))],gpu)
    def unsafe(*a,**kw):raise AssertionError("invalid shape reached raw CUDA kernel")
    monkeypatch.setattr(device,"kernel",unsafe)
    with pytest.raises(ValueError,match="shape"):
        device.apply(gpu.xp.ones(shape,dtype=gpu.xp.complex128))
