"""Every strategy sees the same full metric basis and independent eigen reference."""
from dataclasses import replace
import json
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.metric.partition import make_spatial_partition,model_identity
from kubo_msd.validation import matrix_fingerprint
from kubo_msd.operators.velocity import build_operators
from kubo_msd.solvers.stochastic import iter_stochastic_blocks
from benchmarks.metric_eigen_reference import eigen_observables,expected_arrays,differences


@pytest.mark.parametrize("method",["spectral","stepwise"])
@pytest.mark.parametrize("strategy",["adaptive","cholesky","cluster-jacobi","ic0"])
def test_full_metric_basis_matches_independent_reference(tmp_path,method,strategy):
    S=sp.csr_matrix([[2,.1j,0],[-.1j,1,.2],[0,.2,1.5]])
    H=sp.csr_matrix([[-.5,.2,.1j],[.2,.3,0],[-.1j,0,.8]])
    V=sp.csr_matrix([[.2,.3j,.1],[-.3j,.4,.2],[.1,.2,-.1]])
    model=ModelBundle(H,S,velocity={"x":V},orbital_order=["0","1","2"],
                      atom_slices=[(0,1),(1,2),(2,3)],provenance={"operator_cache":{"physical_key":"fixture"}})
    options={}
    if strategy=="adaptive":options["metric_residual_policy"]="adaptive"
    elif strategy=="cholesky":options["metric_solver"]="cholesky"
    else:options["preconditioner"]=strategy
    if strategy=="cluster-jacobi":
        partition=make_spatial_partition(dict(cell_A=np.eye(3)*10,positions_A=[[0,0,0],[4,0,0],[.1,0,0]],symbols=["A"]*3),
            model.atom_slices,max_orbitals=2,s_fingerprint=matrix_fingerprint(S),**model_identity(model))
        path=tmp_path/"partition.json";path.write_text(json.dumps(partition))
        options["metric_partition_file"]=str(path)
    cfg=RunConfig([-.2,.1,.3],[0,.02,.05],solver="stochastic-kpm",kernel="jackson",
                  axes=("x",),spectral_bounds=(-2,2),n_moments=16,nsamples=3,sample_batch=3,
                  statistics_block=3,metric_rtol=2e-13,integration_dt_fs=.001,
                  stochastic_method=method,**options)
    e,C=la.eigh(H.toarray(),S.toarray());B=np.sqrt(3)*C
    ops=build_operators(model,cfg)
    blocks=list(iter_stochastic_blocks(ops,cfg,method,samples=B))
    actual={**blocks[0].static,**blocks[0].arrays}
    dos,nums=eigen_observables(e,np.sqrt(3)*np.eye(3),lambda _:C.conj().T@V@C,cfg)
    expected=expected_arrays(dos,nums,cfg)
    checks=differences(actual,expected,rtol=3e-10,atol=2e-12)
    assert all(c["passed"] for c in checks.values()),checks
    assert ops.metric.native is not None
    if method=="stepwise":
        from kubo_msd.solvers.exact import iter_exact_blocks
        exact=replace(cfg,solver="exact",metric_solver="pcg",metric_residual_policy="fixed",
                      preconditioner="diagonal",metric_partition_file=None)
        reference=list(iter_exact_blocks(model,build_operators(model,exact),exact))[0]
        np.testing.assert_allclose(actual["msd_A2/commutator"],reference.arrays["msd_A2/pair_analytic"],
                                   rtol=3e-10,atol=2e-12)


def test_native_direct_hot_loop_cannot_callback_python_solve(monkeypatch):
    model=ModelBundle(np.diag([-.5,.5]),np.array([[1,.1j],[-.1j,1]]),
                      velocity={"x":np.eye(2)})
    cfg=RunConfig([0],[0,.1],solver="stochastic-kpm",kernel="jackson",
                  metric_solver="cholesky",axes=("x",),spectral_bounds=(-2,2))
    ops=build_operators(model,cfg)
    def forbidden(*a,**kw):raise AssertionError("native hot loop called Python metric solve")
    monkeypatch.setattr(ops.metric,"solve_s",forbidden)
    from kubo_msd.solvers.stepwise import propagate
    b=np.eye(2,dtype=complex)
    out,_=propagate(b,.05,ops,cfg)
    e,C=la.eigh(model.H_cov,model.S)
    want=C@(np.exp(-1j*e*.05/.6582119569509067)[:,None]*(C.conj().T@model.S@b))
    np.testing.assert_allclose(out,want,atol=1e-12,rtol=1e-12)
