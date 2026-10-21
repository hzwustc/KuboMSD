import json
import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver
from kubo_msd.metric.partition import make_spatial_partition
from kubo_msd.validation import matrix_fingerprint
import scipy.sparse as sp


@pytest.mark.parametrize("kernel",["reference","cpp"])
def test_noncontiguous_cluster_solves_original_system_and_sidecar_replacement(tmp_path,kernel):
    S=sp.csr_matrix([[2,.1j,.4],[-.1j,1,.1j],[.4,-.1j,1.5]])
    slices=[(0,1),(1,2),(2,3)]
    structure=dict(cell_A=np.eye(3)*10,positions_A=[[0,0,0],[5,0,0],[.2,0,0]],symbols=["A"]*3)
    identity=dict(physical_key="p",orbital_order_hash="o")
    d=make_spatial_partition(structure,slices,max_orbitals=2,s_fingerprint=matrix_fingerprint(S),**identity)
    path=tmp_path/"p.json";path.write_text(json.dumps(d))
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",cpu_kernel=kernel,
                  preconditioner="cluster-jacobi",metric_partition_file=str(path),metric_rtol=1e-13)
    m=MetricSolver(S,cfg,atom_slices=slices,metric_identity=identity)
    B=np.array([[1j,0],[2,0],[3j,0]])
    Y,r=m.solve_s(B)
    np.testing.assert_allclose(Y,np.linalg.solve(S.toarray(),B),atol=1e-12)
    assert np.all(r.converged)
    assert m.diagnostics()["resolved_preconditioner"]=="cluster-jacobi"
    d["blocks"][0]["orbitals"]=[0,1]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        m.host_setup.validate(S,cfg,slices)
