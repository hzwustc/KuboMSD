import numpy as np
import scipy.sparse as sp
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver
from kubo_msd.metric.preconditioners import prepare_ic0, ic0_reference


@pytest.mark.parametrize("ordering",["natural","rcm"])
@pytest.mark.parametrize("kernel",["reference","cpp"])
def test_ic0_complex_fixed_spd_and_original_solution(ordering,kernel):
    S=sp.csr_matrix([[3,.2j,0,.3],[-.2j,2,.4j,0],[0,-.4j,1.5,.1],[.3,0,.1,2.]])
    before=S.copy()
    data=prepare_ic0(S,ordering=ordering,shift_max=1e-2)
    M=data.apply(np.eye(4,dtype=complex))
    np.testing.assert_allclose(M,M.conj().T,atol=1e-14)
    assert np.linalg.eigvalsh(M)[0]>0
    z=data.scaling;p=data.permutation
    W=sp.diags(z)@S@sp.diags(z);W=W[p][:,p].tocsr()
    ref=ic0_reference(W,data.report["shift"])
    np.testing.assert_allclose(data.factor.toarray(),ref.toarray(),atol=1e-14)
    cfg=RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",cpu_kernel=kernel,
                  preconditioner="ic0",metric_ic_ordering=ordering,metric_rtol=1e-13)
    m=MetricSolver(S,cfg)
    B=np.array([[1j,0],[2,0],[3j,0],[1,0]])
    Y,r=m.solve_s(B)
    np.testing.assert_allclose(Y,np.linalg.solve(S.toarray(),B),atol=1e-12)
    assert np.all(r.converged)
    np.testing.assert_array_equal(S.data,before.data)


def test_ic_failure_shift_ladder_and_indefinite_original_rejection():
    A=np.array([[0,-4,-2,0,0],[-4,0,0,0,3],[-2,0,0,2,-4],
                [0,0,2,0,0],[0,3,-4,0,0.]])
    S=sp.csr_matrix(np.eye(5)+.17*A)
    assert np.linalg.eigvalsh(S.toarray())[0]>.09
    with pytest.raises(ValueError,match="IC"):
        prepare_ic0(S,ordering="natural",shift_max=0)
    data=prepare_ic0(S,ordering="natural",shift_max=1e-2)
    assert data.report["shift"]==1e-2
    assert data.report["attempts"][0]["shift"]==0
    np.testing.assert_array_equal(S.toarray(),np.eye(5)+.17*A)
    with pytest.raises(ValueError,match="positive definite"):
        prepare_ic0(sp.csr_matrix([[1,2],[2,1.]]),ordering="natural",shift_max=1e-2)


@pytest.mark.parametrize("kw",[{"metric_ic_ordering":"bad"},{"metric_ic_shift_max":-1}])
def test_ic_invalid_config(kw):
    with pytest.raises(ValueError):
        RunConfig([0],[0],solver="stochastic-kpm",kernel="jackson",preconditioner="ic0",**kw)
