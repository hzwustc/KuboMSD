import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_pair_reference import gaussian


def test_native_bad_table_ranges_rejected_without_allocation():
    from kubo_msd.operators.tables import prepare_tables
    for settings in ({'distance_step':1e-300},{'dk':1e-300},{'kmax':float('inf')}):
        with pytest.raises((ValueError,RuntimeError)):prepare_tables({'X':gaussian()},settings=settings,memory_limit_bytes=2**20)


def test_reference_builder_does_not_require_native(tmp_path,monkeypatch):
    import kubo_msd.operators.builder as b
    def missing():raise ValueError('extension unavailable')
    monkeypatch.setattr(b,'native_module',missing)
    o=gaussian();s=dict(symbols=['X'],positions_A=np.zeros((1,3)),cell_A=np.eye(3)*20)
    result=b.build_operators(s,{'X':o},{(0,0,0,0,0):np.array([[1.]])},b.BuilderOptions(operator_builder='reference'))
    assert_allclose(result.S.toarray(),[[1.]],atol=1e-7)


def test_rotation_covariance_all_pdf():
    from kubo_msd.operators.tables import prepare_tables,native_module
    from kubo_msd.operators.lcao_integrals import real_harmonics
    from scipy.spatial.transform import Rotation
    from scipy.linalg import block_diag
    o=gaussian(lmax=3);table=prepare_tables({'X':o});n=o.n_orbitals
    Q=Rotation.from_rotvec([.2,-.4,.3]).as_matrix();v=np.random.default_rng(11).normal(size=(100,3));blocks=[]
    for l in range(4):blocks.append(np.linalg.lstsq(real_harmonics(l,v),real_harmonics(l,v@Q.T),rcond=None)[0].T)
    U=block_diag(*blocks);d=np.array([.29,.43,-.17])
    def calc(d):
        r=native_module().evaluate_pairs(table,dict(species=np.array([[0,0]],np.int64),displacement_A=d[None,:],pair_ids=np.array([0],np.int64)),1)
        return r['S'].reshape(n,n),r['D'].reshape(3,n,n)
    a,b=calc(d);ar,br=calc(Q@d)
    assert_allclose(ar,U@a@U.T,atol=2e-12)
    expected=np.einsum('ab,bij->aij',Q,np.array([U@x@U.T for x in b]))
    assert_allclose(br,expected,atol=2e-12)
