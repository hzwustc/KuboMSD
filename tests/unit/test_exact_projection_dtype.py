from types import SimpleNamespace
import numpy as np
import pytest
import scipy.sparse as sp
from kubo_msd.solvers.exact import eigen_velocity_block


@pytest.mark.parametrize('imaginary',[0.,1e-30])
def test_real_projection_uses_real_blas_without_dropping_imaginary(imaginary):
    seen=[]
    sparse_seen=[]
    class Sparse(sp.csr_matrix):
        def __matmul__(self, other):
            sparse_seen.append(self.dtype.kind)
            return super().__matmul__(other)
    class Bra(np.ndarray):
        def __matmul__(self, other):
            seen.append(other.dtype.kind)
            return np.asarray(self)@other
    left=np.eye(3);right=np.eye(3)
    bra=left.T.view(Bra)
    value=np.diag([1.,2.,3.]).astype(complex)
    value[0,1]=1j*imaginary;value[1,0]=-1j*imaginary
    ops=SimpleNamespace(xp=np,directions={'x':(1,0,0)},velocity={'x':Sparse(value)})
    actual=eigen_velocity_block(ops,'x',left,right,np.arange(3),np.arange(3),left_adjoint=bra)
    np.testing.assert_array_equal(actual,value)
    assert seen==['f' if imaginary==0 else 'c']
    assert sparse_seen==['f' if imaginary==0 else 'c']


def test_real_eigenvectors_with_mixed_real_complex_composite():
    from kubo_msd.units import HBAR_EV_FS
    C=np.eye(3);eps=np.arange(3.)
    bond=np.zeros((3,3))
    xbar=np.zeros((3,3),complex);xbar[0,1]=1j*1e-30
    delta=xbar.conj().T
    ops=SimpleNamespace(xp=np,directions={'x':(1,0,0)},velocity={},position={},
                        composite={'x':tuple(sp.csr_matrix(x) for x in (bond,xbar,delta))})
    expected=1j*(-xbar*eps[None,:]+eps[:,None]*delta)/HBAR_EV_FS
    np.testing.assert_array_equal(eigen_velocity_block(ops,'x',C,C,eps,eps),expected)
