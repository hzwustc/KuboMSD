import numpy as np
import h5py
import pytest
from numpy.testing import assert_allclose

def test_deeptb_complex_reverse_and_periodic_bond(tmp_path):
    from kubo_msd.io.deeptb import read_deeptb
    from kubo_msd.operators.assembly import assemble_images
    with h5py.File(tmp_path/'h.h5','w') as f:
        g=f.create_group('0'); g['0_0_0_0_0']=np.array([[0.]])
        g['1_1_0_0_0']=np.array([[1.]])
        g['0_1_-1_0_0']=np.array([[[2.,3.]]])
    images=read_deeptb(tmp_path/'h.h5',[1,1],index_base=0)
    H,bond,ket=assemble_images(images,[1,1],np.array([[.1,0,0],[9.9,0,0]]),np.diag([10.,10,10]))
    assert_allclose(H.toarray(),[[0,2+3j],[2-3j,1]])
    assert_allclose(bond[0].toarray()[0,1],-.2*(2+3j))

def test_conflicting_reverse_pair_rejected(tmp_path):
    from kubo_msd.io.deeptb import read_deeptb
    with h5py.File(tmp_path/'h.h5','w') as f:
        g=f.create_group('0'); g['0_1_0_0_0']=[[1.]]; g['1_0_0_0_0']=[[2.]]
    with pytest.raises(ValueError): read_deeptb(tmp_path/'h.h5',[1,1],index_base=0)
