import json
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

def write_model(p,sparse=False):
    H=np.array([[0,1j],[-1j,1.]])
    if sparse: sp.save_npz(p/'H.npz',sp.csr_matrix(H))
    else: np.savez(p/'H.npz',matrix=H)
    spec={'format':'kubomsd-model-v1','n_orbitals':2,'basis':'orthogonal','kpoint':[0,0,0],'energy_unit':'eV','length_unit':'angstrom','orbital_order':['a:s','b:s'],'hamiltonian':{'path':'H.npz','storage':'csr' if sparse else 'dense','shape':[2,2],'dtype':'complex128','index_base':0,'representation':'covariant'},'velocity_source':{'type':'position','coordinates':[[0,0,0],[1,0,0]]}}
    (p/'model.json').write_text(json.dumps(spec)); return spec,H

@pytest.mark.parametrize('sparse',[False,True])
def test_native_matrix_io(tmp_path,sparse):
    from kubo_msd.io.model import load_model
    spec,H=write_model(tmp_path,sparse)
    m=load_model(tmp_path/'model.json','cpu')
    assert_allclose(m.H_cov.toarray(),H)
    assert_allclose(m.S.toarray(),np.eye(2))
    assert m.provenance['files'][0]['sha256']

@pytest.mark.parametrize('change',[{'basis':'nonorthogonal'},{'kpoint':[.1,0,0]},{'n_orbitals':3},{'energy_unit':'unknown'},{'orbital_order':['a','a']}])
def test_bad_models_fail(tmp_path,change):
    from kubo_msd.io.model import load_model
    spec,_=write_model(tmp_path); spec.update(change); (tmp_path/'model.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError): load_model(tmp_path/'model.json','cpu')

@pytest.mark.parametrize('storage',['dense','csr'])
def test_hdf5_native_roundtrip(tmp_path,storage):
    import h5py
    from kubo_msd.io.model import load_model
    spec,H=write_model(tmp_path)
    with h5py.File(tmp_path/'H.h5','w') as f:
        if storage=='dense': f['matrix']=H
        else:
            g=f.create_group('matrix'); C=sp.csr_matrix(H)
            for key,v in [('data',C.data),('indices',C.indices),('indptr',C.indptr),('shape',C.shape)]: g[key]=v
    spec['hamiltonian'].update(path='H.h5',storage=storage)
    (tmp_path/'model.json').write_text(json.dumps(spec))
    assert_allclose(load_model(tmp_path/'model.json').H_cov.toarray(),H)
