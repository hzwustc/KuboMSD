import json
import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import block_diag, eigvalsh
from kubo_msd.operators.builder import BuilderOptions, build_operators
from kubo_msd.io.legacy_input import parse_legacy, load_legacy


def fixture(root):
    from kubo_msd.operators.lcao_integrals import OrbitalSet
    r = np.arange(801)*.01
    s = 2/np.pi**.25*np.exp(-r*r/2)
    p = s*r*np.sqrt(2/3)
    o = OrbitalSet('X',r,[(0,0,s),(1,0,p)],[1,1],{})
    text = 'Element X\nMesh 801\ndr .01\n'
    for l,values in [(0,s),(1,p)]:
        text += f'Type L N\n0 {l} 0\n'+'\n'.join(map(str,values))+'\n'
    (root/'X.orb').write_text(text)
    (root/'POSCAR').write_text('sp\n1\n20 0 0\n0 20 0\n0 0 20\nX\n2\nCartesian\n0 0 0\n1.2 .3 -.2\n')
    rng = np.random.default_rng(713)
    H = rng.normal(size=(8,8));H = (H+H.T)/2
    u = np.eye(4)[[0,3,1,2]]*np.array([1,-1,1,-1])[:,None]
    U = block_diag(u,u)
    Hd = U@H@U.T
    h = {}
    with h5py.File(root/'H.h5','w') as f:
        g = f.create_group('0')
        for i,j in [(0,0),(0,1),(1,1)]:
            h[i,j,0,0,0] = H[4*i:4*i+4,4*j:4*j+4]
            g[f'{i}_{j}_0_0_0'] = Hd[4*i:4*i+4,4*j:4*j+4]
    data = dict(structure={'file':'POSCAR'},model=dict(type='lcao',
        orbitals=dict(names=['X.orb']),hamiltonian=dict(path='H.h5',format='deeptb')))
    path = root/'input.json';path.write_text(json.dumps(data))
    structure = dict(cell_A=np.eye(3)*20,positions_A=np.array([[0,0,0],[1.2,.3,-.2]]),symbols=['X','X'])
    return path,data,structure,{'X':o},h,U,Hd


@pytest.mark.parametrize('fast',[False,True])
def test_legacy_h5_and_all_integrals_share_basis(tmp_path,fast):
    path,data,s,o,h,U,Hd = fixture(tmp_path)
    parsed = parse_legacy(path,data)
    assert parsed[0]['orbital_convention']=='deeptb-real'
    assert parsed[3]['orbital_convention']=='deeptb-real'
    original = build_operators(s,o,h,BuilderOptions())
    actual = load_legacy(path,data,builder_options=BuilderOptions() if fast else None)
    assert_allclose(actual.H_cov.toarray(),Hd,atol=0)
    for a,b in [(actual.S,original.S)]+[
        (a,b) for axis in 'xyz' for a,b in zip(actual.composite[axis],original.composite[axis])]:
        assert_allclose(a.toarray(),U@b.toarray()@U.T,atol=1e-6)
    assert_allclose(eigvalsh(actual.H_cov.toarray(),actual.S.toarray()),
                    eigvalsh(original.H_cov.toarray(),original.S.toarray()),atol=1e-6)
    assert any('deeptb' in label for label in actual.orbital_order)


def test_exact_msd_invariant_under_input_basis_change(tmp_path):
    from kubo_msd.operators.cache import material_key,publish_model
    from kubo_msd.config import RunConfig
    from kubo_msd.dispatch import run_input
    from kubo_msd.io.results import read_results
    path,data,s,o,h,U,Hd = fixture(tmp_path)
    models=[]
    for name,parsed in [('abacus',(s,o,h,{})),('deeptb',parse_legacy(path,data))]:
        structure,orbs,images,prov=parsed
        opts=BuilderOptions()
        result=build_operators(structure,orbs,images,opts)
        model=publish_model(result,tmp_path/name,material_key(structure,orbs,images,opts),prov)
        cfg=RunConfig([-1,0,1],[0,.001,.03,.1],solver='exact',axes=('x','y','z'),
                      kernel='gaussian',sigma_eV=.2,metric_rtol=1e-12)
        run_input(model,cfg,tmp_path/(name+'-result'))
        models.append(read_results(tmp_path/(name+'-result'))['arrays'])
    for key in ['eigenvalues_eV','msd_A2/pair_analytic','diffusion_A2_per_fs/pair_analytic']:
        assert_allclose(models[0][key],models[1][key],rtol=1e-9,atol=1e-10)


def test_pre_fix_operator_schema_rejected(tmp_path):
    from kubo_msd.operators.cache import verify_cache
    (tmp_path/'CACHE_READY').write_text(json.dumps(dict(schema='kubomsd-operators-v1',kind='model')))
    with pytest.raises(ValueError,match='schema'):
        verify_cache(tmp_path)
