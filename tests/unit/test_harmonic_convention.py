"""External H5 basis contract, independently frozen from dftio's exporter."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from kubo_msd.operators.lcao_integrals import OrbitalSet, real_harmonics


def orbital(ls):
    r = np.linspace(0, 4, 401)
    return OrbitalSet('X', r, [(l, z, np.exp(-r*r)) for z, l in enumerate(ls)], [], {})


@pytest.mark.parametrize('l,order,sign', [
    (0, [0], [1]),
    (1, [2, 0, 1], [-1, 1, -1]),
    (2, [4, 2, 0, 1, 3], [1, -1, 1, -1, 1]),
    (3, [6, 4, 2, 0, 1, 3, 5], [-1, 1, -1, 1, -1, 1, -1]),
])
def test_exporter_mapping(l, order, sign):
    from kubo_msd.operators.basis import basis_map
    p, s = basis_map(orbital([l]), 'deeptb-real')
    np.testing.assert_array_equal(p, order)
    np.testing.assert_array_equal(s, sign)


def test_deeptb_p_harmonics_cartesian_axes():
    from kubo_msd.operators.basis import basis_map
    p, s = basis_map(orbital([1]), 'deeptb-real')
    # DeePTB p order is y,z,x, with positive Cartesian functions.
    expected = np.sqrt(3/(4*np.pi))*np.array([[0,0,1], [1,0,0], [0,1,0]])
    assert_allclose(real_harmonics(1, np.eye(3))[:, p]*s, expected, atol=1e-15)


def test_multi_zeta_rectangular_complex_operator():
    from kubo_msd.operators.basis import basis_map, transform_block
    a, b = orbital([0,1,1,3]), orbital([0,2])
    pa, sa = basis_map(a, 'deeptb-real')
    pb, sb = basis_map(b, 'deeptb-real')
    Ua = np.eye(a.n_orbitals)[pa]*sa[:,None]
    Ub = np.eye(b.n_orbitals)[pb]*sb[:,None]
    rng = np.random.default_rng(70)
    x = rng.normal(size=(3, a.n_orbitals,b.n_orbitals))+1j*rng.normal(size=(3,a.n_orbitals,b.n_orbitals))
    assert_allclose(transform_block(x,(pa,sa),(pb,sb)), Ua@x@Ub.T)
    np.testing.assert_array_equal(transform_block(x,basis_map(a,'abacus-real'),basis_map(b,'abacus-real')),x)


def test_cache_key_separates_conventions():
    from kubo_msd.operators.builder import BuilderOptions
    from kubo_msd.operators.cache import material_key
    o = {'X': orbital([0,1])}
    s = dict(symbols=['X'],positions_A=[[0,0,0]],cell_A=np.eye(3)*10)
    h = {(0,0,0,0,0): np.eye(4)}
    assert material_key(s,o,h,BuilderOptions()) != material_key(
        {**s,'orbital_convention':'deeptb-real'},o,h,BuilderOptions())
