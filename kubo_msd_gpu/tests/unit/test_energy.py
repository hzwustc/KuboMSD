import numpy as np
import pytest
from numpy.testing import assert_allclose

def test_full_spectrum_is_independent_of_local_output():
    from kubo_msd.kernels.energy import energy_weights, validate_spectral_window
    from kubo_msd.config import RunConfig
    eps = np.array([-9., 0., 19.])
    assert validate_spectral_window(eps, (-10., 20.)) == (5., 15.)
    cfg = RunConfig(energy_eV=[-.5, 0, .5], time_fs=[0, 1], kernel='jackson', spectral_bounds=(-10, 20), n_moments=64)
    a = energy_weights(cfg.energy_eV, eps, cfg)
    b = energy_weights([-5., 0., 10.], eps, cfg)
    assert_allclose(a[1], b[1], rtol=1e-13)
    assert not np.allclose(a[:, 0], a[:, 2])

@pytest.mark.parametrize('L', [1, 2, 64, 1024])
def test_jackson_area_and_actual_term_count(L):
    from kubo_msd.kernels.energy import energy_weights, jackson_coefficients
    from kubo_msd.config import RunConfig
    g = jackson_coefficients(L)
    assert len(g) == L and g[0] == 1
    # Midpoint Gauss-Chebyshev quadrature cancels the endpoint Jacobian.
    theta = (np.arange(4096)+.5)*np.pi/4096
    E = np.cos(theta)[::-1]
    cfg = RunConfig(energy_eV=E, time_fs=[0], kernel='jackson', spectral_bounds=(-1, 1), n_moments=L)
    w = energy_weights(E, [-.37, .2], cfg)
    assert_allclose((w*np.sqrt(1-E[:, None]**2)).sum(0)*np.pi/4096, 1, atol=2e-12)

def test_gaussian_area_and_width():
    from kubo_msd.kernels.energy import energy_weights
    from kubo_msd.config import RunConfig
    E=np.linspace(-2,2,20001); s=.1
    cfg=RunConfig(energy_eV=E, time_fs=[0], sigma_eV=s)
    w=energy_weights(E,[0],cfg)[:,0]
    assert_allclose(np.trapz(w,E),1,atol=1e-12)
    assert_allclose(energy_weights([s*np.sqrt(2*np.log(2))],[0],cfg)[0,0]/w.max(),.5)

def test_no_spectral_clipping():
    from kubo_msd.kernels.energy import energy_weights, validate_spectral_window
    from kubo_msd.config import RunConfig
    with pytest.raises(ValueError): validate_spectral_window([-2,0],(-1,1))
    cfg=RunConfig(energy_eV=[0,1],time_fs=[0],kernel='jackson',spectral_bounds=(-1,1))
    with pytest.raises(ValueError): energy_weights([0,1],[0],cfg)
