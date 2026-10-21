"""A narrow output window must not redefine the Hamiltonian spectral scale."""
import numpy as np
from kubo_msd.config import RunConfig
from kubo_msd.kernels.energy import energy_weights


def test_output_window_keeps_full_spectral_scale():
    energy = np.linspace(-0.5, 0.5, 11)
    eigenvalues = np.array([-8., -2., 0., 2., 15.])
    cfg = RunConfig(energy, [0., 0.1], kernel="jackson", n_moments=64,
                    spectral_bounds=(-10., 20.))
    # Independent cosine-series evaluation, without the product recurrence.
    center, radius = 5., 15.
    order = np.arange(cfg.n_moments)
    angle = np.pi / (cfg.n_moments + 1)
    damping = ((cfg.n_moments + 1 - order) * np.cos(order * angle)
               + np.sin(order * angle) / np.tan(angle)) / (cfg.n_moments + 1)
    x = (energy - center) / radius
    y = (eigenvalues - center) / radius
    expected = ((np.cos(np.arccos(x)[:, None] * order)
                 * damping * np.where(order == 0, 1, 2))
                @ np.cos(order[:, None] * np.arccos(y)[None, :]))
    expected /= (np.pi * radius * np.sqrt(1 - x*x))[:, None]
    actual = energy_weights(energy, eigenvalues, cfg)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert not np.allclose(actual[:, 0], actual[:, 1])
