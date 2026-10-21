"""Energy grids and all large spectral contractions remain on device."""
from .energy import jackson_coefficients
from ..errors import NumericalError


def reconstruction_matrix_gpu(E, bounds, L, context):
    xp = context.xp
    c,a = (bounds[0]+bounds[1])/2,(bounds[1]-bounds[0])/2
    x = (context.asarray(E,'energy_grid')-c)/a
    if a<=0 or not bool(xp.all(xp.isfinite(x))) or bool(xp.any(xp.abs(x)>=1)):
        raise NumericalError('Jackson energies must lie strictly inside full spectral bounds')
    g = jackson_coefficients(L)
    g[1:] *= 2
    g = context.asarray(g,'Jackson_coefficients')
    return xp.cos(xp.arccos(x)[:,None]*xp.arange(L))*g/(xp.pi*a*xp.sqrt(1-x*x)[:,None])


def energy_weights_gpu(E, eps, config, context):
    xp, cfg = context.xp, config
    context.check_device(eps)
    if cfg.kernel == 'gaussian':
        en = context.asarray(E,'energy_grid')
        return xp.exp(-.5*((en[:,None]-eps[None,:])/cfg.sigma_eV)**2)/(xp.sqrt(2*xp.pi)*cfg.sigma_eV)
    if cfg.spectral_bounds is None: raise NumericalError('Jackson needs full spectral bounds')
    lo,hi = cfg.spectral_bounds
    if bool(xp.any(eps<lo)) or bool(xp.any(eps>hi)):
        raise NumericalError('full eigenspectrum outside spectral bounds')
    c,a = (hi+lo)/2,(hi-lo)/2
    G = reconstruction_matrix_gpu(E,cfg.spectral_bounds,cfg.n_moments,context)
    return G@xp.cos(xp.arange(cfg.n_moments)[:,None]*xp.arccos((eps-c)/a)[None,:])
