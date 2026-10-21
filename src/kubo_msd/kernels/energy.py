"""Finite Gaussian/Jackson kernels; never infer spectral scale from outputs."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from ..errors import KuboMSDError

def validate_spectral_window(eps,bounds):
    e=np.asarray(eps,float); b=np.asarray(bounds,float)
    if b.shape!=(2,) or not np.isfinite(b).all() or b[1]<=b[0]: raise KuboMSDError('invalid spectral bounds')
    if not np.isfinite(e).all() or np.any(e<b[0]) or np.any(e>b[1]): raise KuboMSDError('spectral bounds do not contain full spectrum')
    return float((b[0]+b[1])/2),float((b[1]-b[0])/2)

def jackson_coefficients(n_moments):
    L=int(n_moments)
    if L!=n_moments or L<1: raise KuboMSDError('n_moments must be positive integer')
    m=np.arange(L); theta=np.pi/(L+1)
    g=((L+1-m)*np.cos(m*theta)+np.sin(m*theta)/np.tan(theta))/(L+1)
    g[0]=1.
    return g

def reconstruction_matrix(E,bounds,L):
    c,a=validate_spectral_window([],bounds); E=np.asarray(E,float); x=(E-c)/a
    if not np.isfinite(x).all() or np.any(np.abs(x)>=1): raise KuboMSDError('Jackson output energies must lie strictly inside full spectral bounds')
    g=jackson_coefficients(L); g[1:]*=2
    return np.cos(np.arccos(x)[:,None]*np.arange(L))*g/(np.pi*a*np.sqrt(1-x*x)[:,None])

def energy_weights(E,eps,kernel_config):
    cfg=kernel_config; E=np.asarray(E,float); eps=np.asarray(eps,float)
    if not np.isfinite(E).all() or not np.isfinite(eps).all(): raise KuboMSDError('nonfinite energy')
    if cfg.kernel=='gaussian':
        return np.exp(-.5*((E[:,None]-eps[None,:])/cfg.sigma_eV)**2)/(np.sqrt(2*np.pi)*cfg.sigma_eV)
    if cfg.spectral_bounds is None: raise KuboMSDError('Jackson requires resolved full spectral bounds')
    c,a=validate_spectral_window(eps,cfg.spectral_bounds)
    return reconstruction_matrix(E,cfg.spectral_bounds,cfg.n_moments) @ np.cos(np.arange(cfg.n_moments)[:,None]*np.arccos((eps-c)/a)[None,:])

def dos_floor(n_orbitals,width,cfg):
    w=cfg.sigma_eV if cfg.kernel=='gaussian' else (cfg.spectral_bounds[1]-cfg.spectral_bounds[0])/(2*cfg.n_moments)
    return max(cfg.rho_abs,cfg.rho_rel*n_orbitals/max(width,w))

def kernel_resolution(cfg):
    if cfg.kernel=='gaussian': return {'type':'constant_fwhm','fwhm_eV':float(2*np.sqrt(2*np.log(2))*cfg.sigma_eV)}
    # Endpoint-singular kernels may have no isolated local half-height roots.
    c,a=validate_spectral_window([],cfg.spectral_bounds); out=[]
    for E in cfg.energy_eV:
        width=8*np.pi*a/cfg.n_moments
        left=max(c-a+1e-10*a,E-width); right=min(c+a-1e-10*a,E+width)
        f=lambda x: float(energy_weights([x],[E],cfg)[0,0])
        peak=minimize_scalar(lambda x:-f(x),bounds=(left,right),method='bounded').x
        try:
            h=f(peak)/2
            lo=brentq(lambda x:f(x)-h,left,peak); hi=brentq(lambda x:f(x)-h,peak,right)
            out.append(float(hi-lo))
        except ValueError: out.append(None)
    return {'type':'local_jackson_fwhm','fwhm_eV':out,'null_reason':'no isolated half-height crossings in search bracket'}
