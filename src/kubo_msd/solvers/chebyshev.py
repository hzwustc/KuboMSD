"""Shared undamped Bessel propagation and finite Jackson moment operations."""
import numpy as np
from scipy.special import jv,gammaln
from ..units import HBAR_EV_FS
from ..errors import NumericalError
from ..kernels.energy import reconstruction_matrix

def scaling(cfg):
    if cfg.spectral_bounds is None: raise NumericalError('propagation needs resolved full spectral bounds')
    lo,hi=cfg.spectral_bounds
    return (lo+hi)/2,(hi-lo)/2

def scaled_action(ops,cfg,B):
    c,a=scaling(cfg)
    return (ops.apply_h_action(B)-c*B)/a

def bessel_coefficients(dt,cfg,weighted_tail=False):
    c,a=scaling(cfg); z=a*dt/HBAR_EV_FS; az=abs(z)
    if az==0: return np.array([1+0j]),{'order':0,'tail_bound':0.}
    chosen=None
    for k in range(max(1,int(np.ceil(az))),cfg.propagation_max_order+1):
        n=k+1; ratio=az/(2*(n+1))
        if ratio>=1: continue
        logbound=n*np.log(az/2)-gammaln(n+1)+az*az/(4*(n+1))
        bound=2*np.exp(min(logbound,700))/(1-ratio)
        if weighted_tail: bound*=4*(n+1)**2/(1-ratio)**2
        if bound<=cfg.propagation_tol: chosen=k; break
    if chosen is None: raise NumericalError('propagation Bessel tail not converged before max order')
    m=np.arange(chosen+1); b=jv(m,z)*(-1j)**m; b[1:]*=2; b*=np.exp(-1j*c*dt/HBAR_EV_FS)
    return b,{'order':chosen,'tail_bound':float(bound),'tail_policy':'absolute Bessel-series bound; weighted for commutator if requested'}

def evaluate_series(B,action,coeff,xp):
    prev=B; out=coeff[0]*B
    if len(coeff)==1: return out
    curr=action(B); out=out+coeff[1]*curr
    for v in coeff[2:]:
        nxt=2*action(curr)-prev; out=out+v*nxt; prev,curr=curr,nxt
    if not bool(xp.all(xp.isfinite(out))): raise NumericalError('nonfinite Chebyshev propagation; check bounds and solves')
    return out

def chain(B,ops,cfg,length):
    if ops.native is not None:
        c,a=scaling(cfg)
        return ops.native.chain(B,length,c,a)
    xp=ops.xp; out=xp.empty((length,)+B.shape,dtype=xp.complex128); out[0]=B
    if length>1: out[1]=scaled_action(ops,cfg,B)
    for j in range(2,length): out[j]=2*scaled_action(ops,cfg,out[j-1])-out[j-2]
    return out

def projected_quadratic(B,ops,cfg):
    xp=ops.xp; G=ops.backend.asarray(reconstruction_matrix(cfg.energy_eV,cfg.spectral_bounds,cfg.n_moments)); moments=xp.empty((cfg.n_moments,B.shape[1]))
    if ops.native is not None:
        c,a=scaling(cfg)
        return (G@ops.native.moments(B,cfg.n_moments,c,a)).T
    prev=B; SB=ops.S@B
    moments[0]=xp.sum(B.conj()*SB,axis=0).real
    if cfg.n_moments>1:
        curr=scaled_action(ops,cfg,B); moments[1]=xp.sum(SB.conj()*curr,axis=0).real
        for l in range(2,cfg.n_moments):
            nxt=2*scaled_action(ops,cfg,curr)-prev; moments[l]=xp.sum(SB.conj()*nxt,axis=0).real; prev,curr=curr,nxt
    return (G@moments).T
