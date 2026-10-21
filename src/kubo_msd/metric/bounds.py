"""Conservative bounds when available, explicitly estimated extremal Ritz checks otherwise."""
import numpy as np
import scipy.sparse as sp
from ..validation import hermitian
from ..errors import NumericalError

def spd_bounds(S,*,allow_dense=True):
    S=hermitian(S,'overlap'); diag=S.diagonal().real
    radii=np.asarray(abs(S).sum(axis=1)).ravel()-np.abs(diag)
    lo=float(np.min(diag-radii)); hi=float(np.max(diag+radii))
    if lo>0: return lo,hi,{'source':'Gershgorin','certified':True}
    n=S.shape[0]
    if n<=2 and not allow_dense:
        lo=float(diag[0]) if n==1 else float((diag[0]+diag[1])/2-np.hypot((diag[0]-diag[1])/2,abs(S[0,1])))
        source='analytic one/two dimensional SPD check'
    elif n<=128 and allow_dense:
        ev=np.linalg.eigvalsh(S.toarray()); lo=float(ev[0]); hi=float(ev[-1]); source='small-matrix eigenvalues of S'
    else:
        v0=np.random.default_rng(9021).normal(size=n)
        e,V=sp.linalg.eigsh(S,k=1,which='SA',tol=1e-12,v0=v0)
        residual=float(np.linalg.norm(S@V[:,0]-e[0]*V[:,0]))
        lo=float(e[0]-10*residual); source='smallest Ritz value of S minus 10 residuals (estimated)'
    if lo<=0 or not np.isfinite([lo,hi]).all(): raise NumericalError('overlap is not positive definite or SPD verification failed')
    return lo*(1-1e-10),hi*(1+1e-10),{'source':source,'certified':n<=2 or (n<=128 and allow_dense)}

def estimate_spectral_bounds(ops,config):
    H=ops.host_H; S=ops.host_S; n=H.shape[0]
    # Generalized Rayleigh quotient bound |lambda| <= ||H||_inf / lambda_min(S).
    hnorm=float(np.asarray(abs(H).sum(axis=1)).max())
    b=hnorm/ops.metric.s_lower
    if b==0 and config.spectral_bounds is None: raise NumericalError('fully degenerate zero spectrum needs explicit spectral bounds')
    conservative=(-b,b)
    if n<=2:
        # Extreme roots only; no eigenvectors or complete-eigensystem transport.
        import scipy.linalg as la
        extremal=la.eigvalsh(H.toarray(),S.toarray(),subset_by_index=[0,n-1])
        residuals=[0.,0.]
    else:
        extrema=[]; residuals=[]
        for which in ('SA','LA'):
            vals=[]
            for seed in (811,812):
                e,v=sp.linalg.eigsh(H,M=S,k=1,which=which,tol=1e-11,v0=np.random.default_rng(seed).normal(size=n))
                r=np.linalg.norm(H@v[:,0]-e[0]*(S@v[:,0]))/np.linalg.norm(v[:,0])
                vals.append(float(e[0])); residuals.append(float(r))
            extrema.append(min(vals) if which=='SA' else max(vals))
        extremal=np.array(extrema)
    lo=float(extremal[0]); hi=float(extremal[-1])
    if config.spectral_bounds is not None:
        bounds=config.spectral_bounds
        if bounds[0]>=lo or bounds[1]<=hi: raise NumericalError('requested spectral bounds must enclose estimated full spectrum with margin')
        certified=bool(ops.metric.bounds_report['certified'] and bounds[0]<=-b and bounds[1]>=b)
        source='user bounds checked against generalized extremal Ritz values'
    else:
        if hi-lo<=np.finfo(float).eps*max(abs(lo),abs(hi),1): raise NumericalError('degenerate spectrum needs explicit bounds')
        margin=max(.01*(hi-lo),10*max(residuals)/ops.metric.s_lower)
        bounds=(lo-margin,hi+margin); certified=False; source='two-start generalized Ritz extrema plus margin'
    return {'bounds':tuple(bounds),'source':source,'certified':certified,'residuals':residuals,'conservative_bounds':conservative,'extrema':[lo,hi]}
