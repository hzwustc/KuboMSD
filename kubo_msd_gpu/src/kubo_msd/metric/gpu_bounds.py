"""S-inner-product Krylov/Rayleigh–Ritz; see docs/GPU_RITZ.md."""
import numpy as np
from ..errors import NumericalError


def _extrema(ops, seed):
    xp, ctx, n = ops.xp, ops.backend, ops.n
    cap = min(n, 512)
    ctx.require_bytes(16*n*(cap*3+8)+16*cap*cap*8, 'generalized Ritz basis',
                      int(ops.config.memory_limit_gb*1024**3))
    Q = xp.empty((n,cap), dtype=xp.complex128)
    SQ = xp.empty_like(Q)
    w = ctx.asarray(np.random.default_rng(seed).normal(size=(n,1)),
                    f'generalized_Ritz_start_{seed}', 'constants', np.complex128)
    hnorm = float(xp.max(xp.asarray(abs(ops.H).sum(axis=1))))
    snorm = float(xp.max(xp.asarray(abs(ops.S).sum(axis=1))))
    for j in range(cap):
        sw = ops.S@w
        for _ in range(2):
            if j:
                c = Q[:,:j].conj().T@sw
                w -= Q[:,:j]@c
                sw = ops.S@w
        norm2 = float(xp.real(xp.sum(w.conj()*sw)))
        if norm2 < -1e-12: raise NumericalError('negative S norm in generalized Ritz')
        if norm2 < 1e-26:
            if j == 0: raise NumericalError('empty generalized Ritz subspace')
            m = j
        else:
            Q[:,j:j+1] = w/xp.sqrt(norm2)
            SQ[:,j:j+1] = sw/xp.sqrt(norm2)
            m = j+1
        breakdown = m == j
        if m == cap or breakdown or m % 16 == 0:
            q, sq = Q[:,:m], SQ[:,:m]
            hq = ops.H@q
            proj = q.conj().T@hq
            vals, vecs = xp.linalg.eigh(proj)
            pick = xp.asarray([0,m-1])
            ev, v = vals[pick], q@vecs[:,pick]
            residual = xp.linalg.norm(ops.H@v-(ops.S@v)*ev,axis=0)/xp.linalg.norm(v,axis=0)
            scale = hnorm+xp.abs(ev)*snorm
            if bool(xp.all(residual <= 1e-10*xp.maximum(scale,1e-300))):
                return ctx.to_host(ev, 'Ritz_extrema', 'metadata'), ctx.to_host(residual,'Ritz_residuals','metadata'), m
            if m == cap or breakdown:
                raise NumericalError('generalized Ritz extrema did not converge; enlarge/inspect spectrum and PCG accuracy')
        w = ops.apply_h_action(Q[:,j:j+1])
    raise NumericalError('generalized Ritz failed')


def estimate_spectral_bounds_gpu(ops, config):
    xp = ops.xp
    b = float(xp.max(xp.asarray(abs(ops.H).sum(axis=1))))/ops.metric.s_lower
    with ops.backend.phase('spectral_bounds'):
        results = [_extrema(ops, seed) for seed in (811,812)]
    lo = min(r[0][0] for r in results)
    hi = max(r[0][1] for r in results)
    residuals = [float(v) for r in results for v in r[1]]
    allowance = 10*max(residuals)/ops.metric.s_lower
    if config.spectral_bounds is not None:
        bounds = config.spectral_bounds
        if bounds[0] >= lo-allowance or bounds[1] <= hi+allowance:
            raise NumericalError('requested spectral bounds do not enclose checked full spectrum with margin')
        certified = bool(ops.metric.bounds_report['certified'] and bounds[0]<=-b and bounds[1]>=b)
        source = 'user bounds checked against device two-start generalized Ritz'
    else:
        if hi-lo <= np.finfo(float).eps*max(abs(lo),abs(hi),1):
            raise NumericalError('degenerate spectrum needs explicit spectral bounds')
        margin = max(.01*(hi-lo), allowance)
        bounds, certified = (float(lo-margin),float(hi+margin)), False
        source = 'device two-start S-inner-product generalized Ritz plus margin'
    return {'bounds':tuple(bounds), 'source':source, 'certified':certified,
            'residuals':residuals, 'conservative_bounds':(-b,b), 'extrema':[float(lo),float(hi)],
            'subspace_sizes':[r[2] for r in results], 'seeds':[811,812]}
