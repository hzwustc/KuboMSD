"""Device-only overlap validation and metric setup; original PCG is reused."""
import numpy as np
from .solve import MetricSolver
from ..errors import NumericalError


def hermitian_gpu(A, name, context):
    context.check_device(A)
    xp = context.xp
    if A.shape[0] != A.shape[1] or not bool(xp.all(xp.isfinite(A.data))):
        raise NumericalError(f'{name}: invalid dimensions or nonfinite entries')
    defect = A-A.conj().T
    norm = xp.linalg.norm(A.data)
    error = float(xp.linalg.norm(defect.data)/xp.maximum(norm, 1e-300))
    if error > 1e-10:
        raise NumericalError(f'{name}: not Hermitian ({error:g}); no automatic symmetrization')
    return A


def spd_bounds_gpu(S, context):
    xp = context.xp
    hermitian_gpu(S, 'overlap', context)
    d = S.diagonal().real
    radii = xp.asarray(abs(S).sum(axis=1)).ravel()-xp.abs(d)
    lo, hi = float(xp.min(d-radii)), float(xp.max(d+radii))
    if lo > 0:
        return lo, hi, {'source':'device Gershgorin', 'certified':True}
    n = S.shape[0]
    with context.phase('metric_spd'):
        if n <= 128:
            context.require_bytes(16*n*n*4, 'small S eigenspectrum')
            ev = xp.linalg.eigvalsh(S.toarray())
            lo, hi = float(ev[0]), float(ev[-1])
            report = {'source':'full small device S eigenvalues', 'certified':True}
        else:
            from cupyx.scipy.sparse.linalg import eigsh
            # CuPy reuses v0 as the Lanczos SpMV output buffer; it must have
            # the same scalar type as S, including complex128 at Gamma.
            v0 = context.asarray(np.random.default_rng(9021).normal(size=n),
                                 'S_Ritz_start', dtype=S.dtype)
            # The default 33-vector restart can exhaust its iterations for
            # clustered small eigenvalues even at modest condition numbers.
            ncv = min(128, n-1)
            context.require_bytes(16*n*(ncv+8)+16*ncv*ncv*4,
                                  'overlap SPD Ritz workspace')
            e, v = eigsh(S, k=1, which='SA', tol=1e-12, v0=v0, ncv=ncv)
            residual = float(xp.linalg.norm(S@v[:,0]-e[0]*v[:,0]))
            if residual > 1e-10*max(hi, 1):
                raise NumericalError('overlap SPD Ritz failed residual acceptance')
            lo = float(e[0])-10*residual
            report = {'source':'device smallest S Ritz minus 10 residuals',
                      'certified':False, 'residual':residual, 'subspace_size':ncv}
    if lo <= 0 or not np.isfinite([lo, hi]).all():
        raise NumericalError('overlap is not positive definite or SPD verification failed')
    return lo*(1-1e-10), hi*(1+1e-10), report


class DeviceMetricSolver(MetricSolver):
    def __init__(self, S, config, context, atom_slices=()):
        self.context = context
        self.config, self.backend, self.xp = config, context, context.xp
        self.S = S
        self.s_lower, self.s_upper, self.bounds_report = spd_bounds_gpu(S, context)
        self.condition_estimate = self.s_upper/self.s_lower
        self.cache_key = f'device:{context.device_id}:S:{S.data.data.ptr}:{config.preconditioner}'
        self.diagonal = S.diagonal().real
        self.blocks = []
        self.calls = self.iterations = 0
        self.elapsed = self.max_relative_residual = 0.
        self.native = None
        if config.preconditioner == 'block-jacobi':
            slices = atom_slices or [(j,j+1) for j in range(S.shape[0])]
            if [i for a,b in slices for i in range(a,b)] != list(range(S.shape[0])):
                raise NumericalError('atom_slices must partition orbitals')
            import cupyx
            with cupyx.errstate(linalg='raise'):
                for a,b in slices:
                    context.require_bytes(16*(b-a)**2*4, 'block-Jacobi factor')
                    L = self.xp.linalg.cholesky(S[a:b,a:b].toarray())
                    self.blocks.append((a,b,L))

    def diagnostics(self):
        return {**super().diagnostics(), 'kernel':'gpu PCG', 'shared_device_S':True}
