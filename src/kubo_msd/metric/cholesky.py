"""Reusable factor of original S. No inverse, spectral truncation or jitter."""
import time
import numpy as np
import scipy.linalg as la
from ..errors import NumericalError


def storage_bytes(S):
    return S.data.nbytes+S.indices.nbytes+S.indptr.nbytes


def check_budget(S, config, *, resident=0, device_free=None):
    # Input/factor, factorization workspace, layout/native copy, and RHS work.
    # Count resident sparse copies explicitly; budget is not available RAM.
    estimate = resident+2*storage_bytes(S)+4*16*S.shape[0]**2
    if estimate > config.memory_limit_gb*1024**3:
        raise NumericalError("Cholesky factor/workspace exceeds memory budget")
    if device_free is not None and resident+4*16*S.shape[0]**2 > .9*device_free:
        raise NumericalError("Cholesky factor/workspace exceeds available GPU memory")
    return estimate


class CholeskyFactor:
    def __init__(self, metric, config):
        self.backend=metric.backend; self.xp=metric.xp
        self.backend.synchronize(); started=time.perf_counter()
        if self.backend.name=="cpu":
            self.L=metric.host_setup.direct_factor
            if self.L is None:
                raise NumericalError("missing initialized CPU Cholesky factor")
        else:
            check_budget(metric.host_S,config,device_free=self.backend.memory_info()["free_bytes"])
            try:
                # cuSOLVER errors must surface, not silently produce a NaN factor.
                import cupyx
                with cupyx.errstate(linalg="raise"):
                    self.L=self.xp.linalg.cholesky(self.backend.device_dense(metric.S,"S Cholesky working copy"))
            except Exception as exc:
                raise NumericalError(f"GPU Cholesky of original S failed: {exc}") from exc
            if not bool(self.xp.all(self.xp.isfinite(self.L))):
                raise NumericalError("GPU Cholesky produced nonfinite factor")
            self.backend._record("D2D",self.L,"S Cholesky factor","metric_factor","device factorization")
        self.backend.synchronize()
        self.seconds=metric.host_setup.factor_seconds if self.backend.name=="cpu" else time.perf_counter()-started

    def solve_scaled(self,Q):
        if self.backend.name=="cpu":
            return la.cho_solve((self.L,True),Q,check_finite=False)
        from cupyx.scipy.linalg import solve_triangular
        first=solve_triangular(self.L,Q,lower=True,check_finite=False)
        return solve_triangular(self.L,first,lower=True,trans="C",check_finite=False)

    def diagnostics(self):
        return dict(factorizations=1,factor_bytes=int(self.L.nbytes),
                    factor_seconds=self.seconds,factor_backend=self.backend.name)


def prepare_cholesky(metric,config):
    return CholeskyFactor(metric,config)
