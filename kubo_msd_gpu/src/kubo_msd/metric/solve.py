"""Batch independent PCG, with per-column scaling and original-equation residuals."""
import time
import numpy as np
import scipy.linalg as la
from ..backends.cpu import CPUBackend
from ..contracts import MetricSolveReport
from ..errors import NumericalError
from ..validation import hermitian, matrix_fingerprint
from .bounds import spd_bounds

class MetricSolver:
    def __init__(self,S,config,backend=None,atom_slices=None):
        self.config=config; self.backend=backend or CPUBackend(); self.xp=self.backend.xp
        host=hermitian(S,'overlap'); self.s_lower,self.s_upper,self.bounds_report=spd_bounds(host)
        self.condition_estimate=self.s_upper/self.s_lower
        self.cache_key=matrix_fingerprint(host)+':'+config.preconditioner+':'+repr(atom_slices)
        self.S=self.backend.matrix(host); self.diagonal=self.xp.asarray(host.diagonal().real)
        self.blocks=[]; self.calls=0; self.iterations=0; self.elapsed=0.; self.max_relative_residual=0.
        if config.preconditioner=='block-jacobi':
            slices=atom_slices or [(j,j+1) for j in range(host.shape[0])]
            if [i for a,b in slices for i in range(a,b)]!=list(range(host.shape[0])): raise NumericalError('atom_slices must partition orbitals')
            for a,b in slices:
                L=la.cholesky(host[a:b,a:b].toarray(),lower=True)
                self.blocks.append((a,b,self.xp.asarray(L)))
        self.native=None
        from ..backends.native import native_requested, NativeMetric
        if native_requested(config):
            self.native=NativeMetric(host,self.diagonal,self.blocks,config)

    def precondition(self,R):
        if not self.blocks: return R/self.diagonal[:,None]
        out=self.xp.empty_like(R)
        for a,b,L in self.blocks:
            out[a:b]=self.xp.linalg.solve(L.conj().T,self.xp.linalg.solve(L,R[a:b]))
        return out

    def solve_s(self,B,config=None):
        cfg=config or self.config; xp=self.xp; t0=time.perf_counter()
        if self.native is not None: return self.native.solve(B,cfg)
        B=xp.asarray(B,dtype=xp.complex128)
        if B.ndim!=2 or B.shape[0]!=self.S.shape[0] or not bool(xp.all(xp.isfinite(B))): raise NumericalError('metric RHS must be finite (N,batch)')
        # A finite complex value can have an unrepresentable Euclidean modulus.
        scale=xp.max(xp.maximum(xp.abs(B.real),xp.abs(B.imag)),axis=0); safe=xp.where(scale>0,scale,1.)
        Q=B/safe; bn=xp.sqrt(xp.sum(xp.abs(Q)**2,axis=0)); threshold=xp.maximum(cfg.rtol if hasattr(cfg,'rtol') else cfg.metric_rtol,0)*bn
        threshold=xp.maximum(threshold,cfg.metric_atol/safe)
        # Repeated parallel CSR products can differ by roundoff. Leave a
        # margin before freezing a column, while keeping final acceptance
        # at exactly the user's requested true-residual threshold.
        iteration_threshold=.5*threshold
        Y=xp.zeros_like(Q); R=Q.copy(); Z=self.precondition(R); P=Z.copy()
        rz=xp.sum(R.conj()*Z,axis=0).real; active=bn>threshold; iters=xp.zeros(B.shape[1],dtype=xp.int64)
        P[:,~active]=0
        for k in range(cfg.metric_maxiter):
            if not bool(xp.any(active)): break
            SP=self.S@P; curvature=xp.sum(P.conj()*SP,axis=0)
            bad=active & ((curvature.real<=0) | (xp.abs(curvature.imag)>1e-10*xp.maximum(xp.abs(curvature.real),xp.finfo(float).tiny)))
            if bool(xp.any(bad)): raise NumericalError('PCG non-positive/complex curvature breakdown')
            den=xp.where(active,curvature.real,1); alpha=xp.where(active,rz/den,0)
            Y+=P*alpha; R-=SP*alpha; iters+=active
            # Restart on true residual refresh to avoid inconsistent conjugacy.
            refresh=(k+1)%32==0
            if refresh: R=Q-self.S@Y
            rn=xp.sqrt(xp.sum(xp.abs(R)**2,axis=0))
            # A small recursive residual is only a candidate for convergence.
            # Verify it against the original scaled equation while iteration
            # budget remains; refresh/restart only those columns if needed.
            candidates=active & (rn<=iteration_threshold)
            checked_candidates=False
            if not refresh and bool(xp.any(candidates)):
                true_R=Q-self.S@Y
                R[:,candidates]=true_R[:,candidates]
                rn=xp.sqrt(xp.sum(xp.abs(R)**2,axis=0))
                checked_candidates=True
            active=active & (rn>iteration_threshold)
            Z=self.precondition(R); newrz=xp.sum(R.conj()*Z,axis=0).real
            beta=xp.where(active,newrz/xp.where(rz!=0,rz,1),0)
            P=Z if refresh else Z+P*beta
            if checked_candidates: P[:,candidates]=Z[:,candidates]
            P[:,~active]=0; rz=newrz
        residual=Q-self.S@Y; norm=xp.sqrt(xp.sum(xp.abs(residual)**2,axis=0)); conv=norm<=threshold
        relative=xp.where(bn>0,norm/xp.where(bn>0,bn,1),0)
        finite_solution=xp.all(xp.isfinite(Y),axis=0); solution=Y
        if bool(xp.all(finite_solution)):
            # NumPy reports arithmetic overflow as a warning; acceptance below
            # turns it into a NumericalError. CuPy uses the same finite mask.
            with np.errstate(over='ignore',invalid='ignore'): solution=Y*safe
            finite_solution &= xp.all(xp.isfinite(solution),axis=0)
        conv &= finite_solution
        with np.errstate(over='ignore',invalid='ignore'): true_residual=norm*safe
        report=MetricSolveReport(conv,iters,true_residual,relative,xp.zeros_like(conv))
        self.calls+=1; self.iterations+=int(xp.max(iters)); self.max_relative_residual=max(self.max_relative_residual,float(xp.max(relative))); self.elapsed+=time.perf_counter()-t0
        if not bool(xp.all(conv)):
            failure=NumericalError(
                'S solve solution is nonfinite or not representable in complex128 after rescaling'
                if not bool(xp.all(finite_solution)) else
                f'S solve failed true-residual acceptance; max relative residual={float(xp.max(relative)):.3g}')
            failure.diagnostics={key:self.backend.to_host(getattr(report,key)).tolist() for key in ('converged','iterations','true_residual','relative_residual','breakdown')}
            raise failure
        return solution,report

    def diagnostics(self):
        report={'cache_key':self.cache_key,'calls':self.calls,'max_iterations_sum':self.iterations,'seconds':self.elapsed,'max_relative_residual':self.max_relative_residual,'condition_upper_or_estimate':self.condition_estimate,'kappa_times_epsilon':self.condition_estimate*np.finfo(float).eps,'bounds':self.bounds_report,'kernel':'cpp' if self.native is not None else 'reference'}
        if self.native is not None: report.update(self.native.diagnostics())
        return report
