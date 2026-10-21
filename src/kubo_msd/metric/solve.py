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
    def __init__(self,S,config,backend=None,atom_slices=None,*,device_S=None,host_setup=None,metric_identity=None):
        from .setup import prepare_host_metric
        self.config=config; self.backend=backend or CPUBackend(); self.xp=self.backend.xp
        self.atom_slices=atom_slices
        self.host_setup=host_setup or prepare_host_metric(S,config,atom_slices,metric_identity=metric_identity)
        if self.host_setup.metric_identity!=metric_identity:
            raise NumericalError('metric model identity mismatch')
        self.host_setup.validate(S,config,atom_slices)
        host=self.host_S=self.host_setup.S
        self.s_lower,self.s_upper,self.bounds_report=self.host_setup.bounds
        self.condition_estimate=self.s_upper/self.s_lower
        self.cache_key=self.host_setup.key+':'+repr(metric_identity)+':'+str(getattr(self.backend,'device_id','cpu'))
        identity=matrix_fingerprint(host)
        if device_S is not None:
            if self.backend.name!='gpu':raise NumericalError('device_S requires GPU backend')
            self.backend.validate_shared_matrix(device_S,identity)
            self.S=device_S
        else:self.S=self.backend.matrix(host,name='S',identity=identity)
        self.diagonal=self.backend.asarray(self.host_setup.diagonal,name='S diagonal',phase='host_metric_setup')
        self.blocks=[] if self.backend.name=='gpu' else [
            (a,b,self.backend.asarray(L,name='block Jacobi factor',phase='host_metric_setup'))
            for a,b,L in self.host_setup.blocks]
        self.preconditioner_data=self.host_setup.preconditioner_data
        self.index_blocks=[]
        if self.preconditioner_data is not None and self.backend.name!='gpu':
            self.index_blocks=[(self.backend.asarray(i,name='cluster indices'),
                               self.backend.asarray(L,name='cluster factor')) for i,L in self.preconditioner_data.blocks]
        self.calls=0; self.iterations=0; self.elapsed=0.; self.max_relative_residual=0.
        self.true_residual_checks=0; self.residual_restarts=0
        self.refinement_steps=0;self.cholesky=None
        self.control_status_reads=0;self.control_status_bytes=0;self.device_pcg=None;self.device_ic=None;self.device_blocks=None
        if self.backend.name=='gpu':
            from .gpu_kernels import DevicePCG,DeviceIC,DeviceBlocks
            self.device_pcg=DevicePCG(self) if config.metric_solver=='pcg' else None
            if self.preconditioner_data is not None and self.preconditioner_data.kind=='ic0':
                self.device_ic=DeviceIC(self.preconditioner_data,self.backend)
            blocks=[(np.arange(a,b),L) for a,b,L in self.host_setup.blocks]
            if self.preconditioner_data is not None:blocks+=self.preconditioner_data.blocks
            if blocks:self.device_blocks=DeviceBlocks(blocks,self.backend)
        if config.metric_solver=='cholesky':
            from .cholesky import prepare_cholesky
            self.cholesky=prepare_cholesky(self,config)
        self.native=None
        from ..backends.native import native_requested, NativeMetric
        if native_requested(config):
            blocks=self.blocks if self.cholesky is None else [(0,host.shape[0],self.cholesky.L)]
            self.native=NativeMetric(host,self.diagonal,blocks,config,self.preconditioner_data)

    def precondition(self,R,out=None):
        if self.device_blocks is not None:return self.device_blocks.apply(R,out=out)
        if self.preconditioner_data is not None and self.preconditioner_data.kind=='ic0':
            if self.device_ic is not None:return self.device_ic.apply(R)
            return self.preconditioner_data.apply(R)
        if self.index_blocks:
            out=self.xp.empty_like(R)
            for idx,L in self.index_blocks:
                if self.backend.name=='cpu':out[idx]=la.cho_solve((L,True),R[idx],check_finite=False)
            return out
        if not self.blocks: return R/self.diagonal[:,None]
        out=self.xp.empty_like(R)
        for a,b,L in self.blocks:
            if self.backend.name=='cpu':
                # L is already a Cholesky factor. Generic solve refactorizes
                # both triangular matrices for every atom and PCG iteration.
                out[a:b]=la.cho_solve((L,True),R[a:b],check_finite=False)
        return out

    def solve_s(self,B,config=None):
        from types import SimpleNamespace
        cfg=self.config if config is None else SimpleNamespace(**{**self.config.as_dict(),**vars(config)})
        for field in ('metric_solver','preconditioner','metric_partition_file','metric_ic_ordering','metric_ic_shift_max'):
            if getattr(cfg,field)!=getattr(self.config,field):
                raise NumericalError('per-solve override cannot change prepared '+field)
        xp=self.xp; t0=time.perf_counter()
        if self.native is not None: return self.native.solve(B,cfg)
        B=self.backend.asarray(B,dtype=xp.complex128,name='metric RHS')
        if B.ndim!=2 or B.shape[0]!=self.S.shape[0] or B.shape[1]<1 or not bool(xp.all(xp.isfinite(B))): raise NumericalError('metric RHS must be finite nonempty (N,batch)')
        # A finite complex value can have an unrepresentable Euclidean modulus.
        scale=xp.max(xp.maximum(xp.abs(B.real),xp.abs(B.imag)),axis=0); safe=xp.where(scale>0,scale,1.)
        Q=B/safe; bn=xp.sqrt(xp.sum(xp.abs(Q)**2,axis=0)); threshold=xp.maximum(cfg.rtol if hasattr(cfg,'rtol') else cfg.metric_rtol,0)*bn
        threshold=xp.maximum(threshold,cfg.metric_atol/safe)
        Y=xp.zeros_like(Q); R=Q.copy()
        Z=self.precondition(R) if self.cholesky is None and self.device_pcg is None else xp.zeros_like(Q)
        P=Z.copy();breakdown=xp.zeros(B.shape[1],dtype=bool)
        rz=xp.sum(R.conj()*Z,axis=0).real; active=bn>threshold; iters=xp.zeros(B.shape[1],dtype=xp.int64)
        P[:,~active]=0
        if self.cholesky is not None:
            Y=self.cholesky.solve_scaled(Q)
            for _ in range(cfg.metric_refine_maxiter):
                true=Q-self.S@Y;self.true_residual_checks+=1
                refine=xp.sqrt(xp.sum(xp.abs(true)**2,axis=0))>threshold
                if not bool(xp.any(refine)):break
                Y+=self.cholesky.solve_scaled(xp.where(refine[None,:],true,0))
                self.refinement_steps+=1
        if self.device_pcg is not None:Y,iters,breakdown=self.device_pcg.solve(Q,threshold,cfg)
        for k in range(0 if self.cholesky is not None or self.device_pcg is not None else cfg.metric_maxiter):
            if not bool(xp.any(active)): break
            SP=self.S@P; curvature=xp.sum(P.conj()*SP,axis=0)
            bad=active & (~xp.isfinite(curvature) | (curvature.real<=0) | (xp.abs(curvature.imag)>1e-10*xp.maximum(xp.abs(curvature.real),xp.finfo(float).tiny)))
            if bool(xp.any(bad)):
                breakdown |= bad
                break
            den=xp.where(active,curvature.real,1); alpha=xp.where(active,rz/den,0)
            Y+=P*alpha; R-=SP*alpha; iters+=active
            # Restart on true residual refresh to avoid inconsistent conjugacy.
            refresh=(k+1)%cfg.metric_check_interval==0
            rn=xp.sqrt(xp.sum(xp.abs(R)**2,axis=0))
            restart=xp.zeros_like(active)
            if cfg.metric_residual_policy=='adaptive':
                if refresh or bool(xp.any(active & (rn<=threshold))):
                    from .residual import assess_residual
                    true=Q-self.S@Y; self.true_residual_checks+=1
                    converged,restart=assess_residual(R,true,threshold,active,cfg.metric_gap_ratio,xp=xp)
                    R=xp.where(restart[None,:],true,R)
                    active &= ~converged
            else:
                if refresh:
                    R=xp.where(active[None,:],Q-self.S@Y,R)
                    self.true_residual_checks+=1
                    restart=active.copy()
                    rn=xp.sqrt(xp.sum(xp.abs(R)**2,axis=0))
                active &= rn>threshold
            self.residual_restarts+=int(xp.sum(restart))
            Z=self.precondition(R); newrz=xp.sum(R.conj()*Z,axis=0).real
            beta=xp.where(active,newrz/xp.where(rz!=0,rz,1),0)
            P=Z+P*xp.where(restart,0,beta)
            P[:,~active]=0; rz=newrz
        residual=Q-self.S@Y; self.true_residual_checks+=1
        norm=xp.sqrt(xp.sum(xp.abs(residual)**2,axis=0)); conv=norm<=threshold
        relative=xp.where(bn>0,norm/xp.where(bn>0,bn,1),0)
        finite_solution=xp.all(xp.isfinite(Y),axis=0); solution=Y
        if bool(xp.all(finite_solution)):
            # NumPy reports arithmetic overflow as a warning; acceptance below
            # turns it into a NumericalError. CuPy uses the same finite mask.
            with np.errstate(over='ignore',invalid='ignore'): solution=Y*safe
            finite_solution &= xp.all(xp.isfinite(solution),axis=0)
        conv &= finite_solution & ~breakdown
        with np.errstate(over='ignore',invalid='ignore'): true_residual=norm*safe
        report=MetricSolveReport(conv,iters,true_residual,relative,breakdown)
        self.calls+=1; self.iterations+=int(xp.max(iters)); self.max_relative_residual=max(self.max_relative_residual,float(xp.max(relative))); self.elapsed+=time.perf_counter()-t0
        if not bool(xp.all(conv)):
            failure=NumericalError(
                'S solve solution is nonfinite or not representable in complex128 after rescaling'
                if not bool(xp.all(finite_solution)) else
                'PCG non-positive/nonfinite/complex curvature breakdown'
                if bool(xp.any(breakdown)) else
                f'S solve failed true-residual acceptance; max relative residual={float(xp.max(relative)):.3g}')
            failure.diagnostics={key:self.backend.to_host(getattr(report,key)).tolist() for key in ('converged','iterations','true_residual','relative_residual','breakdown')}
            raise failure
        return solution,report

    def diagnostics(self):
        report={'cache_key':self.cache_key,'calls':self.calls,'max_iterations_sum':self.iterations,'seconds':self.elapsed,'max_relative_residual':self.max_relative_residual,'condition_upper_or_estimate':self.condition_estimate,'kappa_times_epsilon':self.condition_estimate*np.finfo(float).eps,'bounds':self.bounds_report,'kernel':'cpp' if self.native is not None else 'reference'}
        report.update(true_residual_checks=self.true_residual_checks,residual_restarts=self.residual_restarts)
        report.update(control_status_reads=self.control_status_reads,control_status_bytes=self.control_status_bytes,
                      control_tracking_scope='PCG batched active/error reads; final acceptance and library-internal synchronizations excluded')
        report.update(solver=self.config.metric_solver,refinement_steps=self.refinement_steps,
                      resolved_preconditioner=self.config.preconditioner if self.cholesky is None else 'not-applicable')
        if self.cholesky is not None:report.update(self.cholesky.diagnostics())
        if self.preconditioner_data is not None:report['preconditioner_setup']=self.preconditioner_data.report
        if self.device_blocks is not None:report['grouped_block_triangular']=self.device_blocks.diagnostics()
        if self.native is not None: report.update(self.native.diagnostics())
        return report
