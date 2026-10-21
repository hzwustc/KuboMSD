"""Covariant physical velocities in A/fs, or implicit periodic composite actions."""
import numpy as np
from types import SimpleNamespace
from ..backends.cpu import CPUBackend
from ..metric.solve import MetricSolver
from ..validation import hermitian,matrix_fingerprint
from ..errors import KuboMSDError,NumericalError
from ..units import HBAR_EV_FS

class OperatorBundle:
    def __init__(self,model,config,backend):
        self.backend=backend; self.xp=backend.xp; self.n=model.n_orbitals; self.config=config
        self.host_H=hermitian(model.H_cov,'Hamiltonian'); self.host_S=hermitian(model.S,'overlap')
        if self.host_H.shape!=self.host_S.shape: raise KuboMSDError('H/S dimensions disagree')
        if config.metric_solver=='cholesky':
            from ..metric.cholesky import check_budget,storage_bytes
            import scipy.sparse as sp
            matrices=[self.host_H,self.host_S,*model.velocity.values(),*model.position.values(),
                      *[m for parts in model.composite.values() for m in parts]]
            # Host canonical arrays and native/device copies coexist.
            resident=2*sum(storage_bytes(sp.csr_matrix(m)) for m in matrices)
            free=backend.memory_info()['free_bytes'] if backend.name=='gpu' else None
            check_budget(self.host_S,config,resident=resident,device_free=free)
        from ..metric.setup import prepare_host_metric
        from ..metric.partition import model_identity
        identity=model_identity(model) if config.preconditioner=='cluster-jacobi' else None
        setup=prepare_host_metric(self.host_S,config,model.atom_slices,metric_identity=identity)
        self.host_S=setup.S
        self.H=backend.matrix(self.host_H,name='H',identity=matrix_fingerprint(self.host_H)); self.S=backend.matrix(self.host_S,name='S',identity=matrix_fingerprint(self.host_S))
        self.metric=MetricSolver(self.host_S,config,backend,model.atom_slices,device_S=self.S if backend.name=='gpu' else None,host_setup=setup,metric_identity=identity)
        if backend.name=='gpu':backend.stages['host_metric_setup']=setup.seconds
        self.velocity={a:backend.matrix(hermitian(v,'velocity '+a)) for a,v in model.velocity.items()}
        self.position={a:backend.matrix(hermitian(v,'position '+a)) for a,v in model.position.items()}
        self.composite={a:tuple(backend.matrix(v) for v in parts) for a,parts in model.composite.items()}
        self.directions={a:v for a,v in zip(config.axes,config.axes_vectors)}; self._h_calls=0; self._v_calls=0; self.native=None
        for direction in config.axes_vectors:
            for axis,c in zip('xyz',direction):
                if c and axis not in self.velocity and axis not in self.position and axis not in self.composite: raise KuboMSDError(f'missing physical velocity/position/dipole for {axis}')
        for collection in (self.velocity,self.position):
            if any(v.shape!=(self.n,self.n) for v in collection.values()): raise KuboMSDError('operator dimension mismatch')
        for axis,parts in self.composite.items():
            if len(parts)!=3: raise KuboMSDError('composite velocity requires bond/xbar/delta')
            for v in parts:
                if v.shape!=(self.n,self.n) or not bool(self.xp.all(self.xp.isfinite(v.data if backend.sparse.issparse(v) else v))): raise KuboMSDError('invalid composite operator dimensions/finite entries')
        if self.metric.native is not None:
            from ..backends.native import NativeOperators
            self.native=NativeOperators(self,config)
        self.velocity_hermiticity={}
        # For small models check every matrix element; otherwise a reproducible
        # bilinear probe detects malformed implicit composite actions without NxN storage.
        rng=np.random.default_rng(551)
        host_probe=np.eye(self.n,dtype=complex) if self.n<=64 else rng.normal(size=(self.n,8))+1j*rng.normal(size=(self.n,8))
        probe=backend.asarray(host_probe,name='velocity Hermiticity probe',phase='transport_init')
        for axis in config.axes:
            self.velocity_hermiticity[axis]=self._check_velocity(axis,probe)

    def _check_velocity(self,axis,probe):
        """Separate validation accuracy from the user's transport accuracy.

        Two approximate S solves can amplify residual errors in the composite
        action. Retry only failed stochastic PCG probes, with tighter solves
        through the same backend/action. Never relax the Hermiticity criterion
        or change the prepared factor or the engine's transport controls.
        """
        cfg=self.config;controls=None;attempts=[];rtol=cfg.metric_rtol;atol=cfg.metric_atol
        retries=2 if cfg.solver=='stochastic-kpm' and cfg.metric_solver=='pcg' else 0
        for attempt in range(retries+1):
            try:
                v=probe.conj().T@self.apply_v_cov(axis,probe,controls)
            except NumericalError as exc:
                attempts.append(dict(metric_rtol=rtol,metric_atol=atol,
                                     status='S_SOLVE_FAILED',reason=str(exc)))
                exc.diagnostics={**(getattr(exc,'diagnostics',None) or {}),
                    'velocity_probe':dict(axis=axis,threshold=1e-10,attempts=attempts)}
                raise
            error=float(self.xp.linalg.norm(v-v.conj().T)/self.xp.maximum(self.xp.linalg.norm(v),1e-300))
            attempts.append(dict(metric_rtol=rtol,metric_atol=atol,relative_error=error))
            if np.isfinite(error) and error<=1e-10:
                return dict(relative_error=error,threshold=1e-10,attempts=attempts,
                            mode='full' if self.n<=64 else '8-vector bilinear probe')
            if not np.isfinite(error) or attempt==retries:break
            rtol=min(rtol*.1,1e-12 if attempt==0 else 1e-13);atol*=.1
            controls=SimpleNamespace(metric_rtol=rtol,metric_atol=atol,
                                     metric_maxiter=cfg.metric_maxiter)
        failure=KuboMSDError(f'constructed velocity {axis} is not Hermitian '
                            f'(relative error {error:g}; probe attempts {attempts})')
        failure.diagnostics={'velocity_probe':dict(axis=axis,threshold=1e-10,attempts=attempts)}
        raise failure

    def solve_s(self,B,config=None): return self.metric.solve_s(B,config)
    @property
    def h_calls(self): return self._h_calls+(self.native.core.h_calls if self.native is not None else 0)
    @property
    def v_calls(self): return self._v_calls+(self.native.core.v_calls if self.native is not None else 0)
    def apply_h_cov(self,B): return self.native.H.matmul(B) if self.native is not None else self.H@B
    def apply_h_action(self,B):
        if self.native is not None: return self.native.h_action(B)
        self._h_calls+=1
        return self.solve_s(self.H@B)[0]
    def _cartesian_v(self,axis,B,config=None):
        if axis in self.velocity: return self.velocity[axis]@B
        if axis in self.position:
            X=self.position[axis]
            return 1j*(self.H@self.solve_s(X@B,config)[0]-X@self.solve_s(self.H@B,config)[0])/HBAR_EV_FS
        bond,xbar,delta=self.composite[axis]
        return 1j*(bond@B-xbar@self.solve_s(self.H@B,config)[0]+self.H@self.solve_s(delta@B,config)[0])/HBAR_EV_FS
    def apply_v_cov(self,axis,B,config=None):
        direction=self.directions.get(axis,np.eye(3)['xyz'.index(axis)] if axis in 'xyz' else None)
        if direction is None: raise KuboMSDError('unknown direction')
        if self.native is not None: return self.native.v_cov(direction,B,config)
        self._v_calls+=1
        result=self.xp.zeros_like(B,dtype=self.xp.complex128)
        for a,c in zip('xyz',direction):
            if c: result+=c*self._cartesian_v(a,B,config)
        return result
    def apply_v_action(self,axis,B): return self.solve_s(self.apply_v_cov(axis,B))[0]
    def inner(self,A,B): return self.xp.sum(A.conj()*(self.native.S.matmul(B) if self.native is not None else self.S@B),axis=0)

def build_operators(model,config):
    if config.backend=='cpu': backend=CPUBackend()
    else:
        from ..backends.gpu import GPUBackend
        backend=GPUBackend()
    return OperatorBundle(model,config,backend)
