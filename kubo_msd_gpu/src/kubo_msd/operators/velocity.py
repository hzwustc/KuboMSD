"""Covariant physical velocities in A/fs, or implicit periodic composite actions."""
import numpy as np
from ..backends.cpu import CPUBackend
from ..metric.solve import MetricSolver
from ..validation import hermitian
from ..errors import KuboMSDError
from ..units import HBAR_EV_FS

class OperatorBundle:
    def __init__(self,model,config,backend):
        self.backend=backend; self.xp=backend.xp; self.n=model.n_orbitals; self.config=config
        self.host_H=hermitian(model.H_cov,'Hamiltonian'); self.host_S=hermitian(model.S,'overlap')
        if self.host_H.shape!=self.host_S.shape: raise KuboMSDError('H/S dimensions disagree')
        self.H=backend.matrix(self.host_H); self.S=backend.matrix(self.host_S)
        self.metric=MetricSolver(self.host_S,config,backend,model.atom_slices)
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
        probe=self.xp.asarray(host_probe)
        for axis in config.axes:
            v=probe.conj().T@self.apply_v_cov(axis,probe)
            error=float(self.xp.linalg.norm(v-v.conj().T)/self.xp.maximum(self.xp.linalg.norm(v),1e-300))
            self.velocity_hermiticity[axis]={'relative_error':error,'mode':'full' if self.n<=64 else '8-vector bilinear probe'}
            if error>1e-10: raise KuboMSDError(f'constructed velocity {axis} is not Hermitian (relative error {error:g})')

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
    def _cartesian_v(self,axis,B):
        if axis in self.velocity: return self.velocity[axis]@B
        if axis in self.position:
            X=self.position[axis]
            return 1j*(self.H@self.solve_s(X@B)[0]-X@self.solve_s(self.H@B)[0])/HBAR_EV_FS
        bond,xbar,delta=self.composite[axis]
        return 1j*(bond@B-xbar@self.solve_s(self.H@B)[0]+self.H@self.solve_s(delta@B)[0])/HBAR_EV_FS
    def apply_v_cov(self,axis,B):
        direction=self.directions.get(axis,np.eye(3)['xyz'.index(axis)] if axis in 'xyz' else None)
        if direction is None: raise KuboMSDError('unknown direction')
        if self.native is not None: return self.native.v_cov(direction,B)
        self._v_calls+=1
        result=self.xp.zeros_like(B,dtype=self.xp.complex128)
        for a,c in zip('xyz',direction):
            if c: result+=c*self._cartesian_v(a,B)
        return result
    def apply_v_action(self,axis,B): return self.solve_s(self.apply_v_cov(axis,B))[0]
    def inner(self,A,B): return self.xp.sum(A.conj()*(self.native.S.matmul(B) if self.native is not None else self.S@B),axis=0)

def build_operators(model,config):
    if config.backend=='cpu': backend=CPUBackend()
    else:
        from ..backends.gpu import GPUBackend
        backend=GPUBackend()
    return OperatorBundle(model,config,backend)
