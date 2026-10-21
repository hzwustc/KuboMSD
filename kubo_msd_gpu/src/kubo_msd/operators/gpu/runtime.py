"""Reuse the established velocity actions without constructing host matrices."""
import numpy as np
from ..velocity import OperatorBundle
from ...metric.gpu_setup import DeviceMetricSolver, hermitian_gpu
from ...errors import KuboMSDError


class DeviceOperatorBundle(OperatorBundle):
    def __init__(self, model, config, context):
        if config.backend != 'gpu' or config.transport_distribution != 'serial':
            raise ValueError('device operators require serial GPU config')
        if model.device_id != context.device_id:
            raise ValueError('model belongs to wrong device')
        self.model, self.backend, self.context = model, context, context
        self.xp, self.n, self.config = context.xp, model.n_orbitals, config
        self.H = hermitian_gpu(model.H_cov,'H',context)
        self.S = hermitian_gpu(model.S,'S',context)
        self.velocity, self.position, self.composite = model.velocity, model.position, model.composite
        self.directions = dict(zip(config.axes,config.axes_vectors))
        self._h_calls = self._v_calls = 0
        self.native = None
        with context.phase('metric_setup'):
            self.metric = DeviceMetricSolver(self.S,config,context,model.atom_slices)
        for direction in config.axes_vectors:
            for axis,c in zip('xyz',direction):
                if c and axis not in self.velocity and axis not in self.position and axis not in self.composite:
                    raise KuboMSDError(f'missing physical operator for {axis}')
        for group in (self.velocity,self.position):
            for axis,mat in group.items(): hermitian_gpu(mat,axis,context)
        for parts in self.composite.values():
            if len(parts)!=3: raise KuboMSDError('composite requires bond/xbar/delta')
            for mat in parts:
                context.check_device(mat)
                if not bool(self.xp.all(self.xp.isfinite(mat.data))):
                    raise KuboMSDError('nonfinite composite')
        self.velocity_hermiticity = {}
        xp = self.xp
        if self.n<=64: probe=xp.eye(self.n,dtype=xp.complex128)
        else:
            rng=np.random.default_rng(551)
            probe=context.asarray(rng.normal(size=(self.n,8))+1j*rng.normal(size=(self.n,8)),
                                  'velocity_probe','constants')
        for axis in config.axes:
            v = probe.conj().T@self.apply_v_cov(axis,probe)
            err = float(xp.linalg.norm(v-v.conj().T)/xp.maximum(xp.linalg.norm(v),1e-300))
            self.velocity_hermiticity[axis] = {'relative_error':err, 'mode':'full' if self.n<=64 else '8-vector probe'}
            if err>1e-10:
                raise KuboMSDError(f'constructed velocity {axis} is not Hermitian ({err:g})')


def make_device_operators(model, run_config, context):
    with context.phase('metric_setup'):
        return DeviceOperatorBundle(model, run_config, context)
