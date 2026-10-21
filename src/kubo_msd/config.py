"""Validated immutable controls. Energies are absolute eV, times are fs."""
from dataclasses import dataclass, fields
import numpy as np
from .errors import KuboMSDError

def grid(values, name, starts_zero=False):
    a=np.array(values,dtype=np.float64,copy=True)
    if a.ndim!=1 or not len(a) or not np.all(np.isfinite(a)) or np.any(np.diff(a)<=0):
        raise KuboMSDError(f'{name} must be a finite, strictly increasing nonempty array')
    if starts_zero and a[0]!=0: raise KuboMSDError('time must start at zero')
    a.setflags(write=False)
    return a

def resolve_output_grid(emin,emax,n_energy):
    if not np.isfinite([emin,emax]).all() or emin>=emax or int(n_energy)!=n_energy or n_energy<2:
        raise KuboMSDError('require finite emin < emax and n_energy >= 2')
    return grid(np.linspace(emin,emax,int(n_energy)),'energy')

def resolve_time_grid(tmax,dt):
    if not np.isfinite([tmax,dt]).all() or tmax<0 or dt<=0: raise KuboMSDError('invalid tmax/dt')
    t=np.arange(int(np.floor(tmax/dt))+1,dtype=float)*dt
    if t[-1]<tmax and not np.isclose(t[-1],tmax,rtol=0,atol=8*np.spacing(max(tmax,dt))): t=np.append(t,tmax)
    else: t[-1]=tmax
    return grid(t,'time',True)

@dataclass(frozen=True)
class RunConfig:
    energy_eV: object
    time_fs: object
    solver: str='exact'
    backend: str='cpu'
    cpu_kernel: str='cpp'
    omp_threads: int=1
    transport_distribution: str='serial'
    kernel: str='gaussian'
    sigma_eV: float=.05
    n_moments: int=256
    spectral_bounds: object=None
    axes: tuple=('x','y','z')
    axes_vectors: object=None
    initial_state_selection: str='all'
    buffer_eV: float=0.
    stochastic_method: str='auto'
    nsamples: int=64
    seed: int=20261019
    sample_batch: int=8
    statistics_block: int=8
    pair_block: int=128
    exact_projection_kernel: str='sparse'
    time_block: int=16
    energy_block: int=256
    metric_rtol: float=1e-10
    metric_atol: float=0.
    metric_maxiter: int=1000
    metric_solver: str='pcg'
    metric_refine_maxiter: int=3
    metric_residual_policy: str='fixed'
    metric_check_interval: int=32
    metric_gap_ratio: float=.1
    metric_gpu_check_batch: int=1
    preconditioner: str='diagonal'
    metric_partition_file: object=None
    metric_ic_ordering: str='rcm'
    metric_ic_shift_max: float=1e-2
    sampling_tol: float=1e-10
    sampling_max_order: int=4096
    propagation_tol: float=1e-12
    propagation_max_order: int=4096
    segment_fs: float=1.
    integration_dt_fs: float=.01
    include_vac: bool=True
    rho_abs: float=0.
    rho_rel: float=1e-12
    memory_limit_gb: float=4.
    output_npz: bool=False
    efermi_eV: object=None
    efermi_source: object=None

    def __post_init__(self):
        object.__setattr__(self,'energy_eV',grid(self.energy_eV,'energy'))
        object.__setattr__(self,'time_fs',grid(self.time_fs,'time',True))
        if self.solver not in ('exact','stochastic-kpm') or self.backend not in ('cpu','hybrid','gpu') or self.kernel not in ('gaussian','jackson'):
            raise KuboMSDError('unsupported solver/backend/kernel')
        if self.stochastic_method not in ('auto','stepwise','spectral'): raise KuboMSDError('unsupported stochastic method')
        if self.cpu_kernel not in ('cpp','reference'): raise KuboMSDError('cpu_kernel must be cpp or reference')
        if isinstance(self.omp_threads,bool) or not isinstance(self.omp_threads,(int,np.integer)) or self.omp_threads<1:
            raise KuboMSDError('omp_threads must be a positive integer')
        if self.transport_distribution not in ('serial','mpi'): raise KuboMSDError('transport_distribution must be serial or mpi')
        if self.transport_distribution=='mpi' and self.solver!='stochastic-kpm':
            raise KuboMSDError('MPI transport currently requires stochastic-kpm; distributed EXACT needs a distributed eigensolver')
        if self.initial_state_selection not in ('all','window'): raise KuboMSDError('invalid initial_state_selection')
        if self.solver=='stochastic-kpm' and (self.backend=='hybrid' or self.kernel!='jackson' or self.initial_state_selection!='all'):
            raise KuboMSDError('stochastic-kpm requires Jackson, cpu/gpu, full-space sampling')
        for name in ('n_moments','nsamples','sample_batch','statistics_block','pair_block','time_block','energy_block','metric_maxiter','sampling_max_order','propagation_max_order'):
            v=getattr(self,name)
            if isinstance(v,bool) or not isinstance(v,(int,np.integer)) or v<1: raise KuboMSDError(f'{name} must be a positive integer')
        if not isinstance(self.seed,(int,np.integer)) or self.seed<0: raise KuboMSDError('seed must be a nonnegative integer')
        for name in ('sigma_eV','metric_rtol','sampling_tol','propagation_tol','segment_fs','integration_dt_fs','memory_limit_gb'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0: raise KuboMSDError(f'{name} must be positive and finite')
        for name in ('buffer_eV','metric_atol','rho_abs','rho_rel'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<0: raise KuboMSDError(f'{name} must be nonnegative and finite')
        if self.preconditioner not in ('diagonal','block-jacobi','full-cholesky','cluster-jacobi','ic0'): raise KuboMSDError('unknown preconditioner')
        if self.metric_ic_ordering not in ('natural','rcm'):raise KuboMSDError('invalid metric_ic_ordering')
        if not np.isfinite(self.metric_ic_shift_max) or self.metric_ic_shift_max<0:raise KuboMSDError('metric_ic_shift_max must be finite nonnegative')
        if self.preconditioner=='ic0' and self.solver!='stochastic-kpm':raise KuboMSDError('ic0 requires stochastic-kpm')
        if self.preconditioner=='cluster-jacobi':
            if self.solver!='stochastic-kpm' or not isinstance(self.metric_partition_file,str) or not self.metric_partition_file:
                raise KuboMSDError('cluster-jacobi requires stochastic-kpm and metric_partition_file')
        elif self.metric_partition_file is not None:raise KuboMSDError('partition file requires cluster-jacobi')
        if self.metric_residual_policy not in ('fixed','adaptive'):
            raise KuboMSDError('metric_residual_policy must be fixed or adaptive')
        if isinstance(self.metric_check_interval,bool) or not isinstance(self.metric_check_interval,(int,np.integer)) or self.metric_check_interval<1:
            raise KuboMSDError('metric_check_interval must be a positive integer')
        if not np.isfinite(self.metric_gap_ratio) or not 0<self.metric_gap_ratio<1:
            raise KuboMSDError('metric_gap_ratio must be strictly between zero and one')
        if isinstance(self.metric_gpu_check_batch,bool) or not isinstance(self.metric_gpu_check_batch,(int,np.integer)) or self.metric_gpu_check_batch<1:
            raise KuboMSDError('metric_gpu_check_batch must be a positive integer')
        if self.metric_solver not in ('pcg','cholesky'): raise KuboMSDError('unknown metric_solver')
        if isinstance(self.metric_refine_maxiter,bool) or not isinstance(self.metric_refine_maxiter,(int,np.integer)) or self.metric_refine_maxiter<0:
            raise KuboMSDError('metric_refine_maxiter must be a nonnegative integer')
        if self.metric_solver=='cholesky':
            if self.solver!='stochastic-kpm':
                raise KuboMSDError('metric_solver=cholesky requires stochastic-kpm')
            if (self.preconditioner!='diagonal' or self.metric_residual_policy!='fixed' or
                self.metric_check_interval!=32 or self.metric_gap_ratio!=.1 or
                self.metric_ic_ordering!='rcm' or self.metric_ic_shift_max!=1e-2 or self.metric_gpu_check_batch!=1):
                raise KuboMSDError('cholesky requires default PCG options; preconditioner is not applicable')
        if self.preconditioner=='full-cholesky' and (self.solver!='exact' or self.backend!='cpu'):
            raise KuboMSDError('full-cholesky is supported only for CPU EXACT')
        if self.exact_projection_kernel not in ('sparse','dense'):
            raise KuboMSDError('unknown exact_projection_kernel')
        if self.exact_projection_kernel=='dense' and (self.solver!='exact' or self.backend!='cpu'):
            raise KuboMSDError('dense projection is supported only for CPU EXACT')
        if self.spectral_bounds is not None:
            b=tuple(float(v) for v in self.spectral_bounds)
            if len(b)!=2 or not np.isfinite(b).all() or b[0]>=b[1]: raise KuboMSDError('invalid full spectral bounds')
            object.__setattr__(self,'spectral_bounds',b)
        axes=tuple(self.axes)
        if not axes or len(set(axes))!=len(axes): raise KuboMSDError('axes must be nonempty and unique')
        if self.axes_vectors is None:
            if any(a not in 'xyz' or len(a)!=1 for a in axes): raise KuboMSDError('custom axes require axes_vectors')
            v=np.array([np.eye(3)['xyz'.index(a)] for a in axes])
        else: v=np.array(self.axes_vectors,dtype=float,copy=True)
        if v.shape!=(len(axes),3) or not np.isfinite(v).all() or not np.allclose(v@v.T,np.eye(len(axes)),atol=1e-12,rtol=0):
            raise KuboMSDError('direction vectors must be orthonormal')
        v.setflags(write=False); object.__setattr__(self,'axes_vectors',v); object.__setattr__(self,'axes',axes)
        if self.efermi_eV is not None and (not np.isfinite(self.efermi_eV) or not self.efermi_source): raise KuboMSDError('EF needs finite value and provenance')

    def as_dict(self):
        return {f.name:(getattr(self,f.name).tolist() if isinstance(getattr(self,f.name),np.ndarray) else getattr(self,f.name)) for f in fields(self)}
