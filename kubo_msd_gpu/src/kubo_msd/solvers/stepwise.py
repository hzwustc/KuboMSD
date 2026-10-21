"""Direct velocity-driven commutator recursion; no eigenvectors or energy-filtered samples."""
import time
import numpy as np
from ..units import HBAR_EV_FS
from ..errors import NumericalError
from .chebyshev import scaling,scaled_action,bessel_coefficients,evaluate_series,projected_quadratic

def propagate(states,dt_fs,ops,prop_config):
    cfg=prop_config; xp=ops.xp; B=xp.asarray(states,dtype=xp.complex128); initial=ops.inner(B,B).real
    nsegments=max(1,int(np.ceil(abs(dt_fs)/cfg.segment_fs))); dt=dt_fs/nsegments
    coeff,diag=bessel_coefficients(dt,cfg)
    for _ in range(nsegments):
        if ops.native is not None:
            c,a=scaling(cfg); B=ops.native.series(B,coeff,c,a)
        else: B=evaluate_series(B,lambda x:scaled_action(ops,cfg,x),coeff,xp)
    final=ops.inner(B,B).real; drift=float(xp.max(xp.abs(final-initial)/xp.maximum(initial,xp.finfo(float).tiny)))
    if drift>1e-7: raise NumericalError(f'S-norm propagation drift {drift:g}; bounds/solver/propagation need correction')
    return B,{**diag,'segments':nsegments,'relative_S_norm_drift':drift}

def stepwise_samples(ops,cfg,samples):
    xp=ops.xp; initial=xp.asarray(samples,dtype=xp.complex128); nb=initial.shape[1]; na=len(cfg.axes); nt=len(cfg.time_fs); ne=len(cfg.energy_eV)
    dos=projected_quadratic(initial,ops,cfg); total=xp.concatenate([initial]+[xp.zeros_like(initial) for _ in cfg.axes],axis=1)
    output=xp.zeros((nb,na,nt,ne),dtype=xp.float64); c,a=scaling(cfg); orders=[]; norm0=ops.inner(initial,initial).real; drift=0.; segments=0; t0=time.perf_counter()
    def augmented(B):
        result=scaled_action(ops,cfg,B)
        for ai,axis in enumerate(cfg.axes): result[:,(ai+1)*nb:(ai+2)*nb]+=(-1j*HBAR_EV_FS/a)*ops.apply_v_action(axis,B[:,:nb])
        return result
    for j,t in enumerate(cfg.time_fs):
        if j:
            span=t-cfg.time_fs[j-1]; count=max(1,int(np.ceil(span/cfg.segment_fs))); coeff,report=bessel_coefficients(span/count,cfg,weighted_tail=True); orders.append(report['order'])
            for _ in range(count):
                total=(ops.native.series(total,coeff,c,a,cfg.axes_vectors) if ops.native is not None
                       else evaluate_series(total,augmented,coeff,xp))
                segments+=1
            norm=ops.inner(total[:,:nb],total[:,:nb]).real; drift=max(drift,float(xp.max(xp.abs(norm-norm0)/xp.maximum(norm0,xp.finfo(float).tiny))))
            if drift>1e-7: raise NumericalError(f'stepwise base propagation S-norm drift {drift:g}')
        for ai in range(na): output[:,ai,j,:]=projected_quadratic(total[:,(ai+1)*nb:(ai+2)*nb],ops,cfg)
    return dos,{'msd_A2/commutator':output},{'algorithm':'velocity-driven augmented Chebyshev commutator','segments':segments,'max_propagation_order':max(orders,default=0),'max_relative_S_norm_drift':drift,'seconds':time.perf_counter()-t0}

def iter_stepwise_blocks(ops,config,samples=None):
    from .stochastic import iter_stochastic_blocks
    yield from iter_stochastic_blocks(ops,config,'stepwise',samples)
