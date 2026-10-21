"""Segmented VAC with trace-first Chebyshev product tables.

T_l T_p = (T_(l+p) + T_|l-p|)/2. Keeping all l terms applies the
same finite Jackson kernel without squaring it. Left chains are shared
by axes. Right velocity chains are axis-specific; no full diagonalization.
"""
import time
import numpy as np
from ..config import resolve_time_grid
from ..kernels.energy import reconstruction_matrix
from ..errors import NumericalError
from .chebyshev import chain,bessel_coefficients,projected_quadratic
from .stepwise import propagate

def spectral_samples(ops,cfg,samples):
    xp=ops.xp; samples=ops.backend.asarray(samples,dtype=xp.complex128); nb=samples.shape[1]; na=len(cfg.axes); ne=len(cfg.energy_eV); nt=len(cfg.time_fs); L=cfg.n_moments
    t0=time.perf_counter(); dos=projected_quadratic(samples,ops,cfg)
    output={key:xp.zeros((nb,na,nt,ne)) for key in ('msd_A2/vac_integral','diffusion_A2_per_fs/vac_integral','vac_A2_per_fs2/vac_integral')}
    times=np.unique(np.r_[resolve_time_grid(cfg.time_fs[-1],cfg.integration_dt_fs),cfg.time_fs]); output_indices={float(t):j for j,t in enumerate(cfg.time_fs)}
    G=ops.backend.asarray(reconstruction_matrix(cfg.energy_eV,cfg.spectral_bounds,L)); c_anchor=samples; v_anchor=[ops.apply_v_action(a,samples) for a in cfg.axes]
    anchor=0.; next_anchor=-1.; tables=None; maxorder=0; segments=0; maxnorm=0.
    M=xp.zeros((nb,na,ne)); D=xp.zeros_like(M); previous=None; prev_t=0.
    for ti,t in enumerate(times):
        if tables is None or t>next_anchor+1e-14:
            new_anchor=0. if tables is None else float(prev_t)
            if new_anchor>anchor:
                both=xp.concatenate([c_anchor]+v_anchor,axis=1)
                both,report=propagate(both,new_anchor-anchor,ops,cfg); maxnorm=max(maxnorm,report['relative_S_norm_drift'])
                c_anchor=both[:,:nb]; v_anchor=[both[:,(i+1)*nb:(i+2)*nb] for i in range(na)]
            anchor=new_anchor; next_anchor=min(max(anchor+cfg.segment_fs,float(t)),float(times[-1])); coeff,report=bessel_coefficients(next_anchor-anchor,cfg); K=len(coeff)-1; maxorder=max(maxorder,K)
            estimate=16*nb*(ops.n*(L+2*K+1)+na*(L+K)*(K+1))+24*nb*na*nt*ne
            if estimate>cfg.memory_limit_gb*1024**3: raise NumericalError('spectral workspace exceeds memory budget; reduce sample_batch/segment_fs or increase explicit budget')
            left=chain(c_anchor,ops,cfg,L+K); tables=[]
            for ai,axis in enumerate(cfg.axes):
                right=chain(v_anchor[ai],ops,cfg,K+1)
                cov=ops.apply_v_cov(axis,right.transpose(1,0,2).reshape(ops.n,-1)).reshape(ops.n,K+1,nb).transpose(1,0,2)
                tables.append(xp.einsum('jnb,knb->bjk',left.conj(),cov))
            segments+=1
        b,rep=bessel_coefficients(float(t-anchor),cfg); Kt=len(b)-1
        # A time inside a segment must not require more terms than its endpoint.
        if Kt>tables[0].shape[-1]-1: raise NumericalError('internal spectral propagation order exceeded segment allocation')
        b=ops.backend.asarray(b); vac=xp.empty((nb,na,ne)); l=xp.arange(L)[:,None]; p=xp.arange(Kt+1)[None,:]
        for ai,table in enumerate(tables):
            q=table[:,:,:Kt+1]@b
            moments=xp.sum(.5*(q[:,l+p]+q[:,xp.abs(l-p)])*b.conj()[None,None,:],axis=2).real
            vac[:,ai,:]=moments@G.T
        if previous is not None:
            h=float(t-prev_t); M=M+2*h*D+h*h/3*(2*previous+vac); D=D+h/2*(previous+vac)
        if float(t) in output_indices:
            j=output_indices[float(t)]; output['msd_A2/vac_integral'][:,:,j,:]=M; output['diffusion_A2_per_fs/vac_integral'][:,:,j,:]=D; output['vac_A2_per_fs2/vac_integral'][:,:,j,:]=vac
        previous=vac; prev_t=t
    if not cfg.include_vac: output.pop('vac_A2_per_fs2/vac_integral')
    return dos,output,{'algorithm':'segmented trace-first Chebyshev product VAC','segments':segments,'max_propagation_order':maxorder,'max_relative_S_norm_drift':maxnorm,'integration_scheme':'piecewise linear VAC, simultaneous M/D recurrence','internal_nodes':len(times),'integration_dt_fs':cfg.integration_dt_fs,'integration_validation':'UNVERIFIED until step-size convergence','segment_policy':'at least one integration interval; propagation itself subdivides at segment_fs','seconds':time.perf_counter()-t0}

def iter_spectral_blocks(ops,config,samples=None):
    from .stochastic import iter_stochastic_blocks
    yield from iter_stochastic_blocks(ops,config,'spectral',samples)
