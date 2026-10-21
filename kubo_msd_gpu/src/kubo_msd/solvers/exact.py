"""Full generalized eigensystem with optional outer-state selection only."""
from dataclasses import replace
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from ..validation import hermitian
from ..errors import NumericalError
from ..contracts import TransportBlock,Reason
from ..kernels.energy import energy_weights,dos_floor,kernel_resolution,validate_spectral_window
from ..kernels.time import pair_kernels

def solve_generalized_gpu(model, context):
    from cupyx.scipy.linalg import solve_triangular
    import cupyx
    xp=context.xp; n=model.n_orbitals
    context.check_device(model.H_cov); context.check_device(model.S)
    context.require_bytes(16*n*n*10,'EXACT irreducible dense eigensystem plus library workspace')
    with context.phase('eigensolve'), cupyx.errstate(linalg='raise'):
        H=model.H_cov.toarray(); S=model.S.toarray()
        L=xp.linalg.cholesky(S)
        T=solve_triangular(L,H,lower=True)
        Ho=solve_triangular(L,T.conj().T,lower=True).conj().T
        del T
        eps,Y=xp.linalg.eigh(Ho)
        del Ho
        C=solve_triangular(L.conj().T,Y,lower=False)
        del L,Y
        # Residual/orthogonality in bounded column slabs, no extra full H@C/S@C.
        hnorm=xp.linalg.norm(H); snorm=xp.linalg.norm(S)
        maxres=0.; orth2=0.
        for first in range(0,n,128):
            c=C[:,first:first+128]; sc=S@c
            nr=xp.linalg.norm(H@c-sc*eps[first:first+128],axis=0)
            denom=(hnorm+xp.abs(eps[first:first+128])*snorm)*xp.linalg.norm(c,axis=0)
            r=xp.where(denom>0,nr/xp.where(denom>0,denom,1),nr)
            maxres=max(maxres,float(xp.max(r)))
            defect=C.conj().T@sc
            ids=xp.arange(c.shape[1]); defect[first+ids,ids]-=1
            orth2+=float(xp.sum(xp.abs(defect)**2))
        orth=float(np.sqrt(orth2/n))
    report={'max_normalized_residual':maxres,'metric_orthogonality':orth,
            'residual_norm':'Frobenius matrix bound; Euclidean vector norm',
            'location':'device CSR densification, Cholesky, eigh, back-transform',
            'dense_budget_policy':'10 complex N^2 arrays, conservative library workspace estimate'}
    if maxres>1e-10 or orth>1e-10 or not np.isfinite([maxres,orth]).all():
        raise NumericalError(f'generalized eigensolve failed acceptance: {report}')
    return eps,C,report

def solve_generalized(H,S,backend='cpu'):
    H=hermitian(H,'H').toarray(); S=hermitian(S,'S').toarray()
    if backend in ('cpu','hybrid'):
        eps,C=la.eigh(H,S,driver='gvd'); xp=np; HH=H; SS=S
    else:
        from ..backends.gpu import GPUBackend
        device=GPUBackend(); xp=device.xp; HH=device.asarray(H); SS=device.asarray(S)
        from cupyx.scipy.linalg import solve_triangular
        import cupyx
        with cupyx.errstate(linalg='raise'):
            L=xp.linalg.cholesky(SS)
            T=solve_triangular(L,HH,lower=True)
            Ho=solve_triangular(L,T.conj().T,lower=True).conj().T
            eps,Y=xp.linalg.eigh(Ho)
            C=solve_triangular(L.conj().T,Y,lower=False)
    nr=xp.linalg.norm(HH@C-(SS@C)*eps,axis=0)
    denom=(xp.linalg.norm(HH)+xp.abs(eps)*xp.linalg.norm(SS))*xp.linalg.norm(C,axis=0)
    residual=xp.where(denom>0,nr/xp.where(denom>0,denom,1),nr)
    orth=xp.linalg.norm(C.conj().T@SS@C-xp.eye(len(eps)))/xp.sqrt(len(eps))
    diag={'max_normalized_residual':float(xp.max(residual)),'metric_orthogonality':float(orth),'residual_norm':'Frobenius matrix bound; Euclidean vector norm'}
    if diag['max_normalized_residual']>1e-10 or diag['metric_orthogonality']>1e-10: raise NumericalError(f'generalized eigensolve failed acceptance: {diag}')
    return eps,C,diag

def select_initial_states(eps,output_bounds,buffer_eV,degeneracy_tol):
    eps=np.asarray(eps); mask=(eps>=output_bounds[0]-buffer_eV)&(eps<=output_bounds[1]+buffer_eV)
    # Include transitive adjacent near-degenerate cluster at either boundary.
    for j in range(1,len(eps)):
        if eps[j]-eps[j-1]<=degeneracy_tol and mask[j-1]: mask[j]=True
    for j in range(len(eps)-2,-1,-1):
        if eps[j+1]-eps[j]<=degeneracy_tol and mask[j+1]: mask[j]=True
    return np.flatnonzero(mask)

def iter_exact_blocks(model,ops,config):
    cfg=config; backend=ops.backend; xp=ops.xp; n=ops.n; t0=time.perf_counter(); limit=cfg.memory_limit_gb*1024**3
    estimate=16*n*n*(7+len(cfg.axes))+8*len(cfg.energy_eV)*n+24*cfg.time_block*min(cfg.pair_block,n)**2
    if estimate>limit: raise NumericalError(f'EXACT working-memory estimate {estimate/1024**3:.3g} GiB exceeds memory_limit_gb={cfg.memory_limit_gb}; set an intentional larger budget or smaller blocks')
    resident=hasattr(ops,'context')
    if cfg.backend!='cpu' and not resident:
        free=backend.memory_info()['free_bytes']
        if estimate>free*.85: raise NumericalError('EXACT estimate exceeds available GPU memory; choose smaller blocks or another explicit backend')
    if resident:
        backend.require_bytes(estimate,'EXACT transport workspaces',int(limit))
        eps,C,diagnostics=solve_generalized_gpu(model,backend)
    else: eps,C,diagnostics=solve_generalized(ops.host_H,ops.host_S,cfg.backend)
    if cfg.backend=='hybrid': eps=backend.asarray(eps); C=backend.asarray(C)
    eig_seconds=time.perf_counter()-t0; e=backend.to_host(eps)
    bounds=cfg.spectral_bounds
    if bounds is not None: validate_spectral_window(e,bounds)
    elif cfg.kernel=='jackson':
        width=e[-1]-e[0]
        if width<=np.finfo(float).eps*max(1,abs(e[0]),abs(e[-1])): raise NumericalError('degenerate spectrum needs explicit spectral bounds for Jackson')
        bounds=(float(e[0]-.01*width),float(e[-1]+.01*width))
        cfg=replace(cfg,spectral_bounds=bounds)
    if resident:
        from ..kernels.energy_gpu import energy_weights_gpu
        with backend.phase('energy_weights'):
            broad=energy_weights_gpu(cfg.energy_eV,eps,cfg,backend)
        rho=backend.to_host(broad.sum(axis=1),'DOS','result_output')
    else: broad=energy_weights(cfg.energy_eV,e,cfg); rho=broad.sum(axis=1)
    if np.min(rho)<-1e-12*max(1,float(np.max(rho))): raise NumericalError('negative deterministic DOS beyond roundoff')
    width=(bounds[1]-bounds[0]) if bounds is not None else float(e[-1]-e[0])
    threshold=dos_floor(n,width,cfg); valid=rho>threshold; reason=np.where(valid,0,int(Reason.LOW_DOS)).astype('uint32')
    deg_tol=max(1e-12,100*np.finfo(float).eps*max(1,np.max(np.abs(e))))
    selected=np.arange(n) if cfg.initial_state_selection=='all' else select_initial_states(e,(cfg.energy_eV[0],cfg.energy_eV[-1]),cfg.buffer_eV,deg_tol)
    if cfg.initial_state_selection=='window': reason|=int(Reason.TRUNCATION_UNVERIFIED)
    if not len(selected): reason|=int(Reason.NO_SELECTED_STATES); valid[:]=False
    sel=backend.asarray(selected,'initial_state_indices','metadata') if resident else selected
    F=broad if cfg.initial_state_selection=='all' else broad[:,sel]
    rho_selected=F.sum(axis=1)
    if resident: rho_selected=backend.to_host(rho_selected,'selected_DOS')
    retained=np.divide(rho_selected,rho,out=np.full_like(rho,np.nan),where=rho>0)
    F=xp.asarray(F); denom=backend.asarray(np.where(valid,rho,np.nan)) if resident else xp.asarray(np.where(valid,rho,np.nan)); t1=time.perf_counter()
    W=xp.empty((len(cfg.axes),len(selected),n),dtype=xp.float64); CS=C if cfg.initial_state_selection=='all' else C[:,sel]
    herm_errors=[]
    for ai,axis in enumerate(cfg.axes):
        for m0 in range(0,n,cfg.pair_block):
            Cm=C[:,m0:m0+cfg.pair_block]; Vc=ops.apply_v_cov(axis,Cm)
            W[ai,:,m0:m0+cfg.pair_block]=xp.abs(CS.conj().T@Vc)**2
        # Composite matrix Hermiticity is checked with deterministic probes too.
        host_probe=np.random.default_rng(551).normal(size=(n,min(n,4)))+1j*np.random.default_rng(552).normal(size=(n,min(n,4)))
        probe=backend.asarray(host_probe) if resident else xp.asarray(host_probe)
        p=probe.conj().T@ops.apply_v_cov(axis,probe); error=float(xp.linalg.norm(p-p.conj().T)/xp.maximum(xp.linalg.norm(p),1e-300)); herm_errors.append(error)
        if error>1e-10: raise NumericalError(f'constructed velocity {axis} is not Hermitian; relative probe error {error:g}')
    transform_seconds=time.perf_counter()-t1
    metadata={'primary_estimator':'pair_analytic','resolved_config':cfg.as_dict(),'eigensolve':diagnostics,'spectral_bounds_source':'user validated against full eigenspectrum' if config.spectral_bounds is not None else 'full eigenspectrum plus 1 percent margin' if bounds else 'Gaussian does not require spectral scaling','kernel_resolution':kernel_resolution(cfg),'dos_floor':threshold,'eigensolve_seconds':eig_seconds,'velocity_transform_seconds':transform_seconds,'velocity_probe_hermiticity':herm_errors,'memory_estimate_bytes':estimate,'degeneracy_tolerance_eV':deg_tol,'science_status':{'NUMERICS_VERIFIED':'UNVERIFIED for this run','STATISTICS_VERIFIED':'not applicable','TRUNCATION_VERIFIED':'NOT RUN' if cfg.initial_state_selection=='window' else 'not applicable'}}
    static={'dos_states_per_eV':rho,'initial_state_indices':selected,'dos_selected':rho_selected,'retained_weight':retained,'eigenvalues_eV':e,'sample_count':np.array(1)}
    for tstart in range(0,len(cfg.time_fs),cfg.time_block):
        times=cfg.time_fs[tstart:tstart+cfg.time_block]; nt=len(times)
        numer=[xp.zeros((len(cfg.axes),nt,len(cfg.energy_eV))) for _ in range(3)]
        for n0 in range(0,len(selected),cfg.pair_block):
            ids=sel[n0:n0+cfg.pair_block]; q=[xp.zeros((len(cfg.axes),nt,len(ids))) for _ in range(3)]
            for m0 in range(0,n,cfg.pair_block):
                delta=eps[ids,None]-eps[None,m0:m0+cfg.pair_block]
                factors=pair_kernels(delta,times,xp); weights=W[:,n0:n0+len(ids),m0:m0+cfg.pair_block]
                for ki,factor in enumerate(factors): q[ki]+=xp.einsum('anm,tnm->atn',weights,factor)
            for E0 in range(0,len(cfg.energy_eV),cfg.energy_block):
                ff=F[E0:E0+cfg.energy_block,n0:n0+len(ids)]
                for ki in range(3): numer[ki][:,:,E0:E0+len(ff)]+=q[ki]@ff.T
        arrays={}
        for key,num in zip(('msd_A2','diffusion_A2_per_fs','vac_A2_per_fs2'),numer):
            if key.startswith('vac') and not cfg.include_vac: continue
            raw=backend.to_host(num); arrays['numerator_'+key+'/pair_analytic']=raw
            arrays[key+'/pair_analytic']=backend.to_host(num/denom)
        shape=(len(cfg.axes),nt,len(cfg.energy_eV)); arrays['valid/pair_analytic']=np.broadcast_to(valid,shape).copy(); arrays['reason_code/pair_analytic']=np.broadcast_to(reason,shape).copy()
        yield TransportBlock(tstart,times,arrays,static if tstart==0 else {},metadata if tstart==0 else {})
