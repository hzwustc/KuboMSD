"""Full generalized eigensystem with optional outer-state selection only."""
from dataclasses import replace
from types import SimpleNamespace
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from ..validation import hermitian
from ..errors import NumericalError
from ..contracts import TransportBlock,Reason
from ..kernels.energy import energy_weights,dos_floor,kernel_resolution,validate_spectral_window
from ..kernels.time import pair_kernels
from ..units import HBAR_EV_FS


def eigen_velocity_block(ops,axis,left,right,e_left,e_right,*,left_adjoint=None):
    """Project covariant velocity using H C = S C eps, with no metric solves.

    Both sides must be accepted generalized eigenvectors of the same H/S.
    This cancellation is specific to EXACT; it is not an approximation to the
    metric action on arbitrary vectors used by stochastic propagation.
    """
    xp=ops.xp
    bra=left.conj().T if left_adjoint is None else left_adjoint
    result=xp.zeros((left.shape[1],right.shape[1]),dtype=xp.complex128)
    for cart,coefficient in zip('xyz',ops.directions[axis]):
        if not coefficient:continue
        def project(matrix):
            source=matrix
            if xp is np and not np.iscomplexobj(right) and sp.issparse(matrix) and np.iscomplexobj(matrix.data) and not np.any(matrix.data.imag):
                source=matrix.real
            applied=source@right
            # Real Gamma eigenvectors and exactly real applied columns should
            # use real BLAS. Check the actual product, not a tolerance; a tiny
            # nonzero imaginary part still takes the complex path.
            if xp is np and not np.iscomplexobj(bra) and np.iscomplexobj(applied) and not np.any(applied.imag):
                applied=applied.real
            return bra@applied
        if cart in ops.velocity:
            value=project(ops.velocity[cart])
        elif cart in ops.position:
            value=1j*(e_left[:,None]-e_right[None,:])*project(ops.position[cart])/HBAR_EV_FS
        else:
            bond,xbar,delta=ops.composite[cart]
            value=project(bond).astype(xp.complex128,copy=False)
            value-=project(xbar)*e_right[None,:]
            value+=e_left[:,None]*project(delta)
            value=value*(1j/HBAR_EV_FS)
        result+=coefficient*value
    return result

def solve_generalized(H,S,backend='cpu',*,device_backend=None):
    dense_seconds=0.
    if backend in ('cpu','hybrid'):
        H=hermitian(H,'H'); S=hermitian(S,'S')
        # Gamma scalar inputs can be exactly real even though the cache uses
        # complex128. Preserve the real problem for LAPACK; never discard a
        # nonzero imaginary component, however small.
        if not np.any(H.data.imag) and not np.any(S.data.imag):
            H=H.real;S=S.real
        H=H.toarray(); S=S.toarray()
        eps,C=la.eigh(H,S,driver='gvd'); xp=np; HH=H; SS=S
    else:
        from ..backends.gpu import GPUBackend
        device=device_backend or GPUBackend(); xp=device.xp
        if not device.sparse.issparse(H):H=hermitian(H,'H')
        if not device.sparse.issparse(S):S=hermitian(S,'S')
        H=device.matrix(H,name='H');S=device.matrix(S,name='S')
        for a,name in ((H,'H'),(S,'S')):
            if a.shape[0]!=a.shape[1] or not bool(xp.all(xp.isfinite(a.data))):raise NumericalError('invalid '+name)
            diff=a-a.conj().T
            err=float(xp.linalg.norm(diff.data)/xp.maximum(xp.linalg.norm(a.data),1e-300))
            if err>1e-10:raise NumericalError('non-Hermitian device '+name)
        device.synchronize();dense_start=time.perf_counter()
        HH=device.device_dense(H,'H');SS=device.device_dense(S,'S')
        dense_seconds=time.perf_counter()-dense_start
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
    diag={'max_normalized_residual':float(xp.max(residual)),'metric_orthogonality':float(orth),'residual_norm':'Frobenius matrix bound; Euclidean vector norm','device_dense_seconds':dense_seconds,'solver_matrix_dtype':str(HH.dtype)}
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

def projection_matrix_count(ops,axis):
    return sum(1 if c in ops.velocity or c in ops.position else 3
               for c,v in zip('xyz',ops.directions[axis]) if v)


def dense_projection_operators(ops,axis):
    """One CPU output axis at a time; caller reserves worst-case complex bytes."""
    out=SimpleNamespace(xp=np,directions=ops.directions,velocity={},position={},composite={})
    def dense(m):
        # Signed real views preserve every entry. Nonzero imaginary parts remain.
        values=m.data if sp.issparse(m) else np.asarray(m)
        source=m.real if not np.any(values.imag) else m
        return source.toarray() if sp.issparse(source) else np.asarray(source)
    for cart,coefficient in zip('xyz',ops.directions[axis]):
        if not coefficient:continue
        if cart in ops.velocity:out.velocity[cart]=dense(ops.velocity[cart])
        elif cart in ops.position:out.position[cart]=dense(ops.position[cart])
        else:out.composite[cart]=tuple(dense(m) for m in ops.composite[cart])
    return out


def iter_exact_blocks(model,ops,config):
    cfg=config; backend=ops.backend; xp=ops.xp; n=ops.n; t0=time.perf_counter(); limit=cfg.memory_limit_gb*1024**3
    estimate=16*n*n*(7+len(cfg.axes)+(cfg.preconditioner=='full-cholesky'))+8*len(cfg.energy_eV)*n+24*cfg.time_block*min(cfg.pair_block,n)**2
    dense_count=max(projection_matrix_count(ops,a) for a in cfg.axes) if cfg.exact_projection_kernel=='dense' else 0
    estimate+=16*n*n*dense_count
    if estimate>limit: raise NumericalError(f'EXACT working-memory estimate {estimate/1024**3:.3g} GiB exceeds memory_limit_gb={cfg.memory_limit_gb}; set an intentional larger budget or smaller blocks')
    if cfg.backend!='cpu':
        free=backend.memory_info()['free_bytes']
        if estimate>free*.85: raise NumericalError('EXACT estimate exceeds available GPU memory; choose smaller blocks or another explicit backend')
    eps,C,diagnostics=solve_generalized(ops.H if cfg.backend=='gpu' else ops.host_H,ops.S if cfg.backend=='gpu' else ops.host_S,cfg.backend,device_backend=backend if cfg.backend=='gpu' else None)
    if cfg.backend=='hybrid': eps=backend.asarray(eps); C=backend.asarray(C)
    eig_seconds=time.perf_counter()-t0; e=backend.to_host(eps)
    bounds=cfg.spectral_bounds
    if bounds is not None: validate_spectral_window(e,bounds)
    elif cfg.kernel=='jackson':
        width=e[-1]-e[0]
        if width<=np.finfo(float).eps*max(1,abs(e[0]),abs(e[-1])): raise NumericalError('degenerate spectrum needs explicit spectral bounds for Jackson')
        bounds=(float(e[0]-.01*width),float(e[-1]+.01*width))
        cfg=replace(cfg,spectral_bounds=bounds)
    broad=energy_weights(cfg.energy_eV,e,cfg); rho=broad.sum(axis=1)
    if np.min(rho)<-1e-12*max(1,float(np.max(rho))): raise NumericalError('negative deterministic DOS beyond roundoff')
    width=(bounds[1]-bounds[0]) if bounds is not None else float(e[-1]-e[0])
    threshold=dos_floor(n,width,cfg); valid=rho>threshold; reason=np.where(valid,0,int(Reason.LOW_DOS)).astype('uint32')
    deg_tol=max(1e-12,100*np.finfo(float).eps*max(1,np.max(np.abs(e))))
    selected=np.arange(n) if cfg.initial_state_selection=='all' else select_initial_states(e,(cfg.energy_eV[0],cfg.energy_eV[-1]),cfg.buffer_eV,deg_tol)
    if cfg.initial_state_selection=='window': reason|=int(Reason.TRUNCATION_UNVERIFIED)
    if not len(selected): reason|=int(Reason.NO_SELECTED_STATES); valid[:]=False
    rho_selected=broad[:,selected].sum(axis=1); retained=np.divide(rho_selected,rho,out=np.full_like(rho,np.nan),where=rho>0)
    F=backend.asarray(broad[:,selected],name='energy weights'); denom=backend.asarray(np.where(valid,rho,np.nan),name='DOS denominator'); t1=time.perf_counter()
    W=xp.empty((len(cfg.axes),len(selected),n),dtype=xp.float64); CS=C[:,selected]; CS_adjoint=CS.conj().T
    herm_errors=[]
    for ai,axis in enumerate(cfg.axes):
        projection_ops=dense_projection_operators(ops,axis) if dense_count else ops
        for m0 in range(0,n,cfg.pair_block):
            Cm=C[:,m0:m0+cfg.pair_block]
            projected=eigen_velocity_block(projection_ops,axis,CS,Cm,eps[selected],eps[m0:m0+cfg.pair_block],left_adjoint=CS_adjoint)
            W[ai,:,m0:m0+cfg.pair_block]=xp.abs(projected)**2
        del projection_ops
        # Composite matrix Hermiticity is checked with deterministic probes too.
        probe=backend.asarray(np.random.default_rng(551).normal(size=(n,min(n,4)))+1j*np.random.default_rng(552).normal(size=(n,min(n,4))))
        p=probe.conj().T@ops.apply_v_cov(axis,probe); error=float(xp.linalg.norm(p-p.conj().T)/xp.maximum(xp.linalg.norm(p),1e-300)); herm_errors.append(error)
        if error>1e-9: raise NumericalError(f'constructed velocity {axis} is not Hermitian; relative probe error {error:g}')
    transform_seconds=time.perf_counter()-t1
    metadata={'primary_estimator':'pair_analytic','resolved_config':cfg.as_dict(),'eigensolve':diagnostics,'spectral_bounds_source':'user validated against full eigenspectrum' if config.spectral_bounds is not None else 'full eigenspectrum plus 1 percent margin' if bounds else 'Gaussian does not require spectral scaling','kernel_resolution':kernel_resolution(cfg),'dos_floor':threshold,'eigensolve_seconds':eig_seconds,'velocity_transform_seconds':transform_seconds,'velocity_transform_method':'generalized eigenpair identity; no per-state metric solves','exact_projection_kernel':cfg.exact_projection_kernel,'projection_extra_memory_budget_bytes':16*n*n*dense_count,'velocity_probe_hermiticity':herm_errors,'memory_estimate_bytes':estimate,'degeneracy_tolerance_eV':deg_tol,'science_status':{'NUMERICS_VERIFIED':'UNVERIFIED for this run','STATISTICS_VERIFIED':'not applicable','TRUNCATION_VERIFIED':'NOT RUN' if cfg.initial_state_selection=='window' else 'not applicable'}}
    static={'dos_states_per_eV':rho,'initial_state_indices':selected,'dos_selected':rho_selected,'retained_weight':retained,'eigenvalues_eV':e,'sample_count':np.array(1)}
    for tstart in range(0,len(cfg.time_fs),cfg.time_block):
        times=cfg.time_fs[tstart:tstart+cfg.time_block]; nt=len(times)
        numer=[xp.zeros((len(cfg.axes),nt,len(cfg.energy_eV))) for _ in range(3)]
        for n0 in range(0,len(selected),cfg.pair_block):
            ids=selected[n0:n0+cfg.pair_block]; q=[xp.zeros((len(cfg.axes),nt,len(ids))) for _ in range(3)]
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
