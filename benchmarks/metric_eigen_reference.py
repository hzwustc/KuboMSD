"""Independent full-eigenbasis reference for synthetic transport tests."""
import numpy as np
from kubo_msd.units import HBAR_EV_FS


def project(C,X,block=256):
    out=np.empty((C.shape[1],X.shape[1]),complex)
    for i in range(0,C.shape[1],block):
        c=C[:,i:i+block]
        out[i:i+block]=c.conj().T@X
    return out


def verify_eigenpairs(model,eps,C,block=256):
    """All columns and all orthogonality pairs, streamed with bounded workspace."""
    import scipy.sparse as sp
    n=model.n_orbitals
    if C.shape!=(n,n) or eps.shape!=(n,) or not np.isfinite(eps).all():
        raise ValueError("reference must contain all eigenstates")
    residual2=0.;orthogonal2=0.;cnorm2=0.
    for start in range(0,n,block):
        c=C[:,start:start+block]
        if not np.isfinite(c).all():raise ValueError("nonfinite reference eigenvectors")
        sc=model.S@c;hc=model.H_cov@c
        residual2+=float(np.linalg.norm(hc-sc*eps[None,start:start+block])**2)
        cnorm2+=float(np.linalg.norm(c)**2)
        gram=project(C,sc)
        gram[np.arange(start,min(start+block,n)),np.arange(c.shape[1])]-=1
        orthogonal2+=float(np.linalg.norm(gram)**2)
    denominator=(sp.linalg.norm(model.H_cov)+np.max(abs(eps))*sp.linalg.norm(model.S))*np.sqrt(cnorm2)
    gen=np.sqrt(residual2)/max(denominator,np.finfo(float).tiny)
    ortho=np.sqrt(orthogonal2/n)
    if not np.isfinite([gen,ortho]).all() or gen>1e-10 or ortho>1e-10:
        raise ValueError(f"full eigenpair verification failed: generalized={gen}, S orthogonality={ortho}")
    return dict(status="PASS",states=n,generalized_residual=float(gen),
                S_orthogonality_relative_frobenius=float(ortho),checked_columns=n,checked_pairs=n*n)


def jackson_weights(eps,cfg):
    lo,hi=cfg.spectral_bounds;c=(lo+hi)/2;a=(hi-lo)/2
    m=np.arange(cfg.n_moments);angle=np.pi/(cfg.n_moments+1)
    damping=((cfg.n_moments+1-m)*np.cos(m*angle)+np.sin(m*angle)/np.tan(angle))/(cfg.n_moments+1)
    x=(cfg.energy_eV-c)/a;y=(eps-c)/a
    return ((np.cos(np.arccos(x)[:,None]*m)*damping*np.where(m==0,1,2))@
            np.cos(m[:,None]*np.arccos(y)[None,:]))/(np.pi*a*np.sqrt(1-x*x))[:,None]


def eigen_observables(eps,q,velocities,cfg):
    weights=jackson_weights(eps,cfg);dos=(weights@abs(q)**2).T
    # Explicit quadrature grid, independent of product resolve_time_grid.
    end=cfg.time_fs[-1];dt=cfg.integration_dt_fs
    grid=np.arange(int(np.floor(end/dt))+1)*dt
    if grid[-1]<end and not np.isclose(grid[-1],end,rtol=0,atol=8*np.spacing(max(end,dt))):
        grid=np.append(grid,end)
    else:grid[-1]=end
    times=np.unique(np.r_[grid,cfg.time_fs]);selected=np.searchsorted(times,cfg.time_fs)
    shape=(q.shape[1],len(cfg.axes),len(cfg.time_fs),len(cfg.energy_eV))
    nums={k:np.zeros(shape) for k in ("msd_A2/vac_integral","diffusion_A2_per_fs/vac_integral","vac_A2_per_fs2/vac_integral")}
    if cfg.stochastic_method=="stepwise":nums["msd_A2/commutator"]=np.zeros(shape)
    for ai,axis in enumerate(cfg.axes):
        V=velocities(axis);Vq=V@q
        vac=np.empty((q.shape[1],len(times),len(cfg.energy_eV)))
        for first in range(0,len(times),16):
            t=times[first:first+16];phase=np.exp(-1j*eps[:,None]*t/HBAR_EV_FS)
            right=(Vq[:,None,:]*phase[:,:,None]).reshape(len(eps),-1)
            applied=(V@right).reshape(len(eps),len(t),q.shape[1])
            terms=(q.conj()[:,None,:]*phase.conj()[:,:,None]*applied).real
            vac[:,first:first+len(t)]=np.einsum("en,ntb->bte",weights,terms)
        M=np.zeros((q.shape[1],len(cfg.energy_eV)));D=M.copy()
        for k,t in enumerate(times):
            if k:
                h=t-times[k-1]
                M=M+2*h*D+h*h/3*(2*vac[:,k-1]+vac[:,k])
                D=D+h/2*(vac[:,k-1]+vac[:,k])
            if k in selected:
                j=int(np.flatnonzero(selected==k)[0])
                nums["msd_A2/vac_integral"][:,ai,j]=M
                nums["diffusion_A2_per_fs/vac_integral"][:,ai,j]=D
                nums["vac_A2_per_fs2/vac_integral"][:,ai,j]=vac[:,k]
        if cfg.stochastic_method=="stepwise":
            for j,t in enumerate(cfg.time_fs):
                amplitude=np.empty_like(q,dtype=complex)
                for start in range(0,len(eps),256):
                    ei=eps[start:start+256,None];ej=eps[None,:]
                    integral=t*np.exp(-.5j*(ei+ej)*t/HBAR_EV_FS)*np.sinc((ej-ei)*t/(2*np.pi*HBAR_EV_FS))
                    amplitude[start:start+256]=-(V[start:start+256]*integral)@q
                nums["msd_A2/commutator"][:,ai,j]=(weights@abs(amplitude)**2).T
    primary="commutator" if cfg.stochastic_method=="stepwise" else "vac_integral"
    if len(cfg.time_fs)>1:
        nums["diffusion_A2_per_fs/msd_gradient"]=.5*np.gradient(
            nums["msd_A2/"+primary],cfg.time_fs,axis=2,edge_order=2 if len(cfg.time_fs)>2 else 1)
    if not cfg.include_vac:nums.pop("vac_A2_per_fs2/vac_integral")
    return dos,nums


def expected_arrays(dos,nums,cfg):
    count=len(dos);blocks=[slice(i,min(i+cfg.statistics_block,count)) for i in range(0,count,cfg.statistics_block)]
    counts=np.array([len(dos[s]) for s in blocks]);rho=dos.mean(0)
    out=dict(dos_states_per_eV=rho,dos_block_sums=np.array([dos[s].sum(0) for s in blocks]),
             block_counts=counts,sample_count=np.array(count),sample_ids=np.arange(count))
    for key,value in nums.items():
        out["numerator_"+key]=value.mean(0)
        out["numerator_block_sums/"+key]=np.array([value[s].sum(0) for s in blocks])
        out[key]=value.mean(0)/rho
    return out


def differences(actual,expected,rtol=2e-6,atol=1e-8):
    result={}
    for key,want in expected.items():
        got=np.asarray(actual[key]);want=np.asarray(want)
        if got.shape!=want.shape:raise ValueError("reference shape mismatch: "+key)
        integer=want.dtype.kind in "biu"
        diff=np.abs(got.astype(float)-want.astype(float)) if integer else np.abs(got-want)
        finite=np.isfinite(diff)
        d=diff[finite];w=np.abs(want)[finite]
        result[key]=dict(passed=bool(np.array_equal(got,want) if integer else np.allclose(got,want,rtol=rtol,atol=atol,equal_nan=True)),
            max_absolute_difference=float(d.max(initial=0)),
            max_relative_difference=float((d/np.maximum(w,np.finfo(float).tiny)).max(initial=0)),
            relative_frobenius_difference=float(np.linalg.norm(d)/max(np.linalg.norm(w),np.finfo(float).tiny)))
    return result

