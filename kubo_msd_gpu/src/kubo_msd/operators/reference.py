"""Independent direct real-space integration with bounded point batches."""
import numpy as np
from scipy.interpolate import CubicSpline
from .lcao_integrals import real_harmonics
from ..units import BOHR_A
from ..errors import KuboMSDError


def direct_pair(bra, ket, displacement_A, settings):
    nr=int(settings.get('radial_points',64)); nt=int(settings.get('polar_points',24))
    nphi=int(settings.get('azimuthal_points',48))
    if nr<8 or nt<4 or nphi<8:
        raise KuboMSDError('invalid reference quadrature')
    d=np.asarray(displacement_A,dtype=float)/BOHR_A
    if d.shape!=(3,) or not np.isfinite(d).all(): raise KuboMSDError('invalid displacement')
    distance=np.linalg.norm(d)
    splines=[[CubicSpline(o.radius_bohr,v,extrapolate=False) for l,z,v in o.radial] for o in (bra,ket)]
    def values(o, spl, points):
        radius=np.linalg.norm(points,axis=1)
        angular={l:real_harmonics(l,points) for l,z,v in o.radial}
        return np.column_stack([np.nan_to_num(f(radius),nan=0.)[:,None]*angular[l] for (l,z,v),f in zip(o.radial,spl)])
    x,wx=np.polynomial.legendre.leggauss(nr)
    ct,wt=np.polynomial.legendre.leggauss(nt)
    phi=2*np.pi*np.arange(nphi)/nphi
    # Rotate quadrature axes, not the supplied geometry or orbital basis. Clip the
    # angular domain to the intersection of the two finite support spheres.
    ez=d/distance if distance>1e-14 else np.array([0.,0.,1.])
    trial=np.eye(3)[np.argmin(abs(ez))]; ex=np.cross(trial,ez); ex/=np.linalg.norm(ex); ey=np.cross(ez,ex)
    ca,cb=bra.radius_bohr[-1],ket.radius_bohr[-1]
    cuts=sorted(set([0.,ca]+[float(v) for v in (abs(distance-cb),distance+cb,distance) if 0<v<ca]))
    s=np.zeros((bra.n_orbitals,ket.n_orbitals)); D=np.zeros((3,*s.shape)); points_used=0
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        for r,wr in zip((x+1)*(hi-lo)/2+lo,wx*(hi-lo)/2):
            low=max(-1.,(r*r+distance*distance-cb*cb)/(2*r*distance)) if distance>1e-14 else -1.
            if low>=1: continue
            c=(ct+1)*(1-low)/2+low; wtheta=wt*(1-low)/2
            unit=(np.sqrt(1-c[:,None]**2)[...,None]*(np.cos(phi)[None,:,None]*ex+np.sin(phi)[None,:,None]*ey)+c[:,None,None]*ez)
            pts=(r*unit).reshape(-1,3)
            w=np.repeat(wtheta,nphi)*wr*r*r*2*np.pi/nphi
            a=values(bra,splines[0],pts); b=values(ket,splines[1],pts-d)
            s+=a.T@(w[:,None]*b)
            for q in range(3): D[q]+=a.T@((w*pts[:,q]*BOHR_A)[:,None]*b)
            points_used+=len(w)
    return s,D,dict(radial_points=nr,polar_points=nt,azimuthal_points=nphi,points=points_used,
                    method='direct real-space support-intersection quadrature',dipole_origin='bra',length_unit='angstrom')


def evaluate_reference_pair(bra,ket,displacement_A,settings=None):
    settings=dict(settings or {}); epsilon=float(settings.get('tolerance',1e-5))
    levels=int(settings.get('max_refinements',3)); budget=epsilon/5
    s,D,diag=direct_pair(bra,ket,displacement_A,settings); history=[diag]
    error=None
    for level in range(1,levels+1):
        fine={k:int(settings.get(k,v))*2**level for k,v in [('radial_points',64),('polar_points',24),('azimuthal_points',48)]}
        sn,dn,diag=direct_pair(bra,ket,displacement_A,fine)
        error=max(float(np.max(abs(sn-s))),float(np.max(abs(dn-D)))/max(1.,bra.cutoff_A+ket.cutoff_A))
        diag['scaled_refinement_difference']=error; history.append(diag)
        s,D=sn,dn
        if error<=budget: break
    return s,D,dict(status='PASS' if error is not None and error<=budget else 'REFERENCE_UNCONVERGED',
                    error=error,budget=budget,history=history)
