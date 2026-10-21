"""Independent numerical two-centre quadrature for ABACUS numerical orbitals.

Reference implementation: Gauss-Legendre radial/cos(theta), periodic phi.
ABACUS real harmonics are m=0, sqrt(2)Re Y_l1, sqrt(2)Im Y_l1, ...,
including the Condon-Shortley phase. Orbitals order is l, zeta, m.
No renormalization, fitted overlap, pseudopotentials or imported RSATB code.
"""
from dataclasses import dataclass
from itertools import product
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.special import lpmv,gammaln
from ..units import BOHR_A
from ..errors import KuboMSDError,NumericalError
from ..contracts import IntegralBundle
from ..validation import file_record

@dataclass
class OrbitalSet:
    element: str
    radius_bohr: np.ndarray
    radial: list
    norms: list
    provenance: dict
    @property
    def n_orbitals(self): return sum(2*l+1 for l,z,r in self.radial)
    @property
    def cutoff_A(self): return float(self.radius_bohr[-1]*BOHR_A)
    @property
    def labels(self): return [f'l{l}:z{z}:m{m}' for l,z,r in self.radial for m in range(2*l+1)]

def read_orb(path,norm_tolerance=1e-4):
    text=Path(path).read_text(); element=re.search(r'Element\s+(\S+)',text); mesh=re.search(r'\bMesh\s+(\d+)',text); dr=re.search(r'\bdr\s+([\d.eE+-]+)',text)
    if not (element and mesh and dr): raise KuboMSDError(f'invalid ABACUS .orb header: {path}')
    n=int(mesh[1]); step=float(dr[1]); r=np.arange(n)*step
    if n<4 or step<=0: raise KuboMSDError('invalid radial grid')
    radial=[]; norms=[]
    blocks=re.split(r'Type\s+L\s+N',text)[1:]
    for block in blocks:
        tokens=block.split(); typ,l,z=map(int,tokens[:3]); values=np.asarray([float(x.replace('D','E')) for x in tokens[3:]],float)
        if len(values)!=n or not np.isfinite(values).all() or l<0: raise KuboMSDError('invalid orbital radial block')
        norm=float(simpson(values**2*r*r,x=r))
        if abs(norm-1)>norm_tolerance: raise NumericalError(f'{path}: radial l={l},zeta={z} norm={norm:g}; no automatic renormalization')
        radial.append((l,z,values)); norms.append(norm)
    if not radial or len(set((l,z) for l,z,v in radial))!=len(radial): raise KuboMSDError('missing/duplicate radial orbitals')
    pairs=sorted(zip(radial,norms),key=lambda x:(x[0][0],x[0][1])); radial,norms=map(list,zip(*pairs))
    return OrbitalSet(element[1],r,radial,norms,file_record(path))

def real_harmonics(l,vectors):
    r=np.linalg.norm(vectors,axis=1); cost=np.divide(vectors[:,2],r,out=np.ones_like(r),where=r>0)
    cost=np.clip(cost,-1,1); phi=np.arctan2(vectors[:,1],vectors[:,0])
    result=[np.sqrt((2*l+1)/(4*np.pi))*lpmv(0,l,cost)]
    for m in range(1,l+1):
        amplitude=np.sqrt((2*l+1)/(2*np.pi)*np.exp(gammaln(l-m+1)-gammaln(l+m+1)))*lpmv(m,l,cost)
        result.extend([amplitude*np.cos(m*phi),amplitude*np.sin(m*phi)])
    return np.column_stack(result)

def orbital_values(orbitals,points_bohr):
    r=np.linalg.norm(points_bohr,axis=1); values=[]; angular={}
    for l,z,R in orbitals.radial:
        if l not in angular: angular[l]=real_harmonics(l,points_bohr)
        f=CubicSpline(orbitals.radius_bohr,R,extrapolate=False)(r)
        f=np.where(r<=orbitals.radius_bohr[-1],f,0)
        values.append(f[:,None]*angular[l])
    return np.column_stack(values)

def two_center(bra,ket,displacement_A,settings):
    nr=int(settings.get('radial_points',64)); nt=int(settings.get('polar_points',24)); np_=int(settings.get('azimuthal_points',48))
    if nr<8 or nt<4 or np_<8: raise KuboMSDError('quadrature requires radial>=8, polar>=4, azimuth>=8')
    x,wr=np.polynomial.legendre.leggauss(nr); r=(x+1)*bra.radius_bohr[-1]/2; wr=wr*bra.radius_bohr[-1]/2*r*r
    ct,wt=np.polynomial.legendre.leggauss(nt); phi=np.arange(np_)*2*np.pi/np_
    unit=np.stack(np.broadcast_arrays(np.sqrt(1-ct[:,None]**2)*np.cos(phi),np.sqrt(1-ct[:,None]**2)*np.sin(phi),ct[:,None]+np.zeros(np_)),axis=-1).reshape(-1,3)
    points=(r[:,None,None]*unit[None,:,:]).reshape(-1,3)
    w=(wr[:,None,None]*wt[None,:,None]*np.ones((1,1,np_))*2*np.pi/np_).ravel()
    A=orbital_values(bra,points); B=orbital_values(ket,points-np.asarray(displacement_A)/BOHR_A)
    S=A.T@(w[:,None]*B)
    X=np.array([A.T@((w*points[:,k]*BOHR_A)[:,None]*B) for k in range(3)])
    return S,X,{'points':len(w),'radial_points':nr,'polar_points':nt,'azimuthal_points':np_,'dipole_origin':'bra center','length_unit':'angstrom'}

def build_lcao_integrals(structure,orbitals,settings):
    from .basis import convention_for,basis_map,transform_block,orbital_labels
    convention=convention_for(structure)
    maps={s:basis_map(o,convention) for s,o in orbitals.items()}
    positions=np.asarray(structure['positions_A']); cell=np.asarray(structure['cell_A']); symbols=structure['symbols']; n=len(symbols)
    settings=dict(settings or {}); tol=float(settings.get('tolerance',1e-5)); levels=int(settings.get('max_refinements',2))
    maxcut=2*max(o.cutoff_A for o in orbitals.values()); inv=np.linalg.inv(cell)
    # Fractional displacement includes unwrapped coordinates; center each enumeration at -df.
    overlaps={}; dipoles={}; maxerror=0.; used=[]; cache={}
    for i in range(n):
        for j in range(i,n):
            oi,oj=orbitals[symbols[i]],orbitals[symbols[j]]; cutoff=oi.cutoff_A+oj.cutoff_A
            df=(positions[j]-positions[i])@inv; limits=cutoff*np.linalg.norm(inv,axis=0)
            ranges=[range(int(np.ceil(-df[k]-limits[k])),int(np.floor(-df[k]+limits[k]))+1) for k in range(3)]
            for R in product(*ranges):
                if i==j and tuple(R)<(0,0,0): continue
                displacement=np.asarray(R)@cell+positions[j]-positions[i]
                if np.linalg.norm(displacement)>=cutoff: continue
                key=(i,j,*R); cachekey=(symbols[i],symbols[j],tuple(np.round(displacement,12)))
                if cachekey not in cache:
                    s0,x0,diag=two_center(oi,oj,displacement,settings); error=float('inf')
                    for level in range(1,levels+1):
                        fine={k:int(settings.get(k,v))*2**level for k,v in [('radial_points',64),('polar_points',24),('azimuthal_points',48)]}
                        s1,x1,diag=two_center(oi,oj,displacement,fine)
                        error=max(float(np.max(abs(s1-s0))),float(np.max(abs(x1-x0)))/max(1,cutoff))
                        s0,x0=s1,x1
                        if error<=tol: break
                    if error>tol: raise NumericalError(f'orbital quadrature not converged for {key}: scaled error {error:g} > {tol:g}')
                    cache[cachekey]=(s0,x0,error,diag)
                S,X,error,diag=cache[cachekey]; maxerror=max(maxerror,error); used.append(diag)
                if convention!='abacus-real':
                    S=transform_block(S,maps[symbols[i]],maps[symbols[j]])
                    X=transform_block(X,maps[symbols[i]],maps[symbols[j]])
                Xabs=X+positions[i,:,None,None]*S
                overlaps[key]=S; dipoles[key]=Xabs
                partner=(j,i,*(-np.array(R)))
                if partner!=key:
                    overlaps[partner]=S.conj().T
                    dipoles[partner]=np.transpose(Xabs.conj()-((np.asarray(R)@cell)[:,None,None]*S),(0,2,1))
    order=[f'atom{i}:{label}' for i,s in enumerate(symbols) for label in orbital_labels(orbitals[s],convention)]
    return IntegralBundle(overlaps,dipoles,order,{'method':'spherical two-center quadrature','max_scaled_refinement_difference':maxerror,'tolerance':tol,'blocks':len(overlaps),'unique_integrals':len(cache),'internal_harmonics':'ABACUS real, Condon-Shortley, 0,+1 cosine,+1 sine,...','orbital_convention':convention,'radial_unit':'bohr','norms':{s:o.norms for s,o in orbitals.items()},'last_quadrature':used[-1] if used else {}})
