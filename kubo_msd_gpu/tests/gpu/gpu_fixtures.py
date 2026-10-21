from pathlib import Path
import numpy as np
from scipy.special import gamma
from kubo_msd.operators.lcao_integrals import OrbitalSet
from kubo_msd.operators.gpu.contracts import ParsedBuildInput,PairPlan

def gaussian(alpha=.5,lmax=3,element='X',extra_zeta=False,radius=10):
    r=np.linspace(0,radius,int(radius*50)+1); radial=[]
    for l in range(lmax+1):
        for z,al in enumerate([alpha,alpha*1.7] if extra_zeta and l==0 else [alpha]):
            c=np.sqrt(2*(2*al)**(l+1.5)/gamma(l+1.5))
            radial.append((l,z,c*r**l*np.exp(-al*r*r)))
    return OrbitalSet(element,r,radial,[1.]*len(radial),{})

def mixed_orbitals():
    return {'X':gaussian(extra_zeta=True),
            'Y':gaussian(alpha=.7,lmax=2,element='Y',extra_zeta=True,radius=8)}

def parsed(orbitals=None):
    orbitals=orbitals or {'X':gaussian()}
    s=next(iter(orbitals));n=orbitals[s].n_orbitals
    return ParsedBuildInput({'cell_A':np.eye(3)*20,'positions_A':np.zeros((1,3)),'symbols':[s]},orbitals,{},tuple(f'o{i}' for i in range(n)),((0,n),),(),'synthetic')

def batch(displacements,orbs,species=None):
    d=np.asarray(displacements,float); ids=np.arange(len(d),dtype=np.int64)
    sp=np.zeros((len(d),2),np.int64) if species is None else np.asarray(species,np.int64)
    counts=[o.n_orbitals for o in orbs.values()]
    shapes=np.array([[counts[a],counts[b]] for a,b in sp],np.int64)
    offs=np.r_[0,np.cumsum(shapes.prod(1))].astype(np.int64)
    return PairPlan(np.zeros((len(d),5),np.int64),sp,d,ids,shapes,offs,{})
