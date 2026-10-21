"""Small host channel definitions and Gaunt constants; no pair numerics."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import CubicSpline
from ..lcao_integrals import real_harmonics
from ...units import BOHR_A
ALGORITHM='fourier-bessel-real-gaunt-v1'


def orbital_payload(orbitals):
    ls=[];species=[];radii=[];coeff=[];offsets=[0];coff=[0]
    for si,orb in enumerate(orbitals.values()):
        for l,z,radial in orb.radial:
            if not 0<=l<=3: raise ValueError('supported input l is 0..3')
            if not np.isfinite(radial).all(): raise ValueError('nonfinite orbital')
            ls.append(l);species.append(si);radii.extend(orb.radius_bohr)
            coeff.extend(CubicSpline(orb.radius_bohr,radial).c.T.ravel())
            offsets.append(len(radii));coff.append(len(coeff))
    return dict(radial_l=np.array(ls,np.int64),radial_species=np.array(species,np.int64),
        radius=np.array(radii,float),coefficients=np.array(coeff,float),grid_offsets=np.array(offsets,np.int64),
        coeff_offsets=np.array(coff,np.int64),species_count=len(orbitals))


@lru_cache(maxsize=1)
def gaunt_constants():
    z,w=np.polynomial.legendre.leggauss(20);phi=2*np.pi*np.arange(40)/40
    xyz=np.stack(np.broadcast_arrays(np.sqrt(1-z[:,None]**2)*np.cos(phi),np.sqrt(1-z[:,None]**2)*np.sin(phi),z[:,None]+np.zeros(40)),axis=-1).reshape(-1,3)
    y=np.column_stack([real_harmonics(l,xyz) for l in range(8)])
    weights=np.repeat(w,40)*2*np.pi/40
    G=np.einsum('qi,qj,qk,q->ijk',y[:,:25],y[:,:25],y,weights,optimize=True)
    G[abs(G)<2e-14]=0
    for a in range(5):
        for b in range(5):
            for c in range(8):
                if (a+b+c)%2 or c<abs(a-b) or c>a+b:
                    G[a*a:(a+1)**2,b*b:(b+1)**2,c*c:(c+1)**2]=0
    G.setflags(write=False)
    return G


def channel_definition(payload):
    ls=payload['radial_l']; transforms=[]
    for a,l in enumerate(ls):
        transforms.extend([(a,int(l),0),(a,int(l)+1,1)])
        if l: transforms.append((a,int(l)-1,1))
    transforms=sorted(transforms); lookup={v:i for i,v in enumerate(transforms)}
    channels=[]; pairs=[]
    for a,lam,power in transforms:
        for b,lb in enumerate(ls):
            for L in range(abs(lam-int(lb)),lam+int(lb)+1,2):
                channels.append((a,b,lam,power,L));pairs.append((lookup[(a,lam,power)],lookup[(b,int(lb),0)]))
    return np.array(transforms,np.int64),np.array(channels,np.int64),np.array(pairs,np.int64)


def angular_terms(payload,channels):
    ns=payload['species_count'];ls=payload['radial_l'];species=payload['radial_species'];go=payload['grid_offsets']
    sizes=np.zeros(ns,np.int64);orb_offset=[];cut=np.zeros(ns)
    for a,l in enumerate(ls):
        si=species[a];orb_offset.append(sizes[si]);sizes[si]+=2*l+1;cut[si]=max(cut[si],payload['radius'][go[a+1]-1])
    group_offsets=np.r_[0,np.cumsum([4*i*j for i in sizes for j in sizes])].astype(np.int64)
    terms=[[] for _ in range(group_offsets[-1])];G=gaunt_constants()
    for ci,(a,b,lam,power,L) in enumerate(channels):
        la,lb=ls[a],ls[b];sa,sb=species[a],species[b];group=sa*ns+sb;size=sizes[sa]*sizes[sb]
        phase=(-1.)**((lam-lb-L)//2)
        for ma in range(2*la+1):
            for mb in range(2*lb+1):
                out=(orb_offset[a]+ma)*sizes[sb]+orb_offset[b]+mb
                for M in range(2*L+1):
                    h=L*L+M
                    if power==0:
                        v=8*phase*G[la*la+ma,lb*lb+mb,h]
                        if v: terms[group_offsets[group]+out].append((ci,h,v))
                    else:
                        for q,qm,sign in ((0,1,-1),(1,2,-1),(2,0,1)):
                            v=sum(G[la*la+ma,1+qm,lam*lam+mu]*G[lam*lam+mu,lb*lb+mb,h] for mu in range(2*lam+1))
                            v*=8*phase*sign*np.sqrt(4*np.pi/3)*BOHR_A
                            if abs(v)>1e-14: terms[group_offsets[group]+(q+1)*size+out].append((ci,h,v))
    offsets=np.r_[0,np.cumsum([len(t) for t in terms])].astype(np.int64)
    flattened=[v for t in terms for v in t]
    return dict(group_offsets=group_offsets,cutoffs=np.array([a+b for a in cut for b in cut]),
        term_offsets=offsets,term_channels=np.array([v[0] for v in flattened],np.int64),
        term_harmonics=np.array([v[1] for v in flattened],np.int64),term_coefficients=np.array([v[2] for v in flattened]),
        species_sizes=sizes)
