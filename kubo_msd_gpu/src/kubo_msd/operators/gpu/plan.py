"""CPU geometry/index planning only; full orbital-cutoff support, independent of H."""
from itertools import product
import numpy as np
from .contracts import PairPlan


def make_pair_plan(parsed):
    structure=parsed.structure;pos=np.asarray(structure['positions_A']);cell=np.asarray(structure['cell_A'])
    if cell.shape!=(3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell))<1e-12:raise ValueError('invalid cell')
    inv=np.linalg.inv(cell);symbols=structure['symbols'];names=list(parsed.orbitals)
    keys=[];dis=[];species=[];shapes=[]
    for i,si in enumerate(symbols):
        for j in range(i,len(symbols)):
            sj=symbols[j];oi=parsed.orbitals[si];oj=parsed.orbitals[sj];cut=oi.cutoff_A+oj.cutoff_A
            df=(pos[j]-pos[i])@inv;lim=cut*np.linalg.norm(inv,axis=0)
            ranges=[range(int(np.ceil(-df[k]-lim[k])),int(np.floor(-df[k]+lim[k]))+1) for k in range(3)]
            for R in product(*ranges):
                if i==j and R<(0,0,0):continue
                d=pos[j]-pos[i]+np.array(R)@cell
                if np.linalg.norm(d)>=cut:continue
                keys.append((i,j,*R));dis.append(d);species.append((names.index(si),names.index(sj)));shapes.append((oi.n_orbitals,oj.n_orbitals))
    shapes=np.array(shapes,np.int64).reshape(-1,2);offsets=np.r_[0,np.cumsum(shapes.prod(1))].astype(np.int64)
    arrays=[np.array(keys,np.int64).reshape(-1,5),np.array(species,np.int64).reshape(-1,2),np.array(dis,float).reshape(-1,3),np.arange(len(keys),dtype=np.int64),shapes,offsets]
    if offsets[-1]>=2**31:raise ValueError('pair COO exceeds current int32 sparse library capacity')
    for a in arrays:a.setflags(write=False)
    return PairPlan(*arrays,{'canonical_pairs':len(keys),'block_values':int(offsets[-1]),'support':'orbital cutoff, independent of H graph','stable_order':'i,j,lexicographic R'})
