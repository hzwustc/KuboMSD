"""Deterministic disjoint whole-atom neighborhoods, with explicit model identity."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from ..errors import NumericalError

SCHEMA="kubomsd-metric-partition-v1"


def content_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),
        default=lambda a:np.asarray(a).tolist()).encode()).hexdigest()


def _slices(atom_slices,n):
    flat=[]
    for a,b in atom_slices:
        if isinstance(a,bool) or isinstance(b,bool) or int(a)!=a or int(b)!=b or not 0<=a<b<=n:
            raise NumericalError("invalid atom slices")
        flat.extend(range(a,b))
    if flat!=list(range(n)):raise NumericalError("atom slices must partition all orbitals")


def periodic_distances(displacements,cell):
    """Exact closest lattice image in 3D, bounded O(N) storage.

    A rounded fractional image provides an upper bound; the smallest singular
    value bounds all translations that could improve it, including skew cells.
    Images are streamed, never an atoms-by-atoms distance matrix.
    """
    f=displacements@np.linalg.inv(cell); f-=np.rint(f)
    best=np.sum((f@cell)**2,axis=1)
    radius=np.sqrt(best.max(initial=0))/np.linalg.svd(cell,compute_uv=False)[-1]
    limit=int(np.ceil(radius+.5))
    for shift in itertools.product(range(-limit,limit+1),repeat=3):
        d=(f-np.asarray(shift))@cell
        best=np.minimum(best,np.sum(d*d,axis=1))
    return np.sqrt(best)


def make_spatial_partition(structure,atom_slices,*,max_orbitals,s_fingerprint,
                           physical_key,orbital_order_hash):
    cell=np.asarray(structure["cell_A"],float);pos=np.asarray(structure["positions_A"],float)
    symbols=list(structure["symbols"]);nat=len(symbols)
    if (cell.shape!=(3,3) or pos.shape!=(nat,3) or len(atom_slices)!=nat or
        not np.isfinite(cell).all() or not np.isfinite(pos).all() or
        abs(np.linalg.det(cell))<1e-12):
        raise NumericalError("invalid partition structure/atom mapping")
    n=atom_slices[-1][1];_slices(atom_slices,n)
    if not isinstance(max_orbitals,int) or max_orbitals<1:
        raise NumericalError("invalid orbital cap")
    sizes=np.array([b-a for a,b in atom_slices])
    if max(sizes)>max_orbitals:raise NumericalError("atom exceeds orbital cap")
    frac=(pos@np.linalg.inv(cell))%1
    tree=cKDTree(frac,boxsize=1.)
    smallest=np.linalg.svd(cell,compute_uv=False)[-1]
    unassigned=set(range(nat));blocks=[]
    while unassigned:
        seed=min(unassigned)
        count=min(32,nat)
        while True:
            fractional_distances,neighbors=tree.query(frac[seed],k=count,workers=1)
            candidates=np.array([int(i) for i in np.atleast_1d(neighbors) if i in unassigned])
            distances=periodic_distances(pos[candidates]-pos[seed],cell)
            sort=np.lexsort((candidates,distances));order=candidates[sort]
            total=int(sizes[seed]);boundary=None
            for at,dist in zip(order,distances[sort]):
                if at==seed:continue
                if total+sizes[at]>max_orbitals:boundary=dist;break
                total+=int(sizes[at])
            # Every unqueried image has physical distance >= sigma_min times
            # its fractional torus distance. Expand until the ordering used by
            # this cluster is certified even in a skew cell.
            if count==nat or (boundary is not None and np.atleast_1d(fractional_distances)[-1]*smallest>boundary+1e-12):
                break
            count=min(nat,2*count)
        atoms=[seed];used=int(sizes[seed])
        for i in order:
            i=int(i)
            if i==seed:continue
            if used+sizes[i]>max_orbitals:break
            atoms.append(i);used+=int(sizes[i])
        unassigned.difference_update(atoms)
        blocks.append(dict(atoms=atoms,orbitals=[j for i in atoms for j in range(*atom_slices[i])]))
    return dict(schema=SCHEMA,algorithm="nearest-periodic-seed-v1",s_fingerprint=s_fingerprint,
                physical_key=physical_key,orbital_order_hash=orbital_order_hash,
                structure_hash=content_hash(structure),n_orbitals=n,n_atoms=nat,
                max_orbitals=max_orbitals,blocks=blocks)


def load_metric_partition(path,*,s_fingerprint,physical_key,orbital_order_hash,
                          atom_slices,n_orbitals):
    d=json.loads(Path(path).read_text())
    for key,want in dict(schema=SCHEMA,s_fingerprint=s_fingerprint,physical_key=physical_key,
                         orbital_order_hash=orbital_order_hash,n_orbitals=n_orbitals).items():
        if d.get(key)!=want:raise NumericalError("partition identity mismatch: "+key)
    _slices(atom_slices,n_orbitals)
    if not isinstance(d.get("structure_hash"),str) or len(d["structure_hash"])!=64:
        raise NumericalError("partition requires structure hash")
    atoms=[];orbitals=[]
    for block in d["blocks"]:
        ids=block["atoms"];idx=block["orbitals"]
        if (not ids or any(type(i)!=int or not 0<=i<len(atom_slices) for i in ids) or
            idx!=[j for i in ids for j in range(*atom_slices[i])] or len(idx)>d["max_orbitals"]):
            raise NumericalError("invalid whole-atom partition block")
        atoms.extend(ids);orbitals.extend(idx)
    if sorted(atoms)!=list(range(len(atom_slices))) or sorted(orbitals)!=list(range(n_orbitals)):
        raise NumericalError("partition must cover each atom/orbital exactly once")
    return d


def model_identity(model):
    cache=model.provenance.get("operator_cache") or {}
    return dict(physical_key=cache.get("physical_key"),
                orbital_order_hash=content_hash(model.orbital_order))
