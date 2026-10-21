"""Parse physical inputs without integration or matrix assembly."""
from pathlib import Path
import json
import re
import h5py
import numpy as np
from .contracts import ParsedBuildInput
from .basis import CONVENTIONS, orbital_labels
from ..lcao_integrals import read_orb
from ...io.legacy_input import read_poscar
from ...validation import file_record
from ...units import HARTREE_EV,RYDBERG_EV


def parse_build_input(path):
    path=Path(path).resolve();data=json.loads(path.read_text());root=path.parent
    model=data.get('model',{});ham=model.get('hamiltonian',{})
    if data.get('version')!=1 or model.get('type')!='lcao' or model.get('spin','none')!='none' or ham.get('format')!='deeptb':
        raise ValueError('raw builder requires scalar LCAO/DeePTB v1 input')
    if data.get('run',{}).get('kpoints',[[0,0,0]])!=[[0,0,0]]:raise ValueError('one Gamma point required')
    convention=ham.get('orbital_convention','deeptb-real')
    if convention not in CONVENTIONS:raise ValueError('unsupported Hamiltonian orbital_convention')
    pos=(root/data['structure']['file']).resolve();structure=read_poscar(pos)
    spec=model['orbitals'];orbitals={}
    if spec.get('convention','abacus-real')!='abacus-real':raise ValueError('unsupported harmonic convention')
    for name in spec['names']:
        orb=read_orb(root/spec.get('dir','.')/name)
        if orb.element in orbitals:raise ValueError('duplicate orbital element')
        if any(l>3 for l,z,r in orb.radial):raise ValueError('input l must be 0..3')
        orbitals[orb.element]=orb
    if any(s not in orbitals for s in structure['symbols']):raise ValueError('missing numerical orbitals')
    counts=[orbitals[s].n_orbitals for s in structure['symbols']];hp=(root/ham['path']).resolve()
    factor={'eV':1.,'hartree':HARTREE_EV,'rydberg':RYDBERG_EV}.get(ham.get('energy_unit','eV'))
    base=ham.get('index_base',0)
    if factor is None or base not in (0,1):raise ValueError('invalid H units or index base')
    images={}
    with h5py.File(hp,'r') as f:
        if '0' not in f or not isinstance(f['0'],h5py.Group):raise ValueError('DeePTB requires group 0')
        for key,ds in f['0'].items():
            if not re.fullmatch(r'-?\d+_-?\d+_-?\d+_-?\d+_-?\d+',key):raise ValueError('invalid H block key')
            i,j,*R=map(int,key.split('_'));i-=base;j-=base;v=np.asarray(ds)
            if v.ndim==3 and v.shape[-1]==2:v=v[...,0]+1j*v[...,1]
            if not 0<=i<len(counts) or not 0<=j<len(counts) or v.shape!=(counts[i],counts[j]) or not np.isfinite(v).all():raise ValueError('invalid Hamiltonian block')
            # Unit conversion is an explicit input boundary; no numerical merge.
            images[(i,j,*R)]=np.ascontiguousarray(v*factor,dtype=np.complex128)
    order=tuple(f'atom{i}:{lab}' for i,s in enumerate(structure['symbols'])
                for lab in orbital_labels(orbitals[s],convention))
    off=np.r_[0,np.cumsum(counts)]
    records=tuple([file_record(path),file_record(pos),file_record(hp)]+[o.provenance for o in orbitals.values()])
    return ParsedBuildInput(structure,orbitals,images,order,tuple(zip(off[:-1].tolist(),off[1:].tolist())),records,str(path),convention)
