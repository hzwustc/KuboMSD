"""Explicit model-only adapter; legacy task/run parameters are not imported."""
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from ..contracts import ModelBundle
from ..errors import KuboMSDError
from ..validation import file_record,hermitian
from .deeptb import read_deeptb
from ..operators.assembly import assemble_images
from ..operators.lcao_integrals import read_orb,build_lcao_integrals

def read_poscar(path):
    lines=[v.strip() for v in Path(path).read_text().splitlines() if v.strip()]
    scale=float(lines[1]); cell=np.array([list(map(float,v.split())) for v in lines[2:5]])
    if scale==0: raise KuboMSDError('POSCAR zero scale')
    factor=scale if scale>0 else (abs(scale)/abs(np.linalg.det(cell)))**(1/3); cell*=factor
    symbols=lines[5].split()
    if any(s.isdigit() for s in symbols): raise KuboMSDError('POSCAR requires explicit element symbols')
    counts=list(map(int,lines[6].split())); k=7
    if lines[k].lower().startswith('s'): k+=1
    direct=lines[k].lower().startswith('d'); cart=lines[k].lower().startswith(('c','k'))
    if not (direct or cart) or len(symbols)!=len(counts): raise KuboMSDError('invalid POSCAR coordinate mode/counts')
    p=np.array([list(map(float,v.split()[:3])) for v in lines[k+1:k+1+sum(counts)]])
    if p.shape!=(sum(counts),3) or not np.isfinite(p).all() or abs(np.linalg.det(cell))<1e-12: raise KuboMSDError('invalid POSCAR positions/cell')
    return {'cell_A':cell,'positions_A':p@cell if direct else p*factor,'symbols':[s for s,c in zip(symbols,counts) for _ in range(c)]}

def load_legacy(path,data):
    root=Path(path).parent; model=data['model']; ham=model.get('hamiltonian',{})
    if model.get('type')!='lcao' or model.get('spin','none')!='none' or ham.get('format')!='deeptb': raise KuboMSDError('legacy adapter supports scalar LCAO/DeePTB only')
    kpoints=data.get('run',{}).get('kpoints',[[0,0,0]])
    if kpoints!=[[0,0,0]]: raise KuboMSDError('legacy adapter requires one Gamma point')
    hp=(root/ham['path']).resolve()
    if not hp.is_file(): raise KuboMSDError(f'missing Hamiltonian file: {hp}; no substitute structure is selected')
    pos=(root/data['structure']['file']).resolve(); structure=read_poscar(pos); orbital_spec=model['orbitals']; orbitals={}
    for name in orbital_spec['names']:
        o=read_orb(root/orbital_spec.get('dir','.')/name)
        if o.element in orbitals: raise KuboMSDError('duplicate element orbital sets')
        orbitals[o.element]=o
    if any(s not in orbitals for s in structure['symbols']): raise KuboMSDError('missing element numerical orbital set')
    convention=orbital_spec.get('convention','abacus-real')
    if convention!='abacus-real': raise KuboMSDError('unsupported orbital harmonic convention')
    counts=[orbitals[s].n_orbitals for s in structure['symbols']]
    images=read_deeptb(hp,counts,index_base=ham.get('index_base',0))
    factor={'eV':1.,'hartree':27.211386245988,'rydberg':13.605693122994}.get(ham.get('energy_unit','eV'))
    if factor is None: raise KuboMSDError('unknown DeePTB Hamiltonian unit')
    images={k:v*factor for k,v in images.items()}
    H,bond,_=assemble_images(images,counts,structure['positions_A'],structure['cell_A'])
    integrals=build_lcao_integrals(structure,orbitals,model.get('integrals',{}))
    S,_,ket=assemble_images(integrals.overlap_images,counts,structure['positions_A'],structure['cell_A'])
    rowpos=np.repeat(structure['positions_A'],counts,axis=0); composite={}
    for k,a in enumerate('xyz'):
        Xi={key:v[k] for key,v in integrals.dipole_images.items()}
        X=assemble_images(Xi,counts,structure['positions_A'],structure['cell_A'])[0]
        composite[a]=(bond[k],X-sp.diags(rowpos[:,k])@S,X-ket[k])
    offsets=np.r_[0,np.cumsum(counts)]
    provenance={'files':[file_record(path),file_record(hp),file_record(pos)]+[o.provenance for o in orbitals.values()],'input_format':'legacy-lcao-deeptb','integrals':integrals.diagnostics,'mapped_model_fields':['structure.file','model.hamiltonian','model.orbitals','model.integrals','model.spin'],'ignored_calculation_sections':['run (except Gamma validation)','tasks','output'],'legacy_defaults':{'hamiltonian_energy_unit':ham.get('energy_unit','eV'),'index_base':ham.get('index_base',0),'orbital_convention':convention},'pseudopotentials':'not used by overlap/dipole commutator velocity'}
    return ModelBundle(hermitian(H,'H'),hermitian(S,'S'),composite=composite,orbital_order=integrals.orbital_order,atom_slices=list(zip(offsets[:-1].tolist(),offsets[1:].tolist())),provenance=provenance)
