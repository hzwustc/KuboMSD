import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import h5py
from ..contracts import ModelBundle
from ..validation import hermitian,file_record
from ..errors import KuboMSDError
from ..units import HARTREE_EV,RYDBERG_EV,BOHR_A,HBAR_EV_FS

def read_matrix(entry,root,n,records):
    required={'path','shape','dtype','storage','index_base','representation'}
    if not required<=entry.keys(): raise KuboMSDError(f'matrix declaration missing {sorted(required-entry.keys())}')
    if entry['shape']!=[n,n] or entry['index_base'] not in (0,1) or entry['representation']!='covariant' or entry['dtype'] not in ('float64','complex128'): raise KuboMSDError('invalid matrix shape/index/representation/dtype')
    path=(root/entry['path']).resolve()
    if not path.is_file(): raise KuboMSDError(f'missing matrix: {path}')
    records.append(file_record(path)); storage=entry['storage']
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as f:
            if storage=='dense': A=np.array(f[entry.get('key','matrix')])
            elif storage=='csr':
                shape=tuple(f['shape']); idx=f['indices']; ptr=f['indptr']; base=entry['index_base']
                if shape!=(n,n) or idx.dtype.kind not in 'iu' or ptr.dtype.kind not in 'iu' or len(ptr)!=n+1 or ptr[0]!=base or ptr[-1]-base!=len(idx) or len(f['data'])!=len(idx) or np.any(np.diff(ptr)<0) or np.any(idx<base) or np.any(idx>=n+base): raise KuboMSDError('malformed CSR arrays')
                A=sp.csr_matrix((f['data'],idx-base,ptr-base),shape=shape)
            else: raise KuboMSDError('matrix storage must be dense/csr')
    elif path.suffix in ('.h5','.hdf5'):
        with h5py.File(path,'r') as f:
            if storage=='dense': A=np.array(f[entry.get('key','matrix')])
            elif storage=='csr':
                g=f[entry.get('key','matrix')]; base=entry['index_base']; A=sp.csr_matrix((g['data'][:],g['indices'][:]-base,g['indptr'][:]-base),shape=tuple(g['shape'][:]))
                A.check_format(full_check=True)
            else: raise KuboMSDError('matrix storage must be dense/csr')
    else: raise KuboMSDError('matrix container must be NPZ/HDF5')
    if A.shape!=(n,n) or str(A.dtype)!=entry['dtype']: raise KuboMSDError('matrix shape/dtype disagrees with declaration')
    return sp.csr_matrix(A,dtype=complex)

def load_model(path,backend='cpu'):
    path=Path(path).resolve()
    if not path.is_file(): raise KuboMSDError(f'missing model: {path}')
    d=json.loads(path.read_text()); root=path.parent
    if d.get('format')!='kubomsd-model-v1':
        if d.get('version')==1 and 'model' in d:
            from .legacy_input import load_legacy
            return load_legacy(path,d)
        raise KuboMSDError('unknown input schema')
    for key in ('n_orbitals','basis','kpoint','energy_unit','length_unit','orbital_order','hamiltonian','velocity_source'):
        if key not in d: raise KuboMSDError('missing model field '+key)
    n=d['n_orbitals']; order=d['orbital_order']
    if not isinstance(n,int) or n<1 or len(order)!=n or len(set(map(str,order)))!=n: raise KuboMSDError('invalid orbital count/order')
    if d['kpoint']!=[0,0,0]: raise KuboMSDError('v1 supports only one Gamma point')
    if d['basis'] not in ('orthogonal','nonorthogonal'): raise KuboMSDError('basis must be explicit')
    ef={'eV':1.,'hartree':HARTREE_EV,'rydberg':RYDBERG_EV}.get(d['energy_unit']); lf={'angstrom':1.,'bohr':BOHR_A}.get(d['length_unit'])
    if ef is None or lf is None: raise KuboMSDError('unsupported energy/length units')
    records=[file_record(path)]; H=hermitian(read_matrix(d['hamiltonian'],root,n,records)*ef,'Hamiltonian')
    if 'overlap' in d: S=hermitian(read_matrix(d['overlap'],root,n,records),'overlap')
    elif d['basis']=='orthogonal': S=sp.eye(n,dtype=complex,format='csr')
    else: raise KuboMSDError('nonorthogonal model requires overlap')
    if d['basis']=='orthogonal' and sp.linalg.norm(S-sp.eye(n))>1e-12: raise KuboMSDError('orthogonal model supplied nonidentity S')
    source=d['velocity_source']; kind=source.get('type'); velocity={}; position={}; composite={}
    if kind=='matrix':
        factor={'angstrom/fs':1.,'eV*angstrom':1/HBAR_EV_FS}.get(source.get('unit'))
        if factor is None: raise KuboMSDError('explicit physical velocity unit required')
        velocity={a:hermitian(read_matrix(e,root,n,records)*factor,'velocity '+a) for a,e in source['matrices'].items()}
    elif kind=='position':
        if 'coordinates' in source:
            if d['basis']!='orthogonal': raise KuboMSDError('atomic coordinates cannot replace nonorthogonal dipole matrices')
            r=np.asarray(source['coordinates'],float)*lf
            if r.shape!=(n,3) or not np.isfinite(r).all(): raise KuboMSDError('invalid orbital positions')
            position={a:sp.diags(r[:,k],format='csr') for k,a in enumerate('xyz')}
        else: position={a:hermitian(read_matrix(e,root,n,records)*lf,'position '+a) for a,e in source['matrices'].items()}
    elif kind=='composite':
        for a,parts in source['axes'].items(): composite[a]=(read_matrix(parts['bond'],root,n,records)*ef*lf,read_matrix(parts['xbar'],root,n,records)*lf,read_matrix(parts['delta'],root,n,records)*lf)
    else: raise KuboMSDError('unknown velocity source')
    return ModelBundle(H,S,velocity,position,composite,order,d.get('atom_slices',[]),{'files':records,'input_format':d['format'],'internal_units':['eV','fs','angstrom'],'velocity_source':kind},d['basis'])
