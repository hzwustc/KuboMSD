"""Native-compatible cache I/O; chunked exports and atomic single-writer publish."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import h5py
import numpy as np
from .contracts import DeviceModelBundle
from ...validation import file_record


def _sha(path):return file_record(path)['sha256']


def save_operator_cache(model,path,context):
    if sum(bool(v) for v in (model.composite,model.velocity,model.position))!=1:
        raise ValueError('native cache requires exactly one velocity source')
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    lock=path.with_name(path.name+'.lock');lock.mkdir()
    temp=None
    try:
        temp=Path(tempfile.mkdtemp(prefix=path.name+'.partial-',dir=path.parent))
        if path.exists():raise FileExistsError(path)
        matrices={'H':model.H_cov,'S':model.S}
        for axis,v in model.velocity.items():matrices['velocity_'+axis]=v
        for axis,v in model.position.items():matrices['position_'+axis]=v
        for axis,parts in model.composite.items():
            for label,v in zip(('bond','xbar','delta'),parts):matrices[axis+'_'+label]=v
        declarations={}
        with context.phase('cache_io'),h5py.File(temp/'operators.h5','w') as f:
            for name,mat in matrices.items():
                context.check_device(mat);group=f.create_group(name);group.create_dataset('shape',data=mat.shape)
                for key in ('data','indices','indptr'):
                    arr=getattr(mat,key);ds=group.create_dataset(key,shape=arr.shape,dtype=arr.dtype)
                    for start in range(0,len(arr),65536):
                        host=context.to_host(arr[start:start+65536],name+'.'+key,'cache_export')
                        ds[start:start+len(host)]=host
                        if not np.array_equal(ds[start:start+len(host)],host):raise IOError('cache readback mismatch')
                declarations[name]={'path':'operators.h5','key':name,'shape':list(mat.shape),'dtype':str(mat.dtype),
                    'storage':'csr','index_base':0,'representation':'covariant'}
            f.flush()
        if model.composite:
            velocity_source={'type':'composite','axes':{a:{key:declarations[a+'_'+key] for key in ('bond','xbar','delta')} for a in model.composite}}
        elif model.velocity:velocity_source={'type':'matrix','unit':'angstrom/fs','matrices':{a:declarations['velocity_'+a] for a in model.velocity}}
        else:velocity_source={'type':'position','matrices':{a:declarations['position_'+a] for a in model.position}}
        data={'format':'kubomsd-model-v1','n_orbitals':model.n_orbitals,'basis':model.basis,'kpoint':[0,0,0],
            'energy_unit':'eV','length_unit':'angstrom','orbital_order':list(model.orbital_order) or [str(i) for i in range(model.n_orbitals)],
            'atom_slices':list(model.atom_slices),'hamiltonian':declarations['H'],'overlap':declarations['S'],
            'velocity_source':velocity_source,'gpu_cache':{'schema':1,'identity':model.cache_identity,'provenance':dict(model.provenance)}}
        (temp/'model.json').write_text(json.dumps(data,indent=2)+'\n')
        ready={'schema':1,'cache_identity':model.cache_identity,'model_sha256':_sha(temp/'model.json'),
               'operators_sha256':_sha(temp/'operators.h5'),'status':'CACHE_READY'}
        (temp/'CACHE_READY').write_text(json.dumps(ready,indent=2)+'\n')
        for name in ('model.json','operators.h5','CACHE_READY'):
            with (temp/name).open('rb') as f:os.fsync(f.fileno())
        os.rename(temp,path)
        return ready
    finally:
        if temp is not None and temp.exists():shutil.rmtree(temp)
        lock.rmdir()


def load_operator_cache_gpu(path,context):
    from ...io.model import load_model
    path=Path(path).resolve()
    if path.is_dir():path=path/'model.json'
    d=json.loads(path.read_text())
    if d.get('format')!='kubomsd-model-v1':raise ValueError('GPU cache loader requires native model schema')
    cache=d.get('gpu_cache')
    if cache:
        provenance=cache.get('provenance',{})
        if (provenance.get('input_format')=='legacy-lcao-deeptb'
                and not provenance.get('basis_alignment','').startswith('operator-basis-v2;')):
            raise ValueError('raw GPU cache has unverified orbital basis; rebuild from the original input')
        ready=json.loads((path.parent/'CACHE_READY').read_text())
        if ready.get('status')!='CACHE_READY' or ready['model_sha256']!=_sha(path) or ready['operators_sha256']!=_sha(path.parent/'operators.h5') or ready['cache_identity']!=cache['identity']:
            raise ValueError('cache checksum/ready identity mismatch')
    with context.phase('cache_io'):
        # Host sparse objects exist only as parsed file input, never as a builder
        # or solver prerequisite. Upload each mathematical matrix exactly once.
        host=load_model(path)
        H=context.matrix(host.H_cov,'H','cache_import');S=context.matrix(host.S,'S','cache_import')
        v={a:context.matrix(m,'velocity_'+a) for a,m in host.velocity.items()}
        p={a:context.matrix(m,'position_'+a) for a,m in host.position.items()}
        c={a:tuple(context.matrix(m,a+'_'+label) for m,label in zip(parts,('bond','xbar','delta'))) for a,parts in host.composite.items()}
    provenance={**(cache.get('provenance',{}) if cache else host.provenance),'builder_used':'cache'}
    identity=cache['identity'] if cache else hashlib.sha256(''.join(r['sha256'] for r in host.provenance['files']).encode()).hexdigest()
    return DeviceModelBundle(H,S,context.device_id,v,p,c,tuple(host.orbital_order),tuple(map(tuple,host.atom_slices)),provenance,host.basis,identity)
