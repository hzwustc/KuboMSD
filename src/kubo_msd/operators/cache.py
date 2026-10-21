"""Content-addressed radial tables and atomic, verified native model publication."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
import zipfile
import numpy as np
import h5py
from ..errors import KuboMSDError
from .tables import ALGORITHM,orbital_payload,resolved_settings,validated_tables,native_module

SCHEMA='kubomsd-operators-v2'


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _hash_array(h,a):
    a=np.ascontiguousarray(a);h.update(str(a.dtype).encode());h.update(_json(a.shape).encode());h.update(memoryview(a).cast('B'))


def table_key(orbitals,options):
    h=hashlib.sha256((SCHEMA+ALGORITHM+'ABACUS-real-CS;bohr;float64').encode())
    for o in orbitals.values():h.update(o.provenance.get('sha256','').encode())
    payload=orbital_payload(orbitals)
    for k,v in sorted(payload.items()):
        h.update(k.encode())
        if isinstance(v,np.ndarray):_hash_array(h,v)
        else:h.update(str(v).encode())
    cfg=resolved_settings(options.tolerance,options.table_settings,options.memory_limit_bytes);cfg.pop('memory_limit_bytes')
    h.update(_json(dict(settings=cfg,tolerance=options.tolerance)).encode())
    return h.hexdigest()


def material_key(structure,orbitals,h_images,options,*,input_records=None):
    from .basis import convention_for
    h=hashlib.sha256(table_key(orbitals,options).encode())
    h.update(convention_for(structure).encode())
    h.update(_json(structure['symbols']).encode())
    if input_records is not None:h.update(_json(sorted(r['sha256'] for r in input_records)).encode())
    for k in ('cell_A','positions_A'):_hash_array(h,np.asarray(structure[k],float))
    for k,v in sorted(h_images.items()):h.update(_json(k).encode());_hash_array(h,np.asarray(v,complex))
    h.update(b'Gamma;xyz;bra-origin;reference' if options.operator_builder=='reference' else b'Gamma;xyz;bra-origin;fast')
    if options.operator_builder=='reference':h.update(_json(options.reference_settings).encode())
    return h.hexdigest()


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*2**20),b''):h.update(block)
    return h.hexdigest()


def _ready(directory,kind,key,files):
    for name in files:
        with (directory/name).open('rb') as stream:os.fsync(stream.fileno())
    record=dict(schema=SCHEMA,kind=kind,physical_key=key,files={p:_sha(directory/p) for p in files})
    p=directory/'CACHE_READY';p.write_text(_json(record)+'\n')
    with p.open('rb') as f:os.fsync(f.fileno())


def verify_cache(directory,kind='model',key=None):
    directory=Path(directory);p=directory/'CACHE_READY'
    if not p.is_file():raise KuboMSDError(f'cache has no CACHE_READY: {directory}')
    receipt=json.loads(p.read_text())
    if receipt.get('schema')!=SCHEMA or receipt.get('kind')!=kind or (key is not None and receipt.get('physical_key')!=key):raise KuboMSDError('cache schema/type/physical key mismatch')
    expected={'table.npz','table.json'} if kind=='table' else {'model.json','operators.h5','build_report.json'}
    if set(receipt.get('files',{}))!=expected:raise KuboMSDError('cache receipt incomplete')
    for name,digest in receipt['files'].items():
        if not (directory/name).is_file() or _sha(directory/name)!=digest:raise KuboMSDError(f'cache content hash mismatch: {name}')
    return receipt


def _write_table(path,table,meta):
    state=table.state();arrays={};description={k:v for k,v in state.items() if k not in ('payload','channel_keys','values')}
    description['payload']={k:v for k,v in state['payload'].items() if not isinstance(v,np.ndarray)}
    for k,v in state['payload'].items():
        if isinstance(v,np.ndarray):arrays['payload_'+k]=v
    arrays.update(channel_keys=state['channel_keys'],values=state['values'])
    np.savez(path/'table.npz',**arrays)
    (path/'table.json').write_text(_json(dict(state=description,report=meta)))


def _read_table(path,budget):
    # Check ZIP uncompressed sizes before numpy can materialize arrays; reserve
    # space for NPZ arrays, native owned copies, plans and decode workspaces.
    with zipfile.ZipFile(path/'table.npz') as archive:
        uncompressed=sum(info.file_size for info in archive.infolist())
        if uncompressed>budget//4:raise KuboMSDError('cached table payload exceeds memory budget before load')
    if (path/'table.json').stat().st_size>budget//8:raise KuboMSDError('table metadata exceeds memory budget')
    d=json.loads((path/'table.json').read_text());state=d['state'];state['settings']['memory_limit_bytes']=budget
    with np.load(path/'table.npz',allow_pickle=False) as f:
        for key in f.files:
            if key.startswith('payload_'):state['payload'][key[8:]]=f[key]
            else:state[key]=f[key]
    if sum(v.nbytes for v in state.values() if isinstance(v,np.ndarray))>budget/2:raise KuboMSDError('cached tables exceed memory budget')
    return native_module().restore_tables(state),d['report']


def cached_tables(orbitals,options):
    key=table_key(orbitals,options);root=Path(options.table_cache_dir);root.mkdir(parents=True,exist_ok=True);dest=root/key
    with _lock(root/(key+'.lock')):
        if dest.exists():
            verify_cache(dest,'table',key);table,meta=_read_table(dest,options.memory_limit_bytes);meta['cache']='hot';meta['physical_key']=key
            return table,meta
        table,meta=validated_tables(orbitals,options.tolerance,options.table_settings,options.memory_limit_bytes)
        tmp=Path(tempfile.mkdtemp(prefix=key+'.partial-',dir=root))
        try:
            _write_table(tmp,table,meta);read,_=_read_table(tmp,options.memory_limit_bytes)
            for k in ('values','channel_keys'):
                if not np.array_equal(table.state()[k],read.state()[k]):raise KuboMSDError('table roundtrip mismatch')
            _ready(tmp,'table',key,['table.npz','table.json']);os.rename(tmp,dest)
        except BaseException:
            shutil.rmtree(tmp,ignore_errors=True);raise
        meta['physical_key']=key
        return table,meta


def _write_model_files(result,directory,key,provenance):
    started=time.perf_counter()
    matrices={'H':result.H_cov,'S':result.S}
    for a,parts in result.composite.items():
        for name,m in zip(('bond','xbar','delta'),parts):matrices[a+'_'+name]=m
    declarations={}
    with h5py.File(directory/'operators.h5','w') as f:
        for name,m in matrices.items():
            m=m.tocsr();g=f.create_group(name)
            for k,v in dict(data=m.data,indices=m.indices,indptr=m.indptr,shape=m.shape).items():g.create_dataset(k,data=v)
            declarations[name]=dict(path='operators.h5',key=name,shape=list(m.shape),dtype=str(m.dtype),storage='csr',index_base=0,representation='covariant')
        f.flush()
    model=dict(format='kubomsd-model-v1',n_orbitals=result.H_cov.shape[0],basis='nonorthogonal',kpoint=[0,0,0],energy_unit='eV',length_unit='angstrom',
               orbital_order=result.orbital_order,atom_slices=result.atom_slices,hamiltonian=declarations['H'],overlap=declarations['S'],
               velocity_source=dict(type='composite',axes={a:{k:declarations[a+'_'+k] for k in ('bond','xbar','delta')} for a in 'xyz'}),
               operator_cache=dict(schema=SCHEMA,physical_key=key),construction_provenance=provenance or {})
    (directory/'model.json').write_text(_json(model)+'\n')
    result.diagnostics['timings']['cache_write']=time.perf_counter()-started
    read_start=time.perf_counter()
    # Read every data/index array back before publishing READY. No numerical
    # tolerance: storage must preserve exactly the values and sparsity indices.
    with h5py.File(directory/'operators.h5','r') as f:
        for name,m in matrices.items():
            for k in ('data','indices','indptr'):
                if not np.array_equal(f[name][k][:],getattr(m,k)):raise KuboMSDError('model cache roundtrip mismatch')
    result.diagnostics['timings']['cache_readback']=time.perf_counter()-read_start
    result.diagnostics['timings']['total_build']+=time.perf_counter()-started
    (directory/'build_report.json').write_text(_json(result.diagnostics)+'\n')


def publish_model(result,out,physical_key,provenance=None):
    out=Path(out).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    with _lock(out.parent/(out.name+'.lock')):
        if out.exists():verify_cache(out,'model',physical_key);return out/'model.json'
        tmp=Path(tempfile.mkdtemp(prefix=out.name+'.partial-',dir=out.parent))
        try:
            _write_model_files(result,tmp,physical_key,provenance)
            _ready(tmp,'model',physical_key,['model.json','operators.h5','build_report.json'])
            verify_cache(tmp,'model',physical_key)
            os.rename(tmp,out)
        except BaseException:
            shutil.rmtree(tmp,ignore_errors=True);raise
    return out/'model.json'
