"""Device radial quadrature and independently identified, validated table cache."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .contracts import DeviceTables, PairPlan
from .constants import ALGORITHM, orbital_payload, channel_definition, angular_terms
from ...units import BOHR_A
from ...errors import NumericalError


def resolved_settings(tolerance, settings=None):
    scale=min(1.,(tolerance/1e-5)**.2)
    out=dict(kmax=32./scale,dk=.1*scale,distance_step=.05*scale,radial_order=4)
    if set(settings or {})-set(out): raise ValueError('unknown radial table setting')
    out.update(settings or {})
    if any(not np.isfinite(out[k]) or out[k]<=0 for k in ('kmax','dk','distance_step')):
        raise ValueError('table grids must be finite positive')
    if type(out['radial_order']) is not int or not 2<=out['radial_order']<=16:
        raise ValueError('radial_order must be integer in 2..16')
    return out


def _identity(payload, options, settings):
    h=hashlib.sha256()
    for key,value in payload.items():
        h.update(key.encode()); h.update(np.asarray(value).tobytes())
    for path in sorted(Path(__file__).parent.glob('kernels/*.cu')):
        h.update(path.name.encode()); h.update(path.read_bytes())
    h.update(Path(__file__).with_name('constants.py').read_bytes())
    h.update(json.dumps({'schema':1,'algorithm':ALGORITHM,'settings':settings,
                         'tolerance':options.integral_tolerance},sort_keys=True).encode())
    return h.hexdigest()


def _raw(payload, cfg, context, identity):
    transforms,channels,transform_pairs=channel_definition(payload)
    cutoff=max(payload['radius'][payload['grid_offsets'][a+1]-1] for a in range(len(payload['radial_l'])))
    nk=int(np.ceil(cfg['kmax']/cfg['dk'])); nk+=nk%2; nk+=1
    nr=max(6,int(np.ceil(2*cutoff/cfg['distance_step']))+1)
    dk=cfg['kmax']/(nk-1); step=2*cutoff/(nr-1)
    nc=len(channels); ntrans=len(transforms)
    if max(nk,nr,nc,ntrans)>=2**31 or nc*nr>=2**31:
        raise ValueError('table dimensions exceed supported kernel indexing')
    context.require_bytes(8*(nc*nr+ntrans*nk)*2,'radial tables')
    constants,sizes=_constants(payload,channels,context)
    with context.phase('table_build_or_load'):
        g,w=np.polynomial.legendre.leggauss(cfg['radial_order'])
        arrays=dict(constants)
        inputs={k:context.asarray(v,'radial_'+k,'input_upload') for k,v in payload.items() if isinstance(v,np.ndarray)}
        tr=context.asarray(transforms,'transform_keys'); ch=context.asarray(channels,'channel_keys')
        tp=context.asarray(transform_pairs,'transform_pairs'); gx=context.asarray(g,'Gauss_nodes'); gw=context.asarray(w,'Gauss_weights')
        F=context.xp.empty((ntrans,nk)); V=context.xp.empty((nc,nr))
        context.kernel('radial.cu','radial_transform')(((ntrans*nk+127)//128,),(128,),
            (tr,inputs['radial_l'],inputs['grid_offsets'],inputs['coeff_offsets'],inputs['radius'],inputs['coefficients'],
             gx,gw,np.int32(cfg['radial_order']),np.int32(nk),np.float64(dk),np.int32(ntrans),F))
        context.kernel('radial.cu','radial_channels')(((nc*nr+127)//128,),(128,),
            (F,tp,ch,np.int32(nk),np.float64(dk),np.int32(nr),np.float64(step),np.int32(nc),V))
        if not bool(context.xp.all(context.xp.isfinite(V))): raise NumericalError('nonfinite radial table')
        arrays['values']=V
    meta={'algorithm':ALGORITHM,'schema':1,'settings':{**cfg,'dk':dk,'distance_step':step},
          'channel_keys':channels.tolist(),'step':step,'ndistance':nr,'nspecies':payload['species_count'],
          'species_sizes':sizes,'table_source':'gpu_cold'}
    return DeviceTables(arrays,meta,identity,context.device_id)


def _constants(payload,channels,context):
    const_id=hashlib.sha256(b''.join(np.asarray(v).tobytes() for v in payload.values())).hexdigest()
    if not hasattr(context,'_table_constants'): context._table_constants={}
    if const_id not in context._table_constants:
        with context.phase('constant_tables'):
            terms=angular_terms(payload,channels)
            arrays={k:context.asarray(v,k,'constants') for k,v in terms.items() if k!='species_sizes'}
            context._table_constants[const_id]=(arrays,terms['species_sizes'].tolist())
    return context._table_constants[const_id]


def _validation_batch(parsed,step):
    orbs=list(parsed.orbitals.values()); sp=[]; dis=[]; shapes=[];scales=[]
    directions=np.array([[1.,2.,3.],[-2.,1.,.7],[.4,-1.,2.],[1.,0.,0.],[0.,1.,1.]])
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    for a,oa in enumerate(orbs):
        for b,ob in enumerate(orbs):
            cut=oa.cutoff_A+ob.cutoff_A
            for j,f in enumerate([0.,.031,.13,.29,.47,.67,.83,.96,.995]):
                r=0. if f==0 else min(cut*(1-1e-8),(np.floor(cut*f/(step*BOHR_A))+.5)*step*BOHR_A)
                sp.append([a,b]);dis.append(r*directions[j%5]);shapes.append([oa.n_orbitals,ob.n_orbitals]);scales.append(max(1,cut))
    shapes=np.array(shapes,np.int64); n=len(sp)
    return PairPlan(np.zeros((n,5),np.int64),np.array(sp,np.int64),np.array(dis),np.arange(n,dtype=np.int64),
                    shapes,np.r_[0,np.cumsum(shapes.prod(1))].astype(np.int64),{}),scales


def _difference(a,b,scales,context):
    xp=context.xp; worst=0.
    for i,scale in enumerate(scales):
        sl=slice(int(a.batch.offsets[i]),int(a.batch.offsets[i+1]))
        worst=max(worst,float(xp.max(xp.abs(a.overlap[sl]-b.overlap[sl]))),
                  float(xp.max(xp.abs(a.dipole[:,sl]-b.dipole[:,sl])))/scale)
    return worst


def build_tables_gpu(parsed, options, context):
    from .pairs import evaluate_pairs_gpu
    payload=orbital_payload(parsed.orbitals)
    cfg=resolved_settings(options.integral_tolerance,options.table_settings)
    identity=_identity(payload,options,cfg)
    cache=Path(options.table_cache_dir)/f'{identity}.npz' if options.table_cache_dir else None
    if cache and cache.exists():
        with np.load(cache,allow_pickle=False) as f:
            meta=json.loads(str(f['metadata']))
            values=np.asarray(f['values'])
        digest=meta.pop('metadata_sha256',None)
        if digest!=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest() or meta['identity']!=identity or meta.get('convergence',{}).get('status')!='PASS' or hashlib.sha256(values.tobytes()).hexdigest()!=meta['values_sha256']:
            raise NumericalError('invalid table cache identity/checksum/validation')
        # Recreate small constants only; never run a radial kernel on a cache hit.
        transforms,channels,_=channel_definition(payload)
        if values.shape!=(len(channels),meta['ndistance']) or not np.isfinite(values).all():
            raise NumericalError('invalid table cache values')
        constants,_=_constants(payload,channels,context)
        arrays=dict(constants)
        arrays['values']=context.asarray(values,'validated_tables','cache_import')
        meta['table_source']='validated_cache'
        return DeviceTables(arrays,meta,identity,context.device_id)
    history=[]
    with context.activate():
        for attempt in range(3):
            table=_raw(payload,cfg,context,identity)
            batch,scales=_validation_batch(parsed,table.metadata['step'])
            base=evaluate_pairs_gpu(table,batch,context)
            tr_cfg={**cfg,'kmax':cfg['kmax']*1.5,'dk':cfg['dk']/2,'radial_order':min(16,cfg['radial_order']+2)}
            refined=_raw(payload,tr_cfg,context,identity)
            error_transform=_difference(base,evaluate_pairs_gpu(refined,batch,context),scales,context)
            del refined
            refined=_raw(payload,{**cfg,'distance_step':cfg['distance_step']/2},context,identity)
            error_interpolation=_difference(base,evaluate_pairs_gpu(refined,batch,context),scales,context)
            del refined,base
            history.append({'settings':table.metadata['settings'],'transform':error_transform,'interpolation':error_interpolation})
            if max(error_transform,error_interpolation)<=options.integral_tolerance/5: break
            del table
            cfg={**tr_cfg,'distance_step':cfg['distance_step']/2}
        else: raise NumericalError(f'table convergence failed epsilon/5: {history}')
    table.metadata.update(identity=identity,convergence={'status':'PASS','tolerance':options.integral_tolerance,
        'history':history,'validation_kind':'finite withheld geometry set; not uniform bound','reference_accuracy':'UNVERIFIED'})
    if cache:
        cache.parent.mkdir(parents=True,exist_ok=True)
        with context.phase('cache_io'):
            values=context.to_host(table.arrays['values'],'radial_tables','cache_export')
            meta={**table.metadata,'values_sha256':hashlib.sha256(values.tobytes()).hexdigest()}
            meta['metadata_sha256']=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()
            fd,tmp=tempfile.mkstemp(prefix=identity+'.',suffix='.npz',dir=cache.parent)
            try:
                with os.fdopen(fd,'wb') as f:
                    np.savez_compressed(f,values=values,metadata=json.dumps(meta,sort_keys=True))
                    f.flush();os.fsync(f.fileno())
                os.replace(tmp,cache)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
    return table
