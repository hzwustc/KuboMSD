import numpy as np
from .contracts import DevicePairBlocks


def evaluate_pairs_gpu(tables, pair_batch, context):
    if tables.device_id != context.device_id: raise ValueError('tables on wrong device')
    b=pair_batch; ids=np.asarray(b.pair_ids)
    if ids.ndim!=1 or np.any(ids<0) or len(np.unique(ids))!=len(ids):
        raise ValueError('pair IDs must be unique nonnegative integers')
    n=len(ids); ns=tables.metadata['nspecies']; sizes=np.asarray(tables.metadata['species_sizes'])
    if b.species.shape!=(n,2) or np.any(b.species<0) or np.any(b.species>=ns):
        raise ValueError('invalid species layout')
    if b.displacement_A.shape!=(n,3) or not np.isfinite(b.displacement_A).all():
        raise ValueError('invalid displacement layout')
    expected=sizes[b.species[:,0]]*sizes[b.species[:,1]]
    if b.offsets.shape!=(n+1,) or b.offsets[0]!=0 or not np.array_equal(np.diff(b.offsets),expected):
        raise ValueError('invalid pair block offsets')
    xp=context.xp; a=tables.arrays; nc=len(tables.metadata['channel_keys']); nr=tables.metadata['ndistance']
    total=int(b.offsets[-1])
    context.require_bytes(total*32+n*(nc+68)*8+n*5*8,'pair evaluation scratch')
    with context.phase('pair_evaluation'):
        d=context.asarray(np.ascontiguousarray(b.displacement_A),'pair_displacement','input_upload',np.float64)
        s=context.asarray(np.ascontiguousarray(b.species),'pair_species','input_upload',np.int64)
        offsets=context.asarray(np.ascontiguousarray(b.offsets),'pair_offsets','input_upload',np.int64)
        did=context.asarray(np.ascontiguousarray(ids),'pair_ids','input_upload',np.int64)
        angular=xp.empty((n,64)); distances=xp.empty(n); radial=xp.empty((n,nc)); out=xp.empty((4,total))
        if n:
            context.kernel('pairs.cu','pair_angular')(((n+63)//64,),(64,),(d,np.int32(n),angular,distances))
            context.kernel('pairs.cu','pair_interpolate')(((n*nc+127)//128,),(128,),
                (a['values'],distances,np.int32(n),np.int32(nc),np.int32(nr),np.float64(tables.metadata['step']),radial))
            context.kernel('pairs.cu','pair_contract')(((total*4+127)//128,),(128,),
                (s,offsets,np.int32(n),np.int32(ns),a['group_offsets'],a['cutoffs'],a['term_offsets'],
                 a['term_channels'],a['term_harmonics'],a['term_coefficients'],radial,angular,distances,
                 np.int32(nc),np.int64(total),out))
    return DevicePairBlocks(out[0],out[1:],did,offsets,b,context.device_id)
