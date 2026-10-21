"""Paired block bootstrap on device, identical CPU-generated resampling indices."""
import numpy as np
from ..errors import KuboMSDError


def estimate_ratio_gpu(block_numerators, block_dos, counts, context, n_resamples=256):
    xp = context.xp
    A, D = block_numerators, block_dos
    context.check_device(A); context.check_device(D)
    counts = np.asarray(counts, int)
    if A.shape[0]!=len(counts) or D.shape[0]!=len(counts) or np.any(counts<=0):
        raise KuboMSDError('inconsistent statistical blocks/counts')
    if D.ndim!=2 or A.shape[-1]!=D.shape[-1]:
        if D.shape==(len(counts),1): D=xp.broadcast_to(D,(len(counts),A.shape[-1]))
        else: raise KuboMSDError('statistical denominator must be (blocks,energy)')
    totalD,totalA=D.sum(0),A.sum(0)
    value=xp.where(totalD!=0,totalA/xp.where(totalD!=0,totalD,1),xp.nan)
    status={'method':'paired block bootstrap of sums','n_blocks':len(counts),'counts':counts.tolist(),
            'reliable_interval':len(counts)>=8,'bootstrap_seed':21491,'n_resamples':n_resamples,
            'input_accumulation':'sum; actual unequal counts retained'}
    if len(counts)<2: return value,None,status
    context.require_bytes(n_resamples*totalA.size*8*4+n_resamples*len(counts)*16,
                          'paired bootstrap slab')
    rng=np.random.default_rng(21491)
    indices=rng.integers(len(counts),size=(n_resamples,len(counts)))
    weights=np.zeros((n_resamples,len(counts)))
    for j,row in enumerate(indices): weights[j]=np.bincount(row,minlength=len(counts))
    w=context.asarray(weights,'paired_bootstrap_weights','constants')
    ra=xp.tensordot(w,A,axes=(1,0)); rd=w@D
    rd=rd.reshape((n_resamples,)+(1,)*(A.ndim-2)+(A.shape[-1],))
    ratios=xp.where(rd!=0,ra/xp.where(rd!=0,rd,1),xp.nan)
    return value,xp.std(ratios,axis=0,ddof=1),status
