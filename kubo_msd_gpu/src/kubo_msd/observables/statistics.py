"""Paired independent-block bootstrap of sums, retaining unequal block counts."""
import numpy as np
from ..errors import KuboMSDError

def estimate_ratio(block_numerators,block_dos,counts,n_resamples=256):
    A=np.asarray(block_numerators,float); D=np.asarray(block_dos,float); counts=np.asarray(counts,int)
    if A.shape[0]!=len(counts) or D.shape[0]!=len(counts) or np.any(counts<=0): raise KuboMSDError('inconsistent statistical blocks/counts')
    if D.ndim!=2 or A.shape[-1]!=D.shape[-1]:
        if D.shape==(len(counts),1): D=np.broadcast_to(D,(len(counts),A.shape[-1]))
        else: raise KuboMSDError('statistical denominator must be (blocks,energy)')
    totalD=D.sum(0); totalA=A.sum(0)
    value=np.divide(totalA,totalD,out=np.full_like(totalA,np.nan),where=totalD!=0)
    status={'method':'paired block bootstrap of sums','n_blocks':len(counts),'counts':counts.tolist(),'reliable_interval':len(counts)>=8,'bootstrap_seed':21491,'n_resamples':n_resamples,'input_accumulation':'sum; actual unequal counts retained'}
    if len(counts)<2: return value,None,status
    rng=np.random.default_rng(21491); indices=rng.integers(len(counts),size=(n_resamples,len(counts)))
    weights=np.zeros((n_resamples,len(counts)))
    for j,row in enumerate(indices): weights[j]=np.bincount(row,minlength=len(counts))
    resampledA=np.tensordot(weights,A,axes=(1,0)); resampledD=weights@D
    resampledD=resampledD.reshape((n_resamples,)+(1,)*(A.ndim-2)+(A.shape[-1],))
    ratios=np.divide(resampledA,resampledD,out=np.full_like(resampledA,np.nan),where=resampledD!=0)
    # Invalid resamples are not silently omitted to improve confidence intervals.
    stderr=np.std(ratios,axis=0,ddof=1)
    return value,stderr,status
