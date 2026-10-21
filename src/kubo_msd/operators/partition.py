"""Stable chunks and deterministic greedy cost balancing."""
import numpy as np
from ..errors import KuboMSDError


def partition_chunks(costs,chunk_size,size):
    if chunk_size<1 or size<1: raise KuboMSDError('invalid partition')
    chunks=[dict(id=i,start=start,stop=min(len(costs),start+chunk_size),cost=float(np.sum(costs[start:start+chunk_size]))) for i,start in enumerate(range(0,len(costs),chunk_size))]
    loads=[0.]*size
    for c in sorted(chunks,key=lambda c:(-c['cost'],c['id'])):
        owner=min(range(size),key=lambda r:(loads[r],r));c['rank']=owner;loads[owner]+=c['cost']
    return chunks


def check_coverage(ids,total):
    ids=np.asarray(ids,dtype=np.int64)
    if ids.shape!=(total,) or not np.array_equal(np.sort(ids),np.arange(total)):
        raise KuboMSDError('pair coverage missing, duplicate or out of range')
