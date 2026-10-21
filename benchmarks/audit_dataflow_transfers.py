#!/usr/bin/env python3
"""Intercept actual CuPy array conversion calls across all backend instances."""
import argparse,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import cupy as cp
import kubo_msd
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.operators.velocity import OperatorBundle
from kubo_msd.backends.gpu import GPUBackend
from kubo_msd.solvers.exact import iter_exact_blocks

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();n=128
original=cp.asarray;transfers=[]
def watched(v,*args,**kwargs):
    out=original(v,*args,**kwargs)
    if not isinstance(v,cp.ndarray):transfers.append(dict(shape=list(out.shape),bytes=out.nbytes,direction='H2D'))
    return out
cp.asarray=watched
S=sp.diags([.03*np.ones(n-1),np.ones(n),.03*np.ones(n-1)],[-1,0,1],dtype=complex)
H=sp.diags([.1*np.ones(n-1),np.linspace(-1,1,n),.1*np.ones(n-1)],[-1,0,1],dtype=complex)
model=ModelBundle(H,S,position={'x':sp.diags(np.arange(n),dtype=complex)})
cfg=RunConfig(np.linspace(-2,2,21),[0,.05,.1],backend='gpu',axes=('x',),time_block=3,metric_rtol=1e-13)
backend=GPUBackend();transfers.clear();ops=OperatorBundle(model,cfg,backend)
blocks=list(iter_exact_blocks(model,ops,cfg));backend.synchronize()
a.out.write_text(json.dumps(dict(package=kubo_msd.__file__,n=n,cupy_asarray_h2d_bytes=sum(r['bytes'] for r in transfers),dense_NxN_uploads=sum(1 for r in transfers if r['shape']==[n,n]),calls=transfers,scope='all actual cupy.asarray host conversions, including backend instances; internal runtime/driver copies not claimed'),indent=2))
