#!/usr/bin/env python3
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.operators.velocity import OperatorBundle
from kubo_msd.backends.gpu import GPUBackend
from kubo_msd.solvers.exact import iter_exact_blocks
from cupy.cuda.memory_hook import MemoryHook

class PeakPoolMemory(MemoryHook):
    name='KuboMSDDataflowBenchmark'
    def __init__(self):self.live={};self.current=0;self.peak=0
    def malloc_postprocess(self,**kw):
        key=(kw['mem_ptr'],kw['pmem_id'])
        self.current+=kw['mem_size']-self.live.get(key,0)
        self.live[key]=kw['mem_size'];self.peak=max(self.peak,self.current)
    def free_postprocess(self,**kw):
        self.current-=self.live.pop((kw['mem_ptr'],kw['pmem_id']),0)

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();n=128
S=sp.diags([.03*np.ones(n-1),np.ones(n),.03*np.ones(n-1)],[-1,0,1],dtype=complex)
H=sp.diags([.1*np.ones(n-1),np.linspace(-1,1,n),.1*np.ones(n-1)],[-1,0,1],dtype=complex)
model=ModelBundle(H,S,position={'x':sp.diags(np.arange(n),dtype=complex)})
cfg=RunConfig(np.linspace(-2,2,21),[0,.05,.1],backend='gpu',axes=('x',),time_block=3,metric_rtol=1e-13)
rows=[]
for rep in range(4):
    backend=GPUBackend();backend.synchronize();hook=PeakPoolMemory()
    with hook:
        start=time.perf_counter();ops=OperatorBundle(model,cfg,backend);backend.synchronize();init=time.perf_counter()-start
        blocks=list(iter_exact_blocks(model,ops,cfg));backend.synchronize()
        elapsed=time.perf_counter()-start
    if rep:rows.append(dict(init_seconds=init,total_seconds=elapsed,memory=backend.memory_info(),pool_peak_during_run_bytes=hook.peak))
root=Path(__file__).resolve().parents[1]
files=['backends/gpu.py','operators/velocity.py','metric/solve.py','metric/sampling.py','solvers/exact.py']
import kubo_msd
package=Path(kubo_msd.__file__).parent
a.out.write_text(json.dumps(dict(n=n,rows=rows,version=kubo_msd.__version__,package=str(package),median_seconds=float(np.median([r['total_seconds'] for r in rows])),memory_scope='peak newly live CuPy pool allocations inside measured run; excludes existing allocations, CUDA context and non-pool library allocations',source_hashes={f:hashlib.sha256((package/f).read_bytes()).hexdigest() for f in files}),indent=2))
