#!/usr/bin/env python3
"""Fail-closed real GPU end-to-end acceptance with matched CPU references.

Synthetic numerical acceptance only. Material inputs and production speedups
are separate. Run from an installed wheel too; no source-path injection.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import gc
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import scipy
import scipy.linalg as la
from numpy.testing import assert_allclose

from kubo_msd.backends.gpu import GPUBackend
from kubo_msd.config import RunConfig
from kubo_msd.dispatch import run_input
from kubo_msd.io.results import read_results


def synthetic_model(folder,n=24):
    folder.mkdir()
    rng=np.random.default_rng(5090128)
    A=np.eye(n)+.015*(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
    Q=la.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
    H=A.conj().T@Q@np.diag(np.linspace(-.6,.7,n))@Q.conj().T@A
    def matrix(name,value):
        np.savez(folder/(name+'.npz'),matrix=value)
        return dict(path=name+'.npz',shape=[n,n],storage='dense',dtype='complex128',index_base=0,representation='covariant')
    velocities={}
    for axis in 'xyz':
        V=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        velocities[axis]=matrix('V'+axis,A.conj().T@(.02*(V+V.conj().T))@A)
    model=dict(format='kubomsd-model-v1',n_orbitals=n,basis='nonorthogonal',kpoint=[0,0,0],energy_unit='eV',length_unit='angstrom',
               orbital_order=[f'{j}:s' for j in range(n)],hamiltonian=matrix('H',H),overlap=matrix('S',A.conj().T@A),
               velocity_source=dict(type='matrix',unit='angstrom/fs',matrices=velocities))
    path=folder/'model.json'; path.write_text(json.dumps(model,indent=2)+'\n')
    return path


def compare(reference,actual):
    assert reference.keys()==actual.keys()
    absolute={}
    for key,want in reference.items():
        got=actual[key]
        if want.dtype.kind in 'OUS':
            np.testing.assert_array_equal(got,want)
            continue
        assert_allclose(got,want,rtol=1e-9,atol=1e-11,equal_nan=True,err_msg=key)
        finite=np.isfinite(want)&np.isfinite(got)
        absolute[key]=float(np.max(np.abs(got[finite].astype(float)-want[finite].astype(float)))) if finite.any() else 0.
    return absolute


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    backend=GPUBackend()  # required: never skip or switch to CPU
    cp=backend.xp
    from cupy.cuda.memory_hook import MemoryHook
    class PeakPoolMemory(MemoryHook):
        name='KuboMSDAcceptanceMemory'
        def __init__(self): self.live={}; self.current=0; self.peak=0
        def malloc_postprocess(self,**kw):
            key=(kw['mem_ptr'],kw['pmem_id'])
            self.current+=kw['mem_size']-self.live.get(key,0)
            self.live[key]=kw['mem_size']; self.peak=max(self.peak,self.current)
        def free_postprocess(self,**kw):
            self.current-=self.live.pop((kw['mem_ptr'],kw['pmem_id']),0)
    config_text=io.StringIO()
    with redirect_stdout(config_text): cp.show_config()
    forbidden={name:importlib.util.find_spec(name) is None for name in ('rsatb','_rsatb')}
    assert all(forbidden.values()),forbidden
    report={'status':'RUNNING','device':backend.memory_info(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
            'cupy':cp.__version__,'cupy_config':config_text.getvalue(),'forbidden_dependencies_absent':forbidden,
            'optional_mpi_installed':importlib.util.find_spec('mpi4py') is not None,
            'tolerance':{'rtol':1e-9,'atol':1e-11},'scope':'synthetic inputs; same CPU/GPU configurations and random sample IDs; no material acceptance or production speedup claim',
            'timing_scope':'run_input with initialization, transport and output; existing CUDA context; memory hook overhead included',
            'memory_scope':'peak live allocations through CuPy pool during each run; excludes CUDA context, non-pool vendor allocations and other processes',
            'runs':[]}
    report['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True,capture_output=True,check=True).stdout.strip()
    path=args.out/'report.json'
    root=Path(__file__).resolve().parents[1]
    inputs=[('complex24',synthetic_model(args.out/'model_complex24'),('x','y','z')),('raw_lcao',root/'examples/raw_lcao/input.json',('x',))]
    try:
        for name,model,axes in inputs:
            cfg=RunConfig(energy_eV=np.linspace(-.1,.4,5),time_fs=[0,.03,.08,.15],axes=axes,kernel='jackson',n_moments=16,
                          nsamples=9,sample_batch=4,statistics_block=2,seed=715,metric_rtol=1e-12,sampling_tol=1e-12,
                          time_block=2,pair_block=5,energy_block=2,segment_fs=.08,integration_dt_fs=.025,output_npz=True)
            references={}
            for method in ('exact','stepwise','spectral','auto'):
                base=cfg if method=='exact' else replace(cfg,solver='stochastic-kpm',stochastic_method=method)
                for device in (('cpu','hybrid','gpu') if method=='exact' else ('cpu','gpu')):
                    run_cfg=replace(base,backend=device)
                    folder=args.out/f'{name}_{method}_{device}'
                    gc.collect(); cp.get_default_memory_pool().free_all_blocks()
                    hook=PeakPoolMemory(); start=time.perf_counter()
                    with hook: summary=run_input(model,run_cfg,folder)
                    cp.cuda.get_current_stream().synchronize()
                    wall=time.perf_counter()-start
                    result=read_results(folder)
                    assert result['metadata']['run_status']=='complete'
                    assert result['metadata']['backend_used']==device
                    assert (folder/'results.npz').is_file()
                    if device=='cpu': references[method]=result['arrays']; errors={}
                    else: errors=compare(references[method],result['arrays'])
                    row={'input':name,'method':method,'backend':device,'status':'PASS','wall_seconds':wall,
                         'peak_live_cupy_pool_bytes':hook.peak,'max_abs_error_by_array':errors,'diagnostics':summary.diagnostics,
                         'eigensolve':result['metadata'].get('eigensolve'),'initialization':result['metadata']['initialization'],
                         'method_decision':result['metadata']['method_decision']}
                    report['runs'].append(row)
                    print(f'{name} {method}/{device}: PASS ({wall:.3f} s)',flush=True)
        report['status']='PASS'
    except BaseException as exc:
        report.update(status='FAIL',error=repr(exc)); raise
    finally:
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(path)


if __name__=='__main__': main()
