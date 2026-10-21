#!/usr/bin/env python3
"""Fresh wheel environment; optional matching local MPI wheel; no source imports."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--mpi-wheel',type=Path)
    p.add_argument('--dependency-wheel',type=Path,action='append',default=[])
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'PYTHONNOUSERSITE':'1','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'2'}
    env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
    logs=[]
    def run(argv,timeout=180):
        r=subprocess.run(list(map(str,argv)),cwd=a.out,env=env,capture_output=True,text=True,timeout=timeout)
        logs.append(dict(argv=list(map(str,argv)),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        (a.out/'commands.json').write_text(json.dumps(logs,indent=2))
        if r.returncode:raise RuntimeError(r.stdout+r.stderr)
        return r.stdout
    # Reuse only the base scientific dependencies. KuboMSD/MPI are installed
    # afresh; this is installation isolation, not dependency-version coverage.
    run([sys._base_executable,'-m','venv','--system-site-packages',a.out/'venv'])
    python=a.out/'venv/bin/python'
    for dependency in a.dependency_wheel:
        run([python,'-m','pip','install','--no-deps',dependency.resolve()])
    run([python,'-m','pip','install','--no-deps',a.wheel.resolve()])
    run([a.out/'venv/bin/kubomsd-prepare','--help'])
    shutil.copytree(root/'examples/raw_lcao',a.out/'raw')
    probe="""
import importlib.util,json,kubo_msd
from pathlib import Path
from kubo_msd import _native,_operators_native
assert str(Path(kubo_msd.__file__).resolve()).startswith(str(Path('venv').resolve()))
assert all(importlib.util.find_spec(n) is None for n in ('mpi4py','cupy','rsatb','_rsatb'))
print(json.dumps(dict(version=kubo_msd.__version__,module=kubo_msd.__file__,
 native=_native.build_info(),operators=_operators_native.capabilities())))
"""
    installed=json.loads(run([python,'-c',probe]))
    prepare=[python,'-m','kubo_msd.operators.prepare','--input','raw/input.json',
             '--out','serial_operators','--mode','build','--operator-builder','cpp-omp','--threads','2']
    run(prepare)
    common=['-m','kubo_msd','run','--input','serial_operators/model.json',
            '--solver','stochastic-kpm','--kernel','jackson','--backend','cpu',
            '--omp-threads','2','--stochastic-method','stepwise','--emin','-.5','--emax','.5',
            '--n-energy','9','--tmax-fs','.1','--dt-fs','.05','--axes','x','--n-moments','32',
            '--spectral-min','-2','--spectral-max','2','--nsamples','17','--statistics-block','2',
            '--sample-batch','2','--no-vac']
    for kernel in ('cpp','reference'):
        run([python,*common,'--cpu-kernel',kernel,'--out',kernel])
    compare="""
import sys,numpy as np
from kubo_msd.io.results import read_results
a=read_results(sys.argv[1])['arrays'];b=read_results(sys.argv[2])['arrays']
assert a.keys()==b.keys()
for k in a:
 if a[k].dtype.kind in 'OUS':np.testing.assert_array_equal(a[k],b[k])
 else:np.testing.assert_allclose(a[k],b[k],rtol=1e-9,atol=1e-11,equal_nan=True,err_msg=k)
print('all result arrays PASS')
"""
    run([python,'-c',compare,'reference','cpp'])
    for name,options in (
        ('metric_adaptive',['--metric-residual-policy','adaptive']),
        ('metric_cholesky',['--metric-solver','cholesky']),
        ('metric_ic0',['--preconditioner','ic0','--metric-residual-policy','adaptive']),
    ):
        run([python,*common,'--cpu-kernel','cpp','--out',name,*options])
        run([python,'-c',compare,'reference',name])
    installed_metrics="""
import json,numpy as np,scipy.sparse as sp
from pathlib import Path
from kubo_msd.metric.partition import make_spatial_partition
from kubo_msd.metric.solve import MetricSolver
from kubo_msd.config import RunConfig
from kubo_msd.validation import matrix_fingerprint
S=sp.csr_matrix([[2,.1j,.4],[-.1j,1,.1j],[.4,-.1j,1.5]])
atoms=[(0,1),(1,2),(2,3)];identity=dict(physical_key='installed',orbital_order_hash='fixture')
p=make_spatial_partition(dict(cell_A=np.eye(3)*10,positions_A=[[0,0,0],[5,0,0],[.2,0,0]],
    symbols=['A']*3),atoms,max_orbitals=2,s_fingerprint=matrix_fingerprint(S),**identity)
Path('partition.json').write_text(json.dumps(p))
c=RunConfig([0],[0],solver='stochastic-kpm',kernel='jackson',preconditioner='cluster-jacobi',
    metric_partition_file='partition.json',metric_rtol=1e-13)
m=MetricSolver(S,c,atom_slices=atoms,metric_identity=identity)
B=np.array([[1j,0],[2,0],[3j,0]])
Y,r=m.solve_s(B)
assert m.native is not None and np.all(r.converged)
np.testing.assert_allclose(Y,np.linalg.solve(S.toarray(),B),rtol=1e-12,atol=1e-12)
print('installed noncontiguous cluster native PASS')
"""
    run([python,'-c',installed_metrics])
    report=dict(status='PASS',installed=installed,serial='PASS',mpi='NOT RUN',
                stochastic_metric=dict(adaptive='PASS',cholesky='PASS',ic0='PASS',cluster_jacobi='PASS'),
                isolation='fresh venv and wheel; source PYTHONPATH removed; base scientific libraries reused; explicit additional dependency wheels installed separately',
                dependency_wheels=list(map(str,a.dependency_wheel)),
                environment=str(a.out/'venv'),commands=str(a.out/'commands.json'))
    if a.mpi_wheel:
        run([python,'-m','pip','install','--no-deps',a.mpi_wheel.resolve()])
        launcher=a.out/'venv/bin/mpiexec'
        report['mpi_library']=run([python,'-c','from mpi4py import MPI; print(MPI.Get_library_version())'])
        mpi_prepare=[str(x).replace('serial_operators','mpi_operators') for x in prepare]
        run([launcher,'-n','2',*mpi_prepare,'--build-distribution','mpi'])
        check="""
from kubo_msd.io.model import load_model
import numpy as np
a=load_model('serial_operators/model.json');b=load_model('mpi_operators/model.json')
for x,y in [(a.H_cov,b.H_cov),(a.S,b.S)]+[(x,y) for q in 'xyz' for x,y in zip(a.composite[q],b.composite[q])]:
 np.testing.assert_array_equal(x.indptr,y.indptr);np.testing.assert_array_equal(x.indices,y.indices)
 np.testing.assert_allclose(x.data,y.data,rtol=1e-12,atol=1e-13)
print('installed MPI build PASS')
"""
        run([python,'-c',check])
        run([launcher,'-n','2',python,*common,'--transport-distribution','mpi',
             '--cpu-kernel','cpp','--out','mpi_transport'])
        run([python,'-c',compare,'cpp','mpi_transport'])
        report['mpi']='PASS'
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(a.out/'report.json')


if __name__=='__main__':
    main()
