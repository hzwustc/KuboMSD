import json
import sys
from pathlib import Path
import numpy as np
from mpi4py import MPI
from kubo_msd.operators.builder import BuilderOptions
from kubo_msd.operators.lcao_integrals import OrbitalSet
from kubo_msd.operators.mpi_builder import build_operators_mpi

out=Path(sys.argv[1]);mode=sys.argv[2];threads=int(sys.argv[3]);comm=MPI.COMM_WORLD
structure=orbitals=h=None
if comm.rank==0:
    r=np.linspace(0,4,401);a=2/np.pi**.25*np.exp(-r*r/2)
    orbitals={'X':OrbitalSet('X',r,[(0,0,a)],[1.],{})}
    structure=dict(cell_A=np.eye(3)*12,positions_A=np.array([[0.,0.,0.],[1.2,.3,.2]]),symbols=['X','X'])
    h={(0,0,0,0,0):np.array([[-1.]]),(1,1,0,0,0):np.array([[1.]]),(0,1,0,0,0):np.array([[.2]])}
    if mode=='root-input':structure=None
opts=BuilderOptions(operator_builder='cpp-omp',threads=threads,build_distribution='mpi',pair_chunk_size=1,progress_path=out/'progress.json')
if mode!='ok':
    import kubo_msd.operators.mpi_builder as mb
    def injection(phase,rank,chunk=None):
        if mode=='worker' and phase=='compute' and rank==1:raise RuntimeError('injected worker compute error')
        if mode=='cleanup' and phase=='cleanup' and rank==0:raise RuntimeError('injected cleanup failure')
        if mode=='root-write' and phase=='merge' and rank==0:
            import kubo_msd.operators.cache as cache
            def failed_write(*args,**kwargs):raise OSError('injected root cache write failure')
            cache._write_model_files=failed_write
        if mode in ('missing','duplicate','corrupt') and phase=='after_shards' and rank==0:
            files=sorted(out.glob('build-*/shards/rank-*/chunk-*.npz'))
            if mode=='missing':files[0].unlink()
            elif mode=='corrupt':files[0].write_bytes(b'corrupt')
            else:
                with np.load(files[1]) as f: values={k:f[k] for k in f.files}
                values['pair_ids'][:]=0;np.savez(files[1],**values)
    mb._test_hook=injection
try:
    result=build_operators_mpi(structure,orbitals,h,opts,comm)
    from kubo_msd.parallel import ParallelContext
    from kubo_msd.operators.cache import publish_model,material_key
    ParallelContext(comm,MPI).stage('publish model',lambda:publish_model(result,out/'operators',material_key(structure,orbitals,h,opts)),root_only=True)
    if comm.rank==0:
        np.savez(out/'matrices.npz',H=result.H_cov.toarray(),S=result.S.toarray(),**{a+str(i):m.toarray() for a,parts in result.composite.items() for i,m in enumerate(parts)})
        (out/'diagnostics.json').write_text(json.dumps(result.diagnostics,indent=2))
except Exception as exc:
    out.mkdir(parents=True,exist_ok=True);(out/f'failure-{comm.rank}.txt').write_text(str(exc));raise
