"""Official raw-input audit/pilot/build command; never submits scheduler jobs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from ..errors import KuboMSDError
from ..io.legacy_input import parse_legacy
from ..parallel import ParallelContext,check_serial_launcher
from .builder import BuilderOptions,build_operators,pair_tasks,get_tables,thread_context
from .cache import material_key,publish_model,verify_cache
from .progress import BuildReport


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--mode',choices=['audit','pilot','build'],default='audit')
    p.add_argument('--operator-builder',choices=['reference','cpp','cpp-omp'],default='cpp')
    p.add_argument('--build-distribution',choices=['serial','mpi'],default='serial')
    p.add_argument('--integral-tol',type=float,default=1e-5);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--table-cache',type=Path);p.add_argument('--memory-limit-gib',type=float,default=2.)
    p.add_argument('--pair-chunk-size',type=int,default=128);p.add_argument('--mpi-io-batch-mib',type=int,default=64)
    p.add_argument('--expect-atoms',type=int);p.add_argument('--expect-orbitals',type=int)
    p.add_argument('--kmax',type=float);p.add_argument('--dk',type=float);p.add_argument('--distance-step',type=float);p.add_argument('--radial-order',type=int)
    p.add_argument('--pilot-pairs-per-type',type=int,default=3)
    return p


def main(argv=None):
    a=parser().parse_args(argv);started=time.perf_counter()
    if a.build_distribution=='mpi':
        from mpi4py import MPI
        ctx=ParallelContext(MPI.COMM_WORLD,MPI)
    else:check_serial_launcher();ctx=ParallelContext()
    def root_parse():
        data=json.loads(a.input.read_text());parsed=parse_legacy(a.input,data,read_h=a.mode=='build');s,o,h,p=parsed
        n=sum(o[k].n_orbitals for k in s['symbols'])
        if a.expect_atoms is not None and len(s['symbols'])!=a.expect_atoms:raise KuboMSDError('atom count assertion failed')
        if a.expect_orbitals is not None and n!=a.expect_orbitals:raise KuboMSDError('orbital count assertion failed')
        return parsed
    parsed=ctx.stage('root parse inputs',root_parse,root_only=True);read_s=time.perf_counter()-started
    def options():
        if a.pilot_pairs_per_type<1:raise KuboMSDError('pilot-pairs-per-type must be positive')
        return BuilderOptions(operator_builder=a.operator_builder,tolerance=a.integral_tol,threads=a.threads,
            build_distribution=a.build_distribution,table_cache_dir=a.table_cache,memory_limit_bytes=int(a.memory_limit_gib*2**30),
            pair_chunk_size=a.pair_chunk_size,mpi_io_batch_bytes=a.mpi_io_batch_mib*2**20,
            progress_path=a.out.parent/(a.out.name+'.progress.json'),
            table_settings={k:getattr(a,k) for k in ('kmax','dk','distance_step','radial_order') if getattr(a,k) is not None})
    opts=ctx.stage('validate build options',options)
    if ctx.root:structure,orbitals,h,provenance=parsed
    else:structure=orbitals=h=provenance=None
    def write_json(name,value):
        a.out.mkdir(parents=True,exist_ok=True);p=a.out/name;p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');print(p,flush=True)
    if a.mode=='audit':
        def audit():
            keys,d,species=pair_tasks(structure,orbitals,opts.memory_limit_bytes)
            return dict(status='PASS',mode='audit',n_atoms=len(structure['symbols']),n_orbitals=sum(orbitals[s].n_orbitals for s in structure['symbols']),unique_pairs=len(keys),hamiltonian_available=Path(provenance['hamiltonian_path']).is_file(),provenance=provenance,read_inputs_s=read_s)
        record=ctx.root_call('audit geometry',audit)
        ctx.stage('write audit',lambda:write_json('audit.json',record),root_only=True)
        return 0
    if a.mode=='pilot':
        # Pilot validates sampled pairs only; it never publishes a full model READY.
        from .mpi_builder import broadcast_tree
        from .tables import native_module
        from .reference import evaluate_reference_pair
        def selected():
            keys,d,species=pair_tasks(structure,orbitals,opts.memory_limit_bytes);indices=[]
            for kind in sorted(set(map(tuple,species))):
                ids=[i for i,v in enumerate(species) if tuple(v)==kind and np.linalg.norm(d[i])>1e-12]
                ids=sorted(ids,key=lambda i:np.linalg.norm(d[i]))
                indices.extend(ids[j] for j in np.unique(np.linspace(0,max(0,len(ids)-1),min(len(ids),a.pilot_pairs_per_type),dtype=int)))
            for kind in sorted(set(map(tuple,species))):
                onsite=[i for i,v in enumerate(species) if tuple(v)==kind and np.linalg.norm(d[i])<=1e-12]
                if onsite:indices.append(onsite[0])
            if not indices:raise KuboMSDError('pilot selected no pairs; no numerical validation performed')
            return dict(keys=keys[indices],displacement_A=d[indices],species=species[indices])
        samples=ctx.root_call('pilot selection',selected)
        # Orbital input is small. Full H and material matrices are never broadcast.
        orbitals=ctx.broadcast(orbitals)
        report=BuildReport(opts.operator_builder)
        with thread_context():
            if opts.operator_builder!='reference':
                table=ctx.stage('pilot root table',lambda:get_tables(orbitals,opts,report),root_only=True)
                if ctx.comm is not None:
                    state=broadcast_tree(ctx,table.state() if ctx.root else None,opts.mpi_io_batch_bytes)
                    table=ctx.stage('pilot restore',lambda:table if ctx.root else native_module().restore_tables(state))
            else:
                if ctx.size>1:raise KuboMSDError('MPI reference pilot not supported')
                table=None
            names=list(orbitals)
            def calculate():
                result=[]
                for i in range(ctx.rank,len(samples['keys']),ctx.size):
                    ia,ib=samples['species'][i];oa,ob=orbitals[names[ia]],orbitals[names[ib]];d=samples['displacement_A'][i]
                    s,D,ref=evaluate_reference_pair(oa,ob,d,dict(tolerance=opts.tolerance))
                    error=None
                    if table is not None:
                        v=native_module().evaluate_pairs(table,dict(species=samples['species'][i:i+1],displacement_A=d[None,:],pair_ids=np.array([i],np.int64)),opts.threads)
                        error=float(max(np.max(abs(v['S']-s.ravel())),np.max(abs(v['D']-D.reshape(3,-1)))/max(1.,oa.cutoff_A+ob.cutoff_A)))
                    result.append(dict(pair_id=i,key=samples['keys'][i].tolist(),displacement_A=d.tolist(),reference=ref,scaled_difference=error,rank=ctx.rank,status='PASS' if ref['status']=='PASS' and (error is None or error<=opts.tolerance) else 'FAIL'))
                return result
            rows=ctx.stage('pilot pairs',calculate);rows=sorted([r for part in ctx.allgather(rows) for r in part],key=lambda r:r['pair_id'])
        record=dict(mode='pilot',status='PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL',samples=rows,table=report.details.get('table'),wall_s=time.perf_counter()-started)
        ctx.stage('write pilot',lambda:write_json('pilot.json',record),root_only=True)
        return 0 if record['status']=='PASS' else 1
    key=ctx.root_call('physical input key',lambda:material_key(structure,orbitals,h,opts,input_records=[r for r in provenance['files'] if Path(r['path'])!=a.input.resolve()]))
    def existing():
        if a.out.exists():verify_cache(a.out,'model',key);return True
        return False
    if ctx.root_call('existing complete cache',existing):
        if ctx.root:print(a.out/'model.json',flush=True)
        return 0
    if ctx.comm is None:result=build_operators(structure,orbitals,h,opts)
    else:
        from .mpi_builder import build_operators_mpi
        result=build_operators_mpi(structure,orbitals,h,opts,ctx.comm)
    def publish():
        result.diagnostics['timings']['read_inputs']=read_s
        result.diagnostics['timings']['total_build']=time.perf_counter()-started
        p=publish_model(result,a.out,key,provenance);print(p,flush=True)
    ctx.stage('single writer publish',publish,root_only=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
