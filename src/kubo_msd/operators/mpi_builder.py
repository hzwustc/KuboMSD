"""Root input/table ownership, chunked broadcasts and deterministic shard merge.

Ordinary Python/native exceptions are coordinated at stage boundaries. Process
loss is NOT fault tolerant MPI; run under a scheduler/launcher timeout.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import resource
import socket
import tempfile
import threading
import time
import numpy as np
from ..errors import KuboMSDError
from ..parallel import ParallelContext
from .builder import (BuilderOptions,pair_tasks,resource_estimate,get_tables,thread_context,
                      assemble_result)
from .partition import partition_chunks,check_coverage
from .progress import BuildReport,environment_info
from .tables import native_module


def _test_hook(phase,rank,chunk=None):
    """Fault-injection seam; no environment-triggered production failures."""


def digest_array(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def broadcast_tree(ctx,value,max_bytes):
    """Only small metadata is pickled; numerical buffers use bounded MPI Bcast."""
    arrays=[]
    def encode(v):
        if isinstance(v,np.ndarray):
            v=np.ascontiguousarray(v);index=len(arrays);arrays.append(v)
            return {'__buffer__':index,'shape':v.shape,'dtype':v.dtype.str,'sha256':digest_array(v)}
        if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [encode(x) for x in v]
        return v
    metadata=ctx.root_call('serialize broadcast metadata',lambda:encode(value))
    def decode(v):
        if isinstance(v,dict) and '__buffer__' in v:
            i=v['__buffer__']
            a=ctx.stage('allocate broadcast buffer',lambda:arrays[i] if ctx.root else np.empty(v['shape'],np.dtype(v['dtype'])))
            raw=a.view(np.uint8).reshape(-1)
            for start in range(0,len(raw),min(max_bytes,2**30)):
                ctx.comm.Bcast(raw[start:start+min(max_bytes,2**30)],root=0)
            def verify():
                if digest_array(a)!=v['sha256']:raise KuboMSDError('MPI table/input buffer hash mismatch')
            ctx.stage('verify broadcast buffer',verify)
            return a
        if isinstance(v,dict):return {k:decode(x) for k,x in v.items()}
        if isinstance(v,list):return [decode(x) for x in v]
        return v
    return decode(metadata)


def build_operators_mpi(structure,orbitals,h_source,options,comm):
    from mpi4py import MPI
    ctx=ParallelContext(comm,MPI)
    def environment():
        if threading.current_thread() is not threading.main_thread() or not MPI.Is_thread_main() or MPI.Query_thread()<MPI.THREAD_FUNNELED:
            raise KuboMSDError('MPI construction requires MPI_THREAD_FUNNELED on its initializing main thread')
        if options is None:return
        options.__post_init__()
        if options.build_distribution!='mpi' or options.operator_builder=='reference':raise KuboMSDError('MPI build supports cpp/cpp-omp only')
    ctx.stage('environment',environment)
    cfg=ctx.root_call('root options',lambda:asdict(options))
    def compare_options():
        if options is not None and asdict(options)!=cfg:raise KuboMSDError('MPI build options disagree')
    ctx.stage('configuration consistency',compare_options)
    options=BuilderOptions(**cfg)
    root_progress=Path(options.progress_path) if options.progress_path else Path.cwd()/'operator-build-progress.json'
    report=BuildReport(options.operator_builder,root_progress if ctx.root else root_progress.with_name(root_progress.name+f'.rank-{ctx.rank}'))
    def resource_info():
        return dict(host=socket.gethostname(),affinity=sorted(os.sched_getaffinity(0)),threads=options.threads,rank=ctx.rank)
    resources=ctx.allgather(resource_info())
    def resource_check():
        for host in set(r['host'] for r in resources):
            rows=[r for r in resources if r['host']==host]
            cores=set(c for r in rows for c in r['affinity'])
            if sum(r['threads'] for r in rows)>len(cores):raise KuboMSDError('MPI ranks times threads exceed node CPU affinity')
            limit=os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
            if os.environ.get('SLURM_MEM_PER_NODE'):limit=min(limit,int(os.environ['SLURM_MEM_PER_NODE'])*2**20)
            if len(rows)*options.memory_limit_bytes>limit:raise KuboMSDError('MPI per-node total memory budgets exceed allocation')
    ctx.stage('node resources',resource_check)
    try:
        with thread_context():
            report.details.update(environment=environment_info(),tolerance=options.tolerance,
                                  threads=options.threads,physical_operator_validation='UNVERIFIED')
            def root_input():
                _test_hook('input',ctx.rank)
                if structure is None or orbitals is None:raise KuboMSDError('missing root structure/orbitals')
                h=h_source() if callable(h_source) else h_source
                if not isinstance(h,dict):raise KuboMSDError('root Hamiltonian source must produce image blocks')
                keys,d,species=pair_tasks(structure,orbitals,options.memory_limit_bytes)
                counts=np.array([orbitals[s].n_orbitals for s in structure['symbols']],np.int64)
                estimated=resource_estimate(keys,counts,options)
                return keys,d,species,counts,h,estimated
            start=time.perf_counter();inputs=ctx.stage('root input and neighbor list',root_input,root_only=True)
            report.timings['neighbor_list']=time.perf_counter()-start
            if ctx.root:keys,d,species,counts,h,estimated=inputs
            else:keys=d=species=counts=h=None
            start=time.perf_counter()
            numerical=broadcast_tree(ctx,dict(keys=keys,displacement_A=d,species=species,counts=counts) if ctx.root else None,options.mpi_io_batch_bytes)
            keys=numerical['keys'];d=numerical['displacement_A'];species=numerical['species'];counts=numerical['counts']
            costs=counts[keys[:,0]]*counts[keys[:,1]]
            chunks=ctx.broadcast(partition_chunks(costs,options.pair_chunk_size,ctx.size) if ctx.root else None)
            report.timings['mpi_distribution']=time.perf_counter()-start
            table=ctx.stage('root tables',lambda:get_tables(orbitals,options,report),root_only=True)
            start=time.perf_counter();state=ctx.stage('export tables',lambda:table.state(),root_only=True)
            state=broadcast_tree(ctx,state,options.mpi_io_batch_bytes)
            table=ctx.stage('restore identical tables',lambda:table if ctx.root else native_module().restore_tables(state))
            del state
            report.timings['table_broadcast']=time.perf_counter()-start
            def directory():
                root_progress.parent.mkdir(parents=True,exist_ok=True)
                p=Path(tempfile.mkdtemp(prefix='build-',dir=root_progress.parent))
                (p/'task_manifest.json').write_text(json.dumps(dict(chunks=chunks,keys=keys.tolist(),world_size=ctx.size,resources=resources),indent=2))
                return str(p)
            work=Path(ctx.root_call('create shared shard directory',directory))
            local=work/'shards'/f'rank-{ctx.rank:04d}'
            ctx.stage('create rank directory',lambda:local.mkdir(parents=True,exist_ok=False))
            owned=[c for c in chunks if c['rank']==ctx.rank];maxround=comm.allreduce(len(owned),op=MPI.MAX)
            records=[];ids=[];compute=0.;write=0.;wait=0.;pair_started=time.perf_counter()
            for step in range(maxround):
                def evaluate():
                    nonlocal compute,write
                    if step>=len(owned):return
                    c=owned[step];_test_hook('compute',ctx.rank,c['id']);a,b=c['start'],c['stop'];t=time.perf_counter()
                    result=native_module().evaluate_pairs(table,dict(species=species[a:b],displacement_A=d[a:b],pair_ids=np.arange(a,b,dtype=np.int64)),options.threads)
                    compute+=time.perf_counter()-t;t=time.perf_counter();p=local/f"chunk-{c['id']:08d}.npz"
                    np.savez(p,**result)
                    records.append(dict(chunk=c['id'],path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
                    ids.extend(range(a,b));write+=time.perf_counter()-t
                t=time.perf_counter();ctx.stage(f'pair batch {step}',evaluate);wait+=time.perf_counter()-t
                completed=comm.allreduce(len(ids),op=MPI.SUM)
                ctx.stage('progress',lambda:report.update(completed_pairs=completed,total_pairs=len(keys)))
            pair_wall=time.perf_counter()-pair_started
            report.timings.update(pair_evaluation=pair_wall,rank_compute=compute,
                                  shard_write=write,communication_wait=max(0.,pair_wall-compute-write))
            local_info=dict(rank=ctx.rank,pair_ids=ids,chunks=[c['id'] for c in owned],compute_s=compute,shard_write_s=write,compute_and_wait_s=wait,communication_wait_s=max(0.,pair_wall-compute-write),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            rank_reports=ctx.allgather(local_info);records=[r for rank in ctx.allgather(records) for r in rank]
            ctx.stage('post-shard validation hook',lambda:_test_hook('after_shards',ctx.rank))
            def merge():
                _test_hook('merge',ctx.rank)
                offsets=np.r_[0,np.cumsum(costs)];s=np.empty(offsets[-1]);D=np.empty((3,offsets[-1]));covered=[]
                if sorted(r['chunk'] for r in records)!=list(range(len(chunks))):raise KuboMSDError('chunk coverage mismatch')
                for record in sorted(records,key=lambda r:r['chunk']):
                    c=chunks[record['chunk']];p=Path(record['path'])
                    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise KuboMSDError('missing or corrupted shard')
                    with np.load(p,allow_pickle=False) as f:
                        expected=np.arange(c['start'],c['stop'],dtype=np.int64)
                        if not np.array_equal(f['pair_ids'],expected):raise KuboMSDError('shard pair IDs duplicate/missing')
                        start,end=offsets[c['start']],offsets[c['stop']]
                        if not np.array_equal(f['offsets'],offsets[c['start']:c['stop']+1]-start):raise KuboMSDError('shard offset mismatch')
                        if f['S'].shape!=(end-start,) or f['D'].shape!=(3,end-start):raise KuboMSDError('shard shape mismatch')
                        s[start:end]=f['S'];D[:,start:end]=f['D'];covered.extend(f['pair_ids'].tolist())
                check_coverage(covered,len(keys))
                result=assemble_result(structure,orbitals,h,keys,d,dict(offsets=offsets,S=s,D=D),options,report)
                report.details['mpi']=dict(world_size=ctx.size,ranks=rank_reports,library=MPI.Get_library_version(),thread_level=MPI.Query_thread(),resources=resources,shared_directory=str(work),coverage='exactly once',table_built_by_rank=0)
                report.details['mpi']['rank_summary']={k:dict(min=float(np.min([r[k] for r in rank_reports])),max=float(np.max([r[k] for r in rank_reports])),mean=float(np.mean([r[k] for r in rank_reports]))) for k in ('compute_s','shard_write_s','communication_wait_s','peak_rss_bytes')}
                report.details['estimated_assembly_bytes']=estimated
                return result
            t=time.perf_counter();result=ctx.stage('deterministic bounded merge',merge,root_only=True);report.timings['merge']=time.perf_counter()-t
            # Keep shards as an audit trail. Cleanup/publication callbacks still have
            # a coordinated boundary; no worker returns before root is successful.
            ctx.stage('cleanup boundary',lambda:_test_hook('cleanup',ctx.rank))
            ctx.stage('finish reports',report.finish)
            if ctx.root:result.diagnostics=report.snapshot()
            return result
    except Exception as exc:
        report.status='FAIL';report.error=str(exc)
        # Reporting failures must also be coordinated, never strand other ranks.
        ctx.stage('failure report',report.update)
        raise
