import os
import platform
import resource
import time
import json
import sys
from pathlib import Path
import numpy as np
import scipy
import h5py
from .contracts import RunSummary
from .io.model import load_model
from .io.results import ResultsWriter
from .operators.velocity import build_operators
from .parallel import context_for

def choose_stochastic_method(config,model_summary,calibration=None):
    if config.stochastic_method!='auto': return {'method':config.stochastic_method,'rule_version':'1.0','reason':'explicit user choice','model':model_summary,'estimated_cost':None}
    return {'method':'stepwise','rule_version':'1.0','reason':'no validated applicable hardware/precision calibration; conservative stepwise','model':model_summary,'estimated_cost':None,'calibration_used':False}

def run_transport(model,operators,config,sink,parallel=None):
    start=time.perf_counter(); ops=operators; cfg=config
    ctx=parallel or context_for(cfg)
    if cfg.solver=='exact':
        from .solvers.exact import iter_exact_blocks
        blocks=iter_exact_blocks(model,ops,cfg); method='exact'; decision={'method':'exact','reason':'explicit solver'}
    else:
        from .solvers.stochastic import iter_stochastic_blocks
        decision=choose_stochastic_method(cfg,{'n_orbitals':ops.n,'nnz':ops.host_H.nnz,'n_times':len(cfg.time_fs),'n_moments':cfg.n_moments},None)
        method=decision['method']; blocks=iter_stochastic_blocks(ops,cfg,method,parallel=ctx)
    ctx.stage('write environment metadata',lambda:sink.metadata.update({'provenance':model.provenance,'method_decision':decision,'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'h5py':h5py.__version__,'platform':platform.platform(),'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},'omp_threads_requested':cfg.omp_threads},'backend_requested':cfg.backend,'backend_used':cfg.backend}),root_only=True)
    try:
        for block in blocks:
            ctx.stage('append result block',lambda:sink.append(block),root_only=True)
    finally:
        blocks.close()
    ctx.stage('synchronize backend',ops.backend.synchronize)
    diagnostics=ctx.stage('collect runtime diagnostics',lambda:{
        'rank':ctx.rank,'transport_and_block_io_seconds':time.perf_counter()-start,
        'metric':ops.metric.diagnostics(),'h_action_calls':ops.h_calls,'v_cov_calls':ops.v_calls,
        'velocity_hermiticity':ops.velocity_hermiticity,
        'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'gpu':ops.backend.memory_info()})
    per_rank=ctx.allgather(diagnostics)
    summary=ctx.root_call('form run summary',lambda:RunSummary(
        str(sink.path),'complete',method,cfg.backend,
        {**diagnostics,'rank_diagnostics':per_rank,
         'max_rank_transport_seconds':max(r['transport_and_block_io_seconds'] for r in per_rank)}))
    ctx.stage('finalize results',lambda:sink.finalize(summary),root_only=True)
    return summary

def run_input(input_path,config,out):
    start=time.perf_counter()
    ctx=context_for(config)
    if ctx.comm is not None:
        return _run_input_mpi(input_path,config,out,ctx,start)
    with ResultsWriter(out,config) as sink:
        model=load_model(input_path,config.backend); loaded=time.perf_counter(); operators=build_operators(model,config); built=time.perf_counter()
        sink.metadata['initialization']={'input_and_integrals_seconds':loaded-start,'operators_and_metric_seconds':built-loaded}
        summary=run_transport(model,operators,config,sink,ctx)
        # Final measured elapsed includes final gradient/aliases/export and flush.
        sink.metadata['performance_end_to_end_seconds']=time.perf_counter()-start; sink._save_meta()
        return summary


def _run_input_mpi(input_path,config,out,ctx,start):
    def cache_required():
        d=json.loads(Path(input_path).read_text())
        if d.get('format')!='kubomsd-model-v1':
            raise ValueError('MPI transport consumes a completed model.json operator cache; '
                             'construct raw LCAO operators before launching sample-parallel transport')
    ctx.stage('check operator cache input',cache_required)
    if config.backend=='gpu':
        ctx.bind_gpu()
    model=ctx.stage('load operator cache',lambda:load_model(input_path,config.backend))
    loaded=time.perf_counter()
    signatures=ctx.allgather((model.n_orbitals,model.provenance))
    if any(s!=signatures[0] for s in signatures):
        raise ValueError('MPI ranks loaded different model inputs or provenance')
    operators=ctx.stage('initialize operators',lambda:build_operators(model,config))
    built=time.perf_counter()
    sink=ctx.stage('open root result file',lambda:ResultsWriter(out,config),root_only=True)
    try:
        ctx.stage('write initialization metadata',lambda:sink.metadata.update({
            'initialization':{'input_and_integrals_seconds':loaded-start,
                              'operators_and_metric_seconds':built-loaded}}),root_only=True)
        summary=run_transport(model,operators,config,sink,ctx)
        def end_metadata():
            sink.metadata['performance_end_to_end_seconds']=time.perf_counter()-start
            sink._save_meta()
        ctx.stage('write final elapsed time',end_metadata,root_only=True)
    except BaseException:
        if ctx.root and sink is not None:
            sink.__exit__(*sys.exc_info())
        raise
    else:
        ctx.stage('close result file',lambda:sink.__exit__(None,None,None),root_only=True)
        return summary
