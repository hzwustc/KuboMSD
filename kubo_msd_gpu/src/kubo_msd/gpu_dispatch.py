"""Public single-device prepare/run lifecycle."""
from dataclasses import fields
import json
from pathlib import Path
import time
from .backends.gpu_context import create_gpu_context
from .config import RunConfig, resolve_output_grid, resolve_time_grid
from .contracts import RunSummary
from .io.results import ResultsWriter, json_safe
from .operators.gpu.contracts import GpuBuildOptions
from .operators.gpu.input import parse_build_input
from .operators.gpu.plan import make_pair_plan
from .operators.gpu.builder import build_gpu_operators
from .operators.gpu.cache import load_operator_cache_gpu, save_operator_cache
from .operators.gpu.runtime import make_device_operators


def resolve_config(args):
    d=vars(args).copy()
    emin,emax=d.pop('emin'),d.pop('emax')
    eg,tg=d.pop('energy_grid'),d.pop('time_grid')
    ne,tm,dt=d.pop('n_energy'),d.pop('tmax_fs'),d.pop('dt_fs')
    if eg:
        if emin is not None or emax is not None: raise ValueError('--energy-grid excludes --emin/--emax')
        E=json.loads(eg.read_text())
    else:
        if emin is None or emax is None: raise ValueError('provide --emin/--emax or --energy-grid')
        E=resolve_output_grid(emin,emax,ne)
    if tg:
        if tm is not None or dt is not None: raise ValueError('--time-grid excludes --tmax-fs/--dt-fs')
        T=json.loads(tg.read_text())
    else: T=resolve_time_grid(50. if tm is None else tm,.1 if dt is None else dt)
    lo,hi=d.pop('spectral_min'),d.pop('spectral_max')
    if (lo is None)!=(hi is None): raise ValueError('both spectral bounds required')
    d['spectral_bounds']=None if lo is None else (lo,hi)
    d['sigma_eV']=d.pop('sigma_ev');d['buffer_eV']=d.pop('buffer_ev')
    d['efermi_eV']=d.pop('efermi_ev');d['output_npz']=d.pop('npz')
    d['include_vac']=not d.pop('no_vac')
    d['axes']=tuple(a.strip() for a in d['axes'].split(','))
    if d['axes_vectors'] is not None: d['axes_vectors']=json.loads(d['axes_vectors'].read_text())
    allowed={f.name for f in fields(RunConfig)}
    return RunConfig(energy_eV=E,time_fs=T,**{k:v for k,v in d.items() if k in allowed})


def _options(args):
    return GpuBuildOptions(integral_tolerance=args.integral_tol,pair_batch_size=args.build_pair_batch,
        table_cache_dir=args.table_cache,device_memory_limit_bytes=None if args.device_memory_limit_gb is None
        else int(args.device_memory_limit_gb*1024**3))


def execute(args):
    started=time.perf_counter()
    options=_options(args)
    if args.command=='prepare':
        parsed=parse_build_input(args.input)
        if args.mode=='audit':
            plan=make_pair_plan(parsed)
            report={'mode':'audit','complete_operator_cache':False,'n_orbitals':len(parsed.orbital_order),
                    'pairs':plan.report,'files':list(parsed.input_hashes),'gpu_numerical_work':'NOT RUN'}
        else:
            context=create_gpu_context(args.device,options.device_memory_limit_bytes)
            if args.mode=='build':
                result=build_gpu_operators(parsed,options,context)
                save_operator_cache(result.model,args.out,context)
                report={**result.report,'mode':'build','complete_operator_cache':True,'gpu_pipeline':context.report()}
                (args.out/'prepare_report.json').write_text(json.dumps(json_safe(report),indent=2)+'\n')
                print(json.dumps({'status':'CACHE_READY','output':str(args.out)}))
                return 0
            from .operators.gpu.tables import build_tables_gpu
            from .operators.gpu.pairs import evaluate_pairs_gpu
            plan=make_pair_plan(parsed);tables=build_tables_gpu(parsed,options,context)
            blocks=evaluate_pairs_gpu(tables,plan.batch(0,min(len(plan.pair_ids),16)),context)
            report={'mode':'pilot','complete_operator_cache':False,'pilot_pairs':len(blocks.batch.pair_ids),
                    'table':tables.metadata,'gpu_pipeline':context.report()}
        args.out.mkdir(parents=True,exist_ok=False)
        (args.out/'prepare_report.json').write_text(json.dumps(json_safe(report),indent=2)+'\n')
        print(json.dumps({'status':args.mode.upper(),'output':str(args.out)}))
        return 0
    cfg=resolve_config(args)
    limit=min(int(cfg.memory_limit_gb*1024**3),options.device_memory_limit_bytes or int(cfg.memory_limit_gb*1024**3))
    context=create_gpu_context(args.device,limit)
    with context.activate():
        path=args.input/'model.json' if args.input.is_dir() else args.input
        with context.phase('read_inputs'):declaration=json.loads(path.read_text())
        if declaration.get('format')=='kubomsd-model-v1':
            model=load_operator_cache_gpu(path,context);build_report={'builder_used':'cache'}
        else:
            with context.phase('read_inputs'):parsed=parse_build_input(path)
            result=build_gpu_operators(parsed,options,context)
            model,build_report=result.model,result.report
        if args.cache_out:save_operator_cache(model,args.cache_out,context)
        ops=make_device_operators(model,cfg,context)
        if cfg.solver=='exact':
            from .solvers.exact import iter_exact_blocks
            method='exact';blocks=iter_exact_blocks(model,ops,cfg)
            decision={'method':'exact','reason':'explicit solver'}
        else:
            from .dispatch import choose_stochastic_method
            from .solvers.stochastic import iter_stochastic_blocks
            decision=choose_stochastic_method(cfg,{'n_orbitals':model.n_orbitals,'nnz':model.nnz},None)
            method=decision['method'];blocks=iter_stochastic_blocks(ops,cfg,method)
        with ResultsWriter(args.out,cfg) as sink:
            sink.metadata.update({'builder_used':build_report['builder_used'],'build_report':build_report,
                'provenance':dict(model.provenance),'cache_identity':model.cache_identity,
                'method_decision':decision,'backend_requested':'gpu','backend_used':'gpu'})
            try:
                with context.phase('transport_and_block_output'):
                    for block in blocks:
                        with context.phase('write_results'):sink.append(block)
            finally:blocks.close()
            context.synchronize()
            sink.metadata['gpu_pipeline']=context.report()
            summary=RunSummary(str(args.out.resolve()),'complete',method,'gpu',
                {'metric':ops.metric.diagnostics(),'gpu':context.memory_info(),
                 'h_action_calls':ops.h_calls,'v_cov_calls':ops.v_calls})
            sink.metadata['performance_before_finalize_seconds']=time.perf_counter()-started
            sink.metadata['timing_boundary']='CLI benchmark measures full end-to-end; metadata timestamp precedes final formatting/commit'
            with context.phase('write_results'):sink.finalize(summary)
    print(json.dumps({'run_status':'complete','method':method,'backend':'gpu','output':str(args.out.resolve()),
                      'scientific_convergence':'separate; see metadata.json'}))
    return 0
