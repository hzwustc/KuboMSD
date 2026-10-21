"""Command-line controls map directly to immutable RunConfig fields."""
import argparse
import json
from pathlib import Path
import sys
from . import __version__
from .config import RunConfig,resolve_output_grid,resolve_time_grid
from .dispatch import run_input

def parser():
    p=argparse.ArgumentParser(prog='kubomsd',description='Single-Gamma Kubo MSD transport; energies are absolute eV.')
    p.add_argument('--version',action='version',version='KuboMSD '+__version__); sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run',help='compute transport from model.json or scalar legacy LCAO input.json')
    run.add_argument('--input',required=True,type=Path); run.add_argument('--out',required=True,type=Path)
    run.add_argument('--solver',choices=['exact','stochastic-kpm'],default='exact'); run.add_argument('--backend',choices=['cpu','hybrid','gpu'],default='cpu'); run.add_argument('--kernel',choices=['gaussian','jackson'],default='gaussian')
    run.add_argument('--stochastic-method',choices=['auto','stepwise','spectral'],default='auto')
    run.add_argument('--cpu-kernel',choices=['cpp','reference'],default='reference',help='Development CPU reference; public product entry is kubomsd-gpu')
    run.add_argument('--omp-threads',type=int,default=1,help='threads in the native CPU stochastic kernel per MPI rank')
    run.add_argument('--transport-distribution',choices=['serial','mpi'],default='serial',help='MPI partitions global stochastic statistics blocks; requires a completed operator cache')
    run.add_argument('--emin',type=float); run.add_argument('--emax',type=float); run.add_argument('--n-energy',type=int,default=201); run.add_argument('--energy-grid',type=Path,help='JSON array of absolute energies; excludes --emin/--emax')
    run.add_argument('--time-grid',type=Path,help='JSON array starting at 0; excludes --tmax-fs/--dt-fs'); run.add_argument('--tmax-fs',type=float); run.add_argument('--dt-fs',type=float)
    run.add_argument('--sigma-ev',type=float,default=.05); run.add_argument('--n-moments',type=int,default=256); run.add_argument('--spectral-min',type=float); run.add_argument('--spectral-max',type=float)
    run.add_argument('--axes',default='x,y,z'); run.add_argument('--axes-vectors',type=Path,help='JSON orthonormal direction rows matching --axes names')
    run.add_argument('--initial-state-selection',choices=['all','window'],default='all'); run.add_argument('--buffer-ev',type=float,default=0.)
    for name,default in [('nsamples',64),('seed',20261019),('sample-batch',8),('statistics-block',8),('pair-block',128),('time-block',16),('energy-block',256),('metric-maxiter',1000),('sampling-max-order',4096),('propagation-max-order',4096)]: run.add_argument('--'+name,type=int,default=default)
    for name,default in [('metric-rtol',1e-10),('metric-atol',0.),('sampling-tol',1e-10),('propagation-tol',1e-12),('segment-fs',1.),('integration-dt-fs',.01),('rho-abs',0.),('rho-rel',1e-12),('memory-limit-gb',4.)]: run.add_argument('--'+name,type=float,default=default)
    run.add_argument('--preconditioner',choices=['diagonal','block-jacobi'],default='diagonal'); run.add_argument('--no-vac',action='store_true',help='skip optional VAC output; stepwise then avoids extra spectral route')
    run.add_argument('--npz',action='store_true',help='also export small results to NPZ'); run.add_argument('--efermi-ev',type=float); run.add_argument('--efermi-source')
    sub.add_parser('capabilities',help='report actual CPU/GPU availability without transport')
    return p

def main(argv=None):
    p=parser(); args=p.parse_args(argv)
    if args.command=='capabilities':
        import importlib.util
        from .backends.gpu import GPUBackend
        from .backends.native import extension_info
        try: gpu=GPUBackend().memory_info(); status='available (numerical validation separate)'
        except ValueError as exc: gpu={'reason':str(exc)}; status='unavailable'
        try: native={'available':True,**extension_info()}
        except ValueError as exc: native={'available':False,'reason':str(exc)}
        print(json.dumps({'version':__version__,'cpu':'available','cpu_native':native,
                         'mpi4py_installed':importlib.util.find_spec('mpi4py') is not None,
                         'mpi_transport':'implemented for stochastic-kpm; runtime and scaling acceptance separate',
                         'distributed_exact':'not implemented','gpu_operator_builder':'not implemented',
                         'gpu':status,'gpu_details':gpu,'supported_combinations':['exact/cpu','exact/hybrid','exact/gpu','stochastic-kpm/cpu','stochastic-kpm/gpu']},indent=2)); return 0
    try:
        d=vars(args).copy(); d.pop('command'); inp=d.pop('input'); out=d.pop('out'); emin=d.pop('emin'); emax=d.pop('emax'); ne=d.pop('n_energy'); eg=d.pop('energy_grid'); tg=d.pop('time_grid'); tm=d.pop('tmax_fs'); dt=d.pop('dt_fs'); lo=d.pop('spectral_min'); hi=d.pop('spectral_max')
        if eg:
            if emin is not None or emax is not None: raise ValueError('--energy-grid excludes --emin/--emax')
            E=json.loads(eg.read_text())
        else:
            if emin is None or emax is None: raise ValueError('provide --emin and --emax, or --energy-grid')
            E=resolve_output_grid(emin,emax,ne)
        if tg:
            if tm is not None or dt is not None: raise ValueError('--time-grid excludes --tmax-fs/--dt-fs')
            T=json.loads(tg.read_text())
        else: T=resolve_time_grid(50. if tm is None else tm,.1 if dt is None else dt)
        if (lo is None)!=(hi is None): raise ValueError('both --spectral-min and --spectral-max required')
        d['spectral_bounds']=None if lo is None else (lo,hi); d['sigma_eV']=d.pop('sigma_ev'); d['buffer_eV']=d.pop('buffer_ev'); d['efermi_eV']=d.pop('efermi_ev'); d['output_npz']=d.pop('npz'); d['include_vac']=not d.pop('no_vac'); d['axes']=tuple(a.strip() for a in d['axes'].split(','))
        if d['axes_vectors'] is not None: d['axes_vectors']=json.loads(d['axes_vectors'].read_text())
        cfg=RunConfig(energy_eV=E,time_fs=T,**d); summary=run_input(inp,cfg,out)
        root=True
        if cfg.transport_distribution=='mpi':
            from mpi4py import MPI
            root=MPI.COMM_WORLD.Get_rank()==0
        if root: print(json.dumps({'run_status':summary.run_status,'method':summary.method,'backend':summary.backend,'output':summary.output_path,'scientific_convergence':'separate; see metadata.json'},indent=2))
        return 0
    except (ValueError,OSError,KeyError,TypeError,MemoryError) as exc:
        print(f'kubomsd: {type(exc).__name__}: {exc}',file=sys.stderr); return 2

if __name__=='__main__': raise SystemExit(main())
