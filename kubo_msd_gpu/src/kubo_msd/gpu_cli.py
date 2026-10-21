"""GPU-only public entry. CPU reference modules are not production fallbacks."""
import argparse
import json
import os
from pathlib import Path
import sys


def validate_scope(device, backend='gpu', distribution='serial'):
    if type(device) is not int or device < 0:
        raise ValueError('exactly one nonnegative integer device required')
    if backend != 'gpu' or distribution != 'serial':
        raise ValueError('this product requires backend=gpu and serial distribution')
    for name in ('OMPI_COMM_WORLD_SIZE','PMI_SIZE','PMIX_SIZE','MV2_COMM_WORLD_SIZE','WORLD_SIZE'):
        if int(os.environ.get(name, '1')) > 1:
            raise ValueError(f'single process required; active launcher {name} > 1')
    if (int(os.environ.get('SLURM_STEP_NUM_TASKS', '1')) > 1
            and ('SLURM_PROCID' in os.environ or 'SLURM_STEP_ID' in os.environ)):
        raise ValueError('single process required; active Slurm step has multiple tasks')


def parser():
    from .cli import parser as original_parser
    p = original_parser()
    p.prog = 'kubomsd-gpu'
    sub = next(a for a in p._actions if isinstance(a,argparse._SubParsersAction))
    run = sub.choices['run']
    run.set_defaults(backend='gpu', cpu_kernel='reference')
    for cmd in (run, sub.choices['capabilities']):
        cmd.add_argument('--device', type=int, default=0)
        cmd.add_argument('--device-memory-limit-gb', type=float)
    run.add_argument('--cache-out', type=Path)
    prep = sub.add_parser('prepare')
    prep.add_argument('--input', type=Path, required=True)
    prep.add_argument('--out', type=Path, required=True)
    prep.add_argument('--mode', choices=['audit','pilot','build'], default='audit')
    prep.add_argument('--device', type=int, default=0)
    prep.add_argument('--device-memory-limit-gb', type=float)
    for cmd in (run, prep):
        cmd.add_argument('--operator-builder', choices=['gpu'], default='gpu')
        cmd.add_argument('--integral-tol', type=float, default=1e-5)
        cmd.add_argument('--build-pair-batch', type=int, default=128)
        cmd.add_argument('--table-cache', type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate_scope(args.device, getattr(args,'backend','gpu'),
                       getattr(args,'transport_distribution','serial'))
        limit = args.device_memory_limit_gb
        if limit is not None and (not __import__('math').isfinite(limit) or limit <= 0):
            raise ValueError('device-memory-limit-gb must be finite positive')
        from .backends.gpu_context import create_gpu_context
        if args.command == 'capabilities':
            ctx = create_gpu_context(args.device, None if limit is None else int(limit*1024**3))
            print(json.dumps(ctx.report(), indent=2))
            return 0
        from .gpu_dispatch import execute
        return execute(args)
    except Exception as exc:
        print(f'kubomsd-gpu: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
