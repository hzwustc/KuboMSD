"""Launched by mpiexec in integration tests; no fake communicator."""
import json
import sys
from pathlib import Path
import numpy as np
from kubo_msd.config import RunConfig
from kubo_msd.dispatch import run_input
from kubo_msd.parallel import context_for

model, out, mode = sys.argv[1:4]
cfg = RunConfig(
    np.linspace(-2, 2, 9), [0, .03, .07],
    solver="stochastic-kpm", kernel="jackson", axes=("x",),
    n_moments=16, nsamples=17, statistics_block=2, sample_batch=3,
    time_block=2, spectral_bounds=(-4., 4.), integration_dt_fs=.02,
    include_vac=True, stochastic_method="spectral" if "spectral" in mode else "stepwise",
    cpu_kernel="cpp" if mode.startswith("native") else "reference",
    omp_threads=2 if mode.startswith("native") else 1, transport_distribution="mpi")
from dataclasses import replace
if mode=="native-adaptive":
    cfg=replace(cfg,metric_residual_policy="adaptive")
if mode=="native-cholesky":
    cfg=replace(cfg,metric_solver="cholesky")
if mode=="native-ic0":
    cfg=replace(cfg,preconditioner="ic0")
if mode=="partition-failure":
    cfg=replace(cfg,preconditioner="cluster-jacobi",metric_partition_file=str(Path(out)/"missing-partition.json"))
ctx = context_for(cfg)
if mode == "config-mismatch":
    from dataclasses import replace
    cfg = replace(cfg, seed=cfg.seed+ctx.rank)
if mode == "input-mismatch" and ctx.rank == 1:
    model = model + ".absent"
if mode in ("gpu-collision", "gpu-single"):
    from dataclasses import replace
    cfg = replace(cfg, backend="gpu")
if mode == "empty":
    from dataclasses import replace
    cfg = replace(cfg, nsamples=3)
if mode == "numeric-failure":
    from dataclasses import replace
    cfg = replace(cfg, metric_maxiter=1)
if mode == "failure":
    try:
        ctx.stage("injected rank failure", lambda: 1/0 if ctx.rank == 1 else 7)
    except ValueError as exc:
        (Path(out)/f"failed-{ctx.rank}.txt").write_text(str(exc))
        raise SystemExit(23)
    raise SystemExit(0)
if mode == "cleanup-failure" and ctx.rank == 1:
    import tempfile
    original_temporary = tempfile.TemporaryDirectory
    def broken_cleanup_directory(*args, **kwargs):
        folder = original_temporary(*args, **kwargs)
        original_cleanup = folder.cleanup
        def cleanup():
            original_cleanup()
            raise OSError("injected rank scratch cleanup failure")
        folder.cleanup = cleanup
        return folder
    tempfile.TemporaryDirectory = broken_cleanup_directory
if mode == "append-failure" and ctx.root:
    from kubo_msd.io.results import ResultsWriter
    original_append = ResultsWriter.append
    def broken_append(self, block):
        original_append(self, block)
        raise OSError("injected root append failure")
    ResultsWriter.append = broken_append
run_input(model, cfg, out)
