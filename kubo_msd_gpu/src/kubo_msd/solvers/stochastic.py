"""Sample-parallel transport; normalize global paired block sums only on root."""
from dataclasses import replace
import tempfile
import shutil
import time
import socket
from pathlib import Path
import numpy as np
from ..contracts import TransportBlock, Reason
from ..errors import NumericalError
from ..parallel import context_for, owned_blocks
from ..metric.bounds import estimate_spectral_bounds
from ..metric.sampling import sample_metric
from ..kernels.energy import dos_floor, kernel_resolution
from ..observables.diffusion import msd_gradient
from ..observables.statistics import estimate_ratio
from .stepwise import stepwise_samples
from .spectral import spectral_samples


def _quantity_keys(cfg, method):
    keys = ["msd_A2/commutator"] if method == "stepwise" else []
    if method == "spectral" or cfg.include_vac:
        keys += ["msd_A2/vac_integral", "diffusion_A2_per_fs/vac_integral"]
        if cfg.include_vac:
            keys += ["vac_A2_per_fs2/vac_integral"]
    if len(cfg.time_fs) > 1:
        keys += ["diffusion_A2_per_fs/msd_gradient"]
    return keys


def _sample_batches(ops, cfg, method, samples, block_ids, accum, dos_blocks):
    started = time.perf_counter()
    backend = ops.backend
    diagnostics, ids_used = [], []
    primary = "commutator" if method == "stepwise" else "vac_integral"
    ns = cfg.nsamples if samples is None else samples.shape[1]
    for local_block, global_block in enumerate(block_ids):
        first_id = global_block*cfg.statistics_block
        end_id = min(ns, first_id + cfg.statistics_block)
        for first in range(first_id, end_id, cfg.sample_batch):
            last = min(first + cfg.sample_batch, end_id)
            ids = np.arange(first, last)
            B = (sample_metric(cfg.seed, ids, ops, cfg) if samples is None
                 else backend.xp.asarray(samples[:, first:last]))
            if method == "stepwise":
                dos, nums, diag = stepwise_samples(ops, cfg, B)
                if cfg.include_vac:
                    _, extra, extra_diag = spectral_samples(ops, cfg, B)
                    nums.update(extra)
                    diag["additional_vac_path"] = extra_diag
                    diag["additional_cost_recorded"] = True
            else:
                dos, nums, diag = spectral_samples(ops, cfg, B)
            if len(cfg.time_fs) > 1:
                nums["diffusion_A2_per_fs/msd_gradient"] = msd_gradient(
                    cfg.time_fs, nums["msd_A2/" + primary], axis=2, xp=ops.xp)
            diagnostics.append(diag)
            ids_used.extend(ids.tolist())
            dh = backend.to_host(dos)
            if not np.isfinite(dh).all():
                raise NumericalError("nonfinite stochastic DOS")
            for local in range(len(ids)):
                dos_blocks[local_block] += dh[local]
            for key, values in nums.items():
                vh = backend.to_host(values)
                if not np.isfinite(vh).all():
                    raise NumericalError("nonfinite stochastic numerator")
                for local in range(len(ids)):
                    accum[key][local_block] += vh[local]
    return diagnostics, ids_used, time.perf_counter() - started


def _normalize_dos(ops, cfg, ns, counts, dos_blocks, report, method,
                   diagnostics, diskbytes, parallel_metadata):
    nbk = len(counts)
    primary = "commutator" if method == "stepwise" else "vac_integral"
    rho, stderr_dos, stat = estimate_ratio(dos_blocks, counts[:, None], counts)
    threshold = dos_floor(ops.n, cfg.spectral_bounds[1] - cfg.spectral_bounds[0], cfg)
    valid = np.isfinite(rho) & (rho > threshold)
    reason = np.where(valid, 0, int(Reason.LOW_DOS)).astype("uint32")
    if stderr_dos is not None:
        uncertain = ~np.isfinite(stderr_dos) | (rho - 1.96*stderr_dos <= 0)
        valid &= ~uncertain
        reason[uncertain] |= int(Reason.UNCERTAIN_TRACE)
    if nbk < 8:
        reason |= int(Reason.INSUFFICIENT_BLOCKS)
    static = {"dos_states_per_eV": rho, "dos_block_sums": dos_blocks,
              "block_counts": counts, "sample_count": np.array(ns), "sample_ids": np.arange(ns)}
    if stderr_dos is not None:
        static["stderr_dos_states_per_eV"] = stderr_dos
    metadata = {
        "primary_estimator": primary,
        "primary_diffusion_estimator": "msd_gradient" if method == "stepwise" else "vac_integral",
        "resolved_config": cfg.as_dict(), "spectral_report": report,
        "kernel_resolution": kernel_resolution(cfg), "dos_floor": threshold,
        "statistical_method": stat,
        "sample_generator": "NumPy PCG64 SeedSequence([seed,sample_id]); unit modulus complex phases",
        "sampling": getattr(getattr(ops, "sampler", None), "report",
                            {"source": "caller supplied coefficients"}),
        "sample_batch_diagnostics": diagnostics,
        "raw_accumulation": "numerator_* are means over nsamples; *_block_sums are sums",
        "scratch_bytes": diskbytes, "parallel": parallel_metadata,
        "science_status": {
            "NUMERICS_VERIFIED": "UNVERIFIED for this run",
            "STATISTICS_VERIFIED": "NOT RUN (no multi-seed convergence study for this run)",
            "TRUNCATION_VERIFIED": "not applicable"}}
    return valid, reason, static, metadata


def _normalize_slab(raw, dos_blocks, counts, ns, valid, key):
    value, se, _ = estimate_ratio(raw, dos_blocks, counts)
    value[..., ~valid] = np.nan
    arrays = {key: value, "numerator_" + key: raw.sum(0)/ns,
              "numerator_block_sums/" + key: raw.copy()}
    if se is not None:
        se[..., ~valid] = np.nan
        arrays["stderr_" + key] = se
    return arrays


def iter_stochastic_blocks(ops, config, method, samples=None, parallel=None):
    if hasattr(ops,'context'):
        from .stochastic_gpu import iter_stochastic_blocks_gpu
        yield from iter_stochastic_blocks_gpu(ops,config,method,samples)
        return
    ctx = parallel or context_for(config)
    cfg = config
    report = ctx.root_call("spectral bounds", lambda: estimate_spectral_bounds(ops, cfg))
    cfg = replace(cfg, spectral_bounds=report["bounds"])
    ns = cfg.nsamples if samples is None else samples.shape[1]
    nbk = (ns + cfg.statistics_block - 1)//cfg.statistics_block
    block_ids = owned_blocks(ns, cfg.statistics_block, ctx.rank, ctx.size)
    counts = np.array([min(cfg.statistics_block, ns-i*cfg.statistics_block) for i in range(nbk)])
    keys = _quantity_keys(cfg, method)
    shape = (len(block_ids), len(cfg.axes), len(cfg.time_fs), len(cfg.energy_eV))
    diskbytes = int(np.prod(shape))*8*len(keys)
    scratch = Path(tempfile.gettempdir())
    scratch_declarations = ctx.stage(
        "inspect scratch", lambda: (socket.gethostname(), shutil.disk_usage(scratch).free, diskbytes))
    scratch_declarations = ctx.allgather(scratch_declarations)

    def check_budget():
        # Conservative also for a shared scratch filesystem.
        if sum(d[2] for d in scratch_declarations) > min(d[1] for d in scratch_declarations)*.8:
            raise NumericalError("insufficient scratch space for paired statistical block sums")
        working = (min(cfg.sample_batch, cfg.statistics_block)*len(cfg.axes)*
                   len(cfg.time_fs)*len(cfg.energy_eV)*8*len(keys)*2)
        slab = nbk*len(cfg.axes)*min(cfg.time_block, len(cfg.time_fs))*len(cfg.energy_eV)*8
        root_working = slab*(len(keys) + 6) + nbk*len(cfg.energy_eV)*16
        # estimate_ratio creates resampled numerators, ratios and std scratch.
        # The resample dimension (256) is independent of the number of blocks.
        if nbk > 1:
            bootstrap = (4*256*len(cfg.axes)*min(cfg.time_block,len(cfg.time_fs))*
                         len(cfg.energy_eV)*8 + 2*256*nbk*8 + 2*256*len(cfg.energy_eV)*8)
            root_working += bootstrap
        if max(working, root_working if ctx.root else 0) > cfg.memory_limit_gb*1024**3:
            raise NumericalError("sample/merge/bootstrap workspace exceeds memory budget; reduce "
                                 "sample_batch/time_block/energy count or raise explicit budget")
    ctx.stage("transport workspace budget", check_budget)
    temporary = ctx.stage("create rank scratch", lambda: tempfile.TemporaryDirectory(prefix="kubomsd-"))
    cleaned = False
    accum = None
    def cleanup():
        if accum is not None:
            for value in accum.values():
                if isinstance(value, np.memmap) and not value._mmap.closed:
                    value._mmap.close()
        temporary.cleanup()
    try:
        def allocate():
            accum = {}
            for key in keys:
                if not block_ids:
                    accum[key] = np.zeros(shape)
                else:
                    name = Path(temporary.name)/(key.replace("/", "__") + ".npy")
                    accum[key] = np.lib.format.open_memmap(name, mode="w+", dtype=float, shape=shape)
                    accum[key][:] = 0
            return accum, np.zeros((len(block_ids), len(cfg.energy_eV)))
        accum, local_dos = ctx.stage("allocate rank accumulators", allocate)
        t0 = time.perf_counter()
        diagnostics, ids_used, sample_seconds = ctx.stage(
            "random sample transport",
            lambda: _sample_batches(ops, cfg, method, samples, block_ids, accum, local_dos))
        rank_record = ctx.stage("collect sample diagnostics", lambda:{
            "rank": ctx.rank, "host": socket.gethostname(), "block_ids": block_ids,
            "sample_ids": ids_used, "sample_seconds": sample_seconds,
            "sample_stage_seconds_including_wait": time.perf_counter() - t0,
            "metric": ops.metric.diagnostics(), "scratch_bytes": diskbytes})
        rank_records = ctx.allgather(rank_record)
        rank_diagnostics = ctx.allgather(diagnostics)
        dos_blocks = ctx.gather_blocks(local_dos, block_ids, nbk)
        parallel_metadata = {**ctx.metadata(), "ranks": rank_records}
        normalized = ctx.stage("normalize global DOS", lambda: _normalize_dos(
            ops, cfg, ns, counts, dos_blocks, report, method,
            [d for group in rank_diagnostics for d in group],
            sum(r["scratch_bytes"] for r in rank_records), parallel_metadata), root_only=True)
        if ctx.root:
            valid, reason, static, metadata = normalized
        for start in range(0, len(cfg.time_fs), cfg.time_block):
            stop = min(start + cfg.time_block, len(cfg.time_fs))
            arrays = {}
            for key in keys:
                raw = ctx.gather_blocks(accum[key][:, :, start:stop, :], block_ids, nbk)
                part = ctx.stage("normalize global numerator", lambda: _normalize_slab(
                    raw, dos_blocks, counts, ns, valid, key), root_only=True)
                if ctx.root:
                    arrays.update(part)

            def make_block():
                for estimator in {k.split("/")[-1] for k in keys}:
                    r = reason.copy()
                    if estimator == "vac_integral":
                        r |= int(Reason.INTEGRATION_UNVERIFIED)
                    dims = (len(cfg.axes), stop-start, len(cfg.energy_eV))
                    arrays["valid/" + estimator] = np.broadcast_to(valid, dims).copy()
                    arrays["reason_code/" + estimator] = np.broadcast_to(r, dims).copy()
                return TransportBlock(start, cfg.time_fs[start:stop], arrays,
                                      static if start == 0 else {}, metadata if start == 0 else {})
            # Every rank yields here so the caller can agree on write success.
            yield ctx.stage("form output block", make_block, root_only=True)
        ctx.stage("release rank scratch", cleanup)
        cleaned = True
    finally:
        # Only successful exhaustion performs a collective cleanup. If a stage
        # failed or the consumer closed this generator, preserve that exception.
        if not cleaned:
            try:
                cleanup()
            except Exception:
                pass
