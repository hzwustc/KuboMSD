#!/usr/bin/env python3
"""Synthetic sparse CPU comparisons with identical inputs and accuracy controls."""
import os
# Both implementations use one BLAS thread; native OpenMP is set per RunConfig.
# Set before importing NumPy/SciPy, including in standalone subprocess execution.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import scipy.sparse as sp

from kubo_msd.backends.native import extension_info
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.dispatch import run_input
from kubo_msd.io.results import read_results
from kubo_msd.metric.sampling import MetricSampler, random_phases
from kubo_msd.operators.velocity import build_operators
from kubo_msd.solvers.chebyshev import chain, projected_quadratic
from kubo_msd.solvers.stepwise import propagate


def synthetic(n):
    """Banded Hermitian model with nonidentity SPD overlap and complex hopping."""
    rng = np.random.default_rng(605+n)
    h1 = .17*np.exp(1j*rng.uniform(-np.pi, np.pi, n-1))
    h3 = .05*np.exp(1j*rng.uniform(-np.pi, np.pi, n-3))
    h = sp.diags([h3.conj(), h1.conj(), rng.uniform(-.2, .2, n), h1, h3],
                 [-3, -1, 0, 1, 3], shape=(n, n), format="csr")
    off = .055*np.exp(1j*rng.uniform(-np.pi, np.pi, n-1))
    s = sp.diags([off.conj(), 1.1+.1*rng.random(n), off],
                 [-1, 0, 1], shape=(n, n), format="csr")
    v = sp.diags([h1.conj()*1j, rng.uniform(-.05, .05, n), -1j*h1],
                 [-1, 0, 1], shape=(n, n), format="csr")
    return ModelBundle(h, s, velocity={"x": v}, atom_slices=[
        (i, min(i+4, n)) for i in range(0, n, 4)])


def save_model(model, target):
    target.mkdir()
    def entry(name, matrix):
        path = target/(name+".npz")
        sp.save_npz(path, matrix.astype(np.complex128), compressed=False)
        return dict(path=path.name, shape=list(matrix.shape), dtype="complex128",
                    storage="csr", index_base=0, representation="covariant")
    doc = dict(format="kubomsd-model-v1", n_orbitals=model.n_orbitals,
               basis="nonorthogonal", kpoint=[0, 0, 0], energy_unit="eV",
               length_unit="angstrom", orbital_order=list(range(model.n_orbitals)),
               hamiltonian=entry("H", model.H_cov), overlap=entry("S", model.S),
               atom_slices=model.atom_slices,
               velocity_source=dict(type="matrix", unit="angstrom/fs",
                                    matrices={"x": entry("Vx", model.velocity["x"])}))
    path = target/"model.json"
    path.write_text(json.dumps(doc, indent=2)+"\n")
    return path


def measured(function, repeats):
    function()  # Separate warmup; elapsed measures only requested operations.
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        seconds.append(time.perf_counter()-start)
    return value, float(np.median(seconds)), seconds


def max_difference(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.array_equal(
            np.isfinite(actual), np.isfinite(expected)):
        raise AssertionError("kernel/reference shape or finite masks differ")
    finite = np.isfinite(expected)
    return float(np.max(np.abs(actual[finite]-expected[finite]), initial=0))


def case(n, args):
    target = args.out/f"n{n}"
    target.mkdir()
    model = synthetic(n)
    path = save_model(model, target/"input")
    cfg = RunConfig(
        energy_eV=np.linspace(-.45, .45, 9), time_fs=[0, .1, .2, .3],
        axes=("x",), solver="stochastic-kpm", stochastic_method="stepwise",
        kernel="jackson", n_moments=args.moments, spectral_bounds=(-1, 1),
        nsamples=args.nsamples, sample_batch=4, statistics_block=2,
        include_vac=False, metric_rtol=1e-11, sampling_tol=1e-11,
        segment_fs=.15, omp_threads=args.threads)
    ops = {key: build_operators(model, replace(cfg, cpu_kernel=key))
           for key in ("reference", "cpp")}
    xi = random_phases(n, cfg.seed, range(min(4, args.nsamples)))
    functions = {}
    for key, op in ops.items():
        sampler = MetricSampler(op.metric, cfg)
        states = sampler.transform(xi)
        functions[key] = {
            "spmm": lambda op=op: op.apply_h_cov(xi),
            "pcg": lambda op=op: op.solve_s(xi)[0],
            "sampling": lambda s=sampler: s.transform(xi),
            "chain": lambda op=op, b=states: chain(b, op, cfg, cfg.n_moments),
            "projected": lambda op=op, b=states: projected_quadratic(b, op, cfg),
            "propagate": lambda op=op, b=states: propagate(b, .3, op, cfg)[0],
        }
    kernels = {}
    for name in functions["reference"]:
        values, timings, samples = {}, {}, {}
        for key in ("reference", "cpp"):
            values[key], timings[key], samples[key] = measured(
                functions[key][name], args.repeats)
        error = max_difference(values["cpp"], values["reference"])
        np.testing.assert_allclose(values["cpp"], values["reference"], rtol=2e-8, atol=1e-9)
        kernels[name] = {
            "reference_seconds": timings["reference"], "cpp_seconds": timings["cpp"],
            "speedup_reference_over_cpp": timings["reference"]/timings["cpp"],
            "max_absolute_difference": error, "timing_samples_seconds": samples}
    results, runs = {}, {}
    for key in ("reference", "cpp"):
        start = time.perf_counter()
        summary = run_input(path, replace(cfg, cpu_kernel=key), target/key)
        elapsed = time.perf_counter()-start
        results[key] = read_results(target/key)
        runs[key] = {"wall_seconds": elapsed, "diagnostics": summary.diagnostics}
    array_error = max(max_difference(results["cpp"]["arrays"][key], value)
                      for key, value in results["reference"]["arrays"].items()
                      if value.dtype.kind in "fc")
    for key, value in results["reference"]["arrays"].items():
        if value.dtype.kind in "fc":
            np.testing.assert_allclose(results["cpp"]["arrays"][key], value,
                                       rtol=2e-8, atol=1e-9, equal_nan=True)
    for key in ("dos_states_per_eV", "dos_block_sums", "block_counts", "sample_ids"):
        np.testing.assert_allclose(results["cpp"]["arrays"][key],
                                   results["reference"]["arrays"][key], rtol=2e-9, atol=1e-10)
    return dict(
        n_orbitals=n, nnz_H=model.H_cov.nnz, nnz_S=model.S.nnz,
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        resolved_config=cfg.as_dict(), kernels=kernels,
        end_to_end={**runs, "max_absolute_array_difference": array_error,
                    "speedup_reference_over_cpp": runs["reference"]["wall_seconds"]/runs["cpp"]["wall_seconds"]})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", default="32,512")
    parser.add_argument("--nsamples", type=int, default=8)
    parser.add_argument("--moments", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    sizes = [int(v) for v in args.sizes.split(",")]
    if min(sizes) < 4 or min(args.nsamples, args.repeats, args.threads) < 1:
        parser.error("sizes >=4; nsamples/repeats/threads >=1 required")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "synthetic sparse models only; material and server performance unverified",
        "timing_policy": "kernel median after warmup; end-to-end includes input, bounds, transport and HDF5 output; kernels run reference then cpp",
        "accuracy_policy": "identical complex128 model, seed/sample IDs, moments, times and solve/sampling tolerances",
        "native": extension_info(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform(),
                        "omp_threads": args.threads, "blas_threads": 1},
        "cases": []}
    for n in sizes:
        report["cases"].append(case(n, args))
        (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(args.out/"report.json")


if __name__ == "__main__":
    main()
