#!/usr/bin/env python3
"""Deterministic exact-diagonalization MSD/D reference for RSATB inputs.

The numerical definitions are the parameterized forms of
``gen_reference_ops.py`` and ``make_ref_curves.py::ref_msd`` in this directory.
RSATB imports are intentionally deferred so ``--help`` and the pure numerical
tests do not require a compiled ``_rsatb`` extension.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy
import scipy.linalg as sla
import scipy.sparse as sp


HBAR_EV_FS = 0.6582119569
DENSITY_TOL = 1.0e-12
HERMITICITY_RTOL = 1.0e-8
RSATB_MSD_DEFAULTS = {"dt_fs": 10.0, "n_steps": 200, "n_kpm": 1000}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
GIB = float(1024**3)


@dataclass(frozen=True)
class RunParameters:
    dt_fs: float
    n_steps: int
    n_kpm: int
    sources: Dict[str, str]


def jackson_kernel(n_moments: int) -> np.ndarray:
    """Return the Jackson damping coefficients used by RSATB KPM."""
    if n_moments < 1:
        raise ValueError("n_kpm must be at least 1")
    denominator = n_moments + 1.0
    theta = np.pi / denominator
    indices = np.arange(1, n_moments, dtype=np.float64)
    result = np.empty(n_moments, dtype=np.float64)
    result[0] = 1.0
    if n_moments > 1:
        cot_theta = np.cos(theta) / np.sin(theta)
        result[1:] = (
            (denominator - indices) * np.cos(theta * indices)
            + cot_theta * np.sin(theta * indices)
        ) / denominator
    return result


def chebyshev_energy_grid(emin_eV: float, emax_eV: float, count: int) -> np.ndarray:
    """Return RSATB's ascending Chebyshev-node energy grid."""
    if count < 1:
        raise ValueError("energy grid count must be at least 1")
    if not np.isfinite([emin_eV, emax_eV]).all() or emax_eV <= emin_eV:
        raise ValueError("energy bounds must be finite and satisfy emin < emax")
    center = 0.5 * (emin_eV + emax_eV)
    half_width = 0.5 * (emax_eV - emin_eV)
    k = np.arange(count, dtype=np.float64)
    return center + half_width * np.cos(np.pi * (count - 1.0 - k + 0.5) / count)


def window_from_chebyshev_grid(energy_eV: np.ndarray) -> Tuple[float, float]:
    """Recover ``(center, half_width)`` from an ascending Chebyshev grid."""
    energy = np.asarray(energy_eV, dtype=np.float64)
    if energy.ndim != 1 or energy.size < 2:
        raise ValueError("a Chebyshev grid needs at least two one-dimensional points")
    if not np.isfinite(energy).all() or not np.all(np.diff(energy) > 0.0):
        raise ValueError("energy grid must be finite and strictly increasing")
    count = energy.size
    first_node = np.cos(np.pi * (count - 0.5) / count)
    last_node = np.cos(np.pi * 0.5 / count)
    half_width = (energy[-1] - energy[0]) / (last_node - first_node)
    center = energy[0] - half_width * first_node
    reconstructed = chebyshev_energy_grid(
        center - half_width, center + half_width, count
    )
    scale = max(1.0, float(np.max(np.abs(energy))))
    if not np.allclose(energy, reconstructed, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError("energy array is not the RSATB Chebyshev-node grid")
    return float(center), float(half_width)


def delta_matrix(
    energy_eV: np.ndarray,
    eigenvalues_eV: np.ndarray,
    center_eV: float,
    half_width_eV: float,
    n_kpm: int,
    moment_chunk: int = 256,
) -> np.ndarray:
    """Jackson-broadened ``delta_J(E-eps)`` in 1/eV.

    This is the old ``make_ref_curves.py::delta_matrix`` formula, including
    the endpoint clipping and energy-unit division by the spectral half-width.
    """
    energy = np.asarray(energy_eV, dtype=np.float64)
    eigenvalues = np.asarray(eigenvalues_eV, dtype=np.float64)
    if energy.ndim != 1 or eigenvalues.ndim != 1:
        raise ValueError("energy and eigenvalues must be one-dimensional")
    if not np.isfinite(energy).all() or not np.isfinite(eigenvalues).all():
        raise ValueError("energy and eigenvalues must be finite")
    if not np.isfinite(half_width_eV) or half_width_eV <= 0.0:
        raise ValueError("spectral half-width must be positive")
    if moment_chunk < 1:
        raise ValueError("moment_chunk must be positive")

    jackson = jackson_kernel(n_kpm)
    x = np.clip((energy - center_eV) / half_width_eV, -1.0 + 1e-12, 1.0 - 1e-12)
    y = np.clip((eigenvalues - center_eV) / half_width_eV, -1.0, 1.0)
    theta_x = np.arccos(x)[:, None]
    theta_y = np.arccos(y)[None, :]
    broadened = np.full((energy.size, eigenvalues.size), jackson[0], dtype=np.float64)
    for start in range(1, n_kpm, moment_chunk):
        moments = np.arange(start, min(start + moment_chunk, n_kpm))
        broadened += 2.0 * np.einsum(
            "m,km,mn->kn",
            jackson[moments],
            np.cos(theta_x * moments[None, :]),
            np.cos(theta_y.T * moments[None, :]).T,
            optimize=True,
        )
    broadened /= np.pi * np.sqrt(np.maximum(1e-14, 1.0 - x[:, None] ** 2))
    return broadened / half_width_eV


def velocity_weights(
    eigenvalues_eV: np.ndarray,
    eigenvectors: np.ndarray,
    bond: np.ndarray,
    xbar: np.ndarray,
    delta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the reference velocity matrix and ``abs(V)**2``."""
    eps = np.asarray(eigenvalues_eV, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.complex128)

    def rotate(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.complex128)
        return vectors.conj().T @ matrix @ vectors

    bond_t = rotate(bond)
    xbar_t = rotate(xbar)
    delta_t = rotate(delta)
    velocity = 1j * (
        bond_t - xbar_t * eps[None, :] + eps[:, None] * delta_t
    )
    velocity = 0.5 * (velocity + velocity.conj().T)
    return velocity, np.abs(velocity) ** 2


def solve_generalized_eigenproblem(
    hamiltonian: np.ndarray, overlap: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve ``H C = S C E`` without modifying or regularizing ``S``."""
    hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
    overlap = np.asarray(overlap, dtype=np.complex128)
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("Hamiltonian must be a square matrix")
    if overlap.shape != hamiltonian.shape:
        raise ValueError("Hamiltonian and overlap must have identical shapes")
    if not (
        np.isfinite(hamiltonian.real).all()
        and np.isfinite(hamiltonian.imag).all()
        and np.isfinite(overlap.real).all()
        and np.isfinite(overlap.imag).all()
    ):
        raise ValueError("Hamiltonian and overlap must be finite")
    try:
        return sla.eigh(
            hamiltonian,
            overlap,
            overwrite_a=True,
            overwrite_b=True,
            check_finite=True,
        )
    except sla.LinAlgError as error:
        raise RuntimeError(
            "generalized eigensolve failed; overlap matrix is not positive definite "
            "or the dense problem did not converge (no diagonal jitter was applied)"
        ) from error


def _weights_from_bare_velocity(
    eigenvectors: np.ndarray, matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    velocity = eigenvectors.conj().T @ matrix @ eigenvectors
    velocity = 0.5 * (velocity + velocity.conj().T)
    return velocity, np.abs(velocity) ** 2


def compute_msd_diffusion(
    eigenvalues_eV: np.ndarray,
    velocity_weight: np.ndarray,
    energy_eV: np.ndarray,
    time_fs: np.ndarray,
    *,
    center_eV: float,
    half_width_eV: float,
    n_kpm: int,
    density_tol: float = DENSITY_TOL,
    broadened_delta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return density, valid mask, MSD and D with shape ``(time, energy)``."""
    eps = np.asarray(eigenvalues_eV, dtype=np.float64)
    weights = np.asarray(velocity_weight, dtype=np.float64)
    energy = np.asarray(energy_eV, dtype=np.float64)
    times = np.asarray(time_fs, dtype=np.float64)
    if eps.ndim != 1 or weights.shape != (eps.size, eps.size):
        raise ValueError("velocity weights must have shape (norb, norb)")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("time grid needs at least two samples")
    if not np.isfinite(times).all() or not np.all(np.diff(times) > 0.0):
        raise ValueError("time grid must be finite and strictly increasing")
    if not np.isfinite(weights).all():
        raise ValueError("velocity weights must be finite")
    if density_tol < 0.0:
        raise ValueError("density tolerance cannot be negative")

    if broadened_delta is None:
        broadened = delta_matrix(
            energy, eps, center_eV, half_width_eV, n_kpm
        )
    else:
        broadened = np.asarray(broadened_delta, dtype=np.float64)
        if broadened.shape != (energy.size, eps.size):
            raise ValueError(
                "precomputed broadened_delta must have shape (n_energy, norb)"
            )
        if not np.isfinite(broadened).all():
            raise ValueError("precomputed broadened_delta must be finite")
    density = broadened.sum(axis=1)
    valid = np.abs(density) > density_tol
    msd = np.full((times.size, energy.size), np.nan, dtype=np.float64)

    energy_difference = eps[:, None] - eps[None, :]
    small = np.abs(energy_difference) < 1.0e-9
    non_small = ~small
    for index, current_time in enumerate(times):
        pair_kernel = np.empty_like(energy_difference)
        pair_kernel[small] = (current_time / HBAR_EV_FS) ** 2
        pair_kernel[non_small] = (
            4.0
            * np.sin(
                energy_difference[non_small]
                * current_time
                / (2.0 * HBAR_EV_FS)
            )
            ** 2
            / energy_difference[non_small] ** 2
        )
        numerator_per_state = np.einsum(
            "nm,nm->n", weights, pair_kernel, optimize=True
        )
        numerator = broadened @ numerator_per_state
        msd[index, valid] = numerator[valid] / density[valid]

    diffusion = 0.5 * np.gradient(msd, times, axis=0, edge_order=1)
    return density, valid, msd, diffusion


def estimate_resources(norb: int, n_energy: int) -> Dict[str, Any]:
    """Conservative dense-memory estimate for this implementation.

    The peak reserves ten complex128 NxN equivalents for LAPACK workspace,
    eigenvectors, streamed operator transforms, pair-energy and velocity-weight
    arrays, plus the real float64 Jackson delta matrix (n_energy x norb).
    """
    if norb < 1 or n_energy < 1:
        raise ValueError("norb and n_energy must be positive")
    one_complex = 16 * norb * norb
    delta_bytes = 8 * n_energy * norb
    peak_bytes = 10 * one_complex + delta_bytes
    if peak_bytes < 16 * GIB:
        status = "SAFE"
    elif peak_bytes < 64 * GIB:
        status = "LARGE"
    else:
        status = "VERY LARGE"
    return {
        "norb": int(norb),
        "n_energy": int(n_energy),
        "estimated_dense_H_GiB": one_complex / GIB,
        "estimated_dense_S_GiB": one_complex / GIB,
        "estimated_eigenvectors_GiB": one_complex / GIB,
        "estimated_per_axis_operator_GiB": one_complex / GIB,
        "estimated_delta_matrix_GiB": delta_bytes / GIB,
        "estimated_peak_memory_GiB": peak_bytes / GIB,
        "complexity": "dense generalized eigensolve: O(norb^3)",
        "status": status,
    }


def parse_axes(value: str) -> Tuple[str, ...]:
    normalized = value.lower().replace(",", "").replace(" ", "")
    if not normalized:
        raise ValueError("at least one axis is required")
    invalid = [axis for axis in normalized if axis not in AXIS_INDEX]
    if invalid:
        raise ValueError("axes may contain only x, y, z")
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate axes are not allowed")
    return tuple(normalized)


def time_grid(dt_fs: float, n_steps: int) -> np.ndarray:
    if not np.isfinite(dt_fs) or dt_fs <= 0.0:
        raise ValueError("dt_fs must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be at least 1")
    return np.arange(n_steps + 1, dtype=np.float64) * dt_fs


def resolve_run_parameters(
    resolved: Mapping[str, Any],
    *,
    axes: Sequence[str],
    dt_fs: Optional[float],
    n_steps: Optional[int],
    n_kpm: Optional[int],
) -> RunParameters:
    """Resolve CLI > matching JSON MSD task > current RSATB defaults."""
    requested = set(axes)
    tasks = [
        task
        for task in resolved.get("tasks", [])
        if task.get("type") == "msd" and task.get("axis", "x") in requested
    ]
    cli_values = {"dt_fs": dt_fs, "n_steps": n_steps, "n_kpm": n_kpm}
    values: Dict[str, Any] = {}
    sources: Dict[str, str] = {}
    for key, cli_value in cli_values.items():
        if cli_value is not None:
            values[key] = cli_value
            sources[key] = "cli"
            continue
        task_values = {task.get(key, RSATB_MSD_DEFAULTS[key]) for task in tasks}
        if len(task_values) > 1:
            raise ValueError(
                f"conflicting MSD tasks specify different {key}; set --{key.replace('_', '-')}"
            )
        if task_values:
            values[key] = task_values.pop()
            sources[key] = "json"
        else:
            values[key] = RSATB_MSD_DEFAULTS[key]
            sources[key] = "rsatb-default"

    result = RunParameters(
        dt_fs=float(values["dt_fs"]),
        n_steps=int(values["n_steps"]),
        n_kpm=int(values["n_kpm"]),
        sources=sources,
    )
    time_grid(result.dt_fs, result.n_steps)
    jackson_kernel(result.n_kpm)
    return result


def run_parameters_from_benchmark_grid(
    energy_eV: np.ndarray,
    time_fs: np.ndarray,
    explicit_n_kpm: Optional[int] = None,
) -> RunParameters:
    """Use an existing RSATB MSD grid as the complete grid authority."""
    energy = np.asarray(energy_eV)
    times = np.asarray(time_fs, dtype=np.float64)
    if energy.ndim != 1 or energy.size < 1:
        raise ValueError("benchmark energy grid must be one-dimensional and non-empty")
    if times.ndim != 1 or times.size < 2:
        raise ValueError("benchmark time grid needs at least two samples")
    steps = np.diff(times)
    if not np.allclose(steps, steps[0], rtol=1e-12, atol=1e-14):
        raise ValueError("benchmark time grid must be uniform")
    if explicit_n_kpm is not None and explicit_n_kpm != energy.size:
        raise ValueError(
            "--grid-from-npz requires explicit --n-kpm to equal the energy-grid size"
        )
    sources = {
        "dt_fs": "grid-from-npz",
        "n_steps": "grid-from-npz",
        "n_kpm": "grid-from-npz",
    }
    return RunParameters(
        dt_fs=float(steps[0]),
        n_steps=int(times.size - 1),
        n_kpm=int(energy.size),
        sources=sources,
    )


def coo_matrix_from_dump(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    *,
    shape: Tuple[int, int],
    label: str,
) -> sp.csr_matrix:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(values, dtype=np.complex128)
    if rows.ndim != 1 or cols.ndim != 1 or values.ndim != 1:
        raise ValueError(f"{label} COO arrays must be one-dimensional")
    if not (rows.size == cols.size == values.size):
        raise ValueError(f"{label} COO arrays have different lengths")
    if shape[0] != shape[1] or shape[0] < 1:
        raise ValueError(f"{label} must have a positive square shape")
    if rows.size and (
        rows.min() < 0
        or cols.min() < 0
        or rows.max() >= shape[0]
        or cols.max() >= shape[1]
    ):
        raise ValueError(f"{label} COO index is outside shape {shape}")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError(f"{label} contains non-finite values")
    matrix = sp.coo_matrix((values, (rows, cols)), shape=shape).tocsr()
    matrix.sum_duplicates()
    return matrix


def sparse_hermiticity_error(matrix: sp.spmatrix) -> float:
    difference = (matrix - matrix.getH()).tocsr()
    difference.eliminate_zeros()
    if difference.nnz == 0:
        return 0.0
    return float(np.max(np.abs(difference.data)))


def output_paths(output: Path) -> Dict[str, Path]:
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ValueError("--output must end in .npz")
    return {
        "npz": output,
        "json": output.with_suffix(".json"),
        "log": output.with_suffix(".log"),
        "intermediate": output.with_name(output.stem + "_intermediate.npz"),
    }


def ensure_output_targets_available(
    output: Path, force: bool, include_intermediate: bool = False
) -> None:
    paths = output_paths(output)
    keys = ["npz", "json"]
    if include_intermediate:
        keys.append("intermediate")
    conflicts = [paths[key] for key in keys if paths[key].exists()]
    if conflicts and not force:
        listed = ", ".join(str(path) for path in conflicts)
        raise FileExistsError(f"output exists ({listed}); use --force to overwrite")


def make_shape_metadata(
    energy_eV: np.ndarray,
    time_fs: np.ndarray,
    msd_by_axis: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    shapes: Dict[str, Any] = {
        "energy_eV": list(np.asarray(energy_eV).shape),
        "time_fs": list(np.asarray(time_fs).shape),
    }
    for axis, values in msd_by_axis.items():
        shapes[f"msd_{axis}_A2"] = list(np.asarray(values).shape)
        shapes[f"diffusion_{axis}_A2_per_fs"] = list(np.asarray(values).shape)
    return shapes


def _atomic_save_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.npz", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("exact_msd_diffusion")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def _infer_norb_from_overlap_dump(dump: Tuple[np.ndarray, ...]) -> int:
    rows, cols, _ = dump
    if len(rows) == 0:
        raise ValueError("S operator dump is empty; cannot determine norb")
    return int(max(np.max(rows), np.max(cols))) + 1


def _dump_sparse(
    cpp: Any, which: str, axis: int, k_index: int, norb: int
) -> sp.csr_matrix:
    return coo_matrix_from_dump(
        *cpp.dump_operator_coo(which, axis, k_index),
        shape=(norb, norb),
        label=which,
    )


def _max_abs_sparse(matrix: sp.spmatrix) -> float:
    return float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0


def _validate_hermitian(matrix: sp.spmatrix, label: str) -> float:
    error = sparse_hermiticity_error(matrix)
    scale = max(1.0, _max_abs_sparse(matrix))
    if error > HERMITICITY_RTOL * scale:
        raise ValueError(
            f"{label} is not Hermitian: max error {error:.6e}, scale {scale:.6e}"
        )
    return error


def _gamma_index(resolved: Mapping[str, Any]) -> int:
    from rsatb.inputs.build import build_kpoints

    kpoints = build_kpoints(
        resolved.get("run", {}).get("kpoints", [[0.0, 0.0, 0.0]])
    )
    for index, point in enumerate(kpoints):
        if all(abs(float(component)) < 1e-12 for component in point):
            return index
    raise ValueError("run.kpoints does not contain Gamma [0, 0, 0]")


def _input_paths(resolved: Mapping[str, Any]) -> Dict[str, Any]:
    structure = resolved.get("structure", {})
    model = resolved.get("model", {})
    return {
        "structure": structure.get("file", "inline"),
        "hamiltonian": model.get("hamiltonian", {}).get("path"),
        "orbitals": copy.deepcopy(model.get("orbitals", {})),
    }


def _verify_checksum_file(config_path: Path, logger: logging.Logger) -> Dict[str, Any]:
    checksum_file = config_path.parent / "metadata" / "checksums.sha256"
    if not checksum_file.exists():
        logger.info("input checksum file not present: %s", checksum_file)
        return {"requested": True, "available": False, "verified": 0}
    verified = 0
    failures = []
    for line_number, raw_line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            failures.append(f"line {line_number}: invalid sha256 format")
            continue
        expected, relative = parts
        relative = relative.lstrip("*")
        candidate = (config_path.parent / relative).resolve()
        if not candidate.is_file():
            failures.append(f"missing: {candidate}")
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest.lower() != expected.lower():
            failures.append(f"checksum mismatch: {candidate}")
            continue
        verified += 1
    if failures:
        raise ValueError("input checksum verification failed:\n" + "\n".join(failures))
    logger.info("input checksum verification passed: %d files", verified)
    return {
        "requested": True,
        "available": True,
        "verified": verified,
        "file": str(checksum_file),
    }


def _blas_metadata() -> Dict[str, Any]:
    variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
    )
    info: Dict[str, Any] = {
        "thread_environment": {name: os.environ.get(name) for name in variables}
    }
    try:
        from threadpoolctl import threadpool_info

        info["threadpools"] = threadpool_info()
    except Exception:
        info["threadpools"] = "threadpoolctl unavailable"
    return info


def _git_commit(path: Path) -> Optional[str]:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _rsatb_version() -> Optional[str]:
    try:
        import importlib.metadata

        return importlib.metadata.version("rsatb")
    except Exception:
        try:
            import rsatb

            return getattr(rsatb, "__version__", None)
        except Exception:
            return None


def _load_grid_from_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "energy_eV" in data:
            energy = np.array(data["energy_eV"], dtype=np.float64)
        elif "energy" in data:
            energy = np.array(data["energy"], dtype=np.float64)
        else:
            raise ValueError("grid NPZ needs energy_eV or energy")
        if "time_fs" not in data:
            raise ValueError("grid NPZ needs time_fs")
        times = np.array(data["time_fs"], dtype=np.float64)
    if energy.ndim != 1 or times.ndim != 1 or times.size < 2:
        raise ValueError("benchmark energy/time grids must be one-dimensional")
    if not np.all(np.diff(energy) > 0.0) or not np.all(np.diff(times) > 0.0):
        raise ValueError("benchmark energy/time grids must be strictly increasing")
    if not np.isclose(times[0], 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("benchmark time grid must start at 0 fs")
    return energy, times


def _resolve_grids(
    args: argparse.Namespace,
    params: RunParameters,
    spectral_bounds: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, float, float, int, str]:
    if args.grid_from_npz:
        energy, times = _load_grid_from_npz(Path(args.grid_from_npz).resolve())
        if args.n_kpm is not None and args.n_kpm != energy.size:
            raise ValueError(
                "--grid-from-npz requires explicit --n-kpm to equal the energy-grid size"
            )
        center, half_width = window_from_chebyshev_grid(energy)
        return energy, times, center, half_width, int(energy.size), "npz"

    explicit = (args.emin, args.emax, args.n_energy)
    if any(value is not None for value in explicit) and not all(
        value is not None for value in explicit
    ):
        raise ValueError("--emin, --emax and --n-energy must be given together")
    times = time_grid(params.dt_fs, params.n_steps)
    if all(value is not None for value in explicit):
        if args.n_energy < 2:
            raise ValueError("--n-energy must be at least 2")
        if args.emax <= args.emin:
            raise ValueError("--emin must be smaller than --emax")
        energy = np.linspace(args.emin, args.emax, args.n_energy, dtype=np.float64)
        center = 0.5 * (args.emin + args.emax)
        half_width = 0.5 * (args.emax - args.emin)
        return energy, times, center, half_width, params.n_kpm, "cli-uniform"

    emin, emax = map(float, spectral_bounds)
    energy = chebyshev_energy_grid(emin, emax, params.n_kpm)
    return energy, times, 0.5 * (emin + emax), 0.5 * (emax - emin), params.n_kpm, "rsatb-chebyshev"


def _dense_operator(cpp: Any, which: str, axis: int, k_index: int, norb: int) -> np.ndarray:
    return _dump_sparse(cpp, which, axis, k_index, norb).toarray()


def _rotate_dumped_operator(
    cpp: Any,
    which: str,
    axis: int,
    k_index: int,
    norb: int,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """Dump, densify and rotate one operator without retaining other raw operators."""
    matrix = _dense_operator(cpp, which, axis, k_index, norb)
    work = matrix @ eigenvectors
    rotated = eigenvectors.conj().T @ work
    del matrix, work
    return rotated


def _velocity_for_axis(
    cpp: Any,
    axis: int,
    k_index: int,
    norb: int,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        bond_t = _rotate_dumped_operator(
            cpp, "bond", axis, k_index, norb, eigenvectors
        )
        xbar_t = _rotate_dumped_operator(
            cpp, "xbar", axis, k_index, norb, eigenvectors
        )
        delta_t = _rotate_dumped_operator(
            cpp, "delta", axis, k_index, norb, eigenvectors
        )
    except RuntimeError as composite_error:
        try:
            bare = _dense_operator(cpp, "matrix", axis, k_index, norb)
        except RuntimeError:
            raise composite_error
        velocity, weights = _weights_from_bare_velocity(eigenvectors, bare)
        return velocity, weights, "bare matrix rotated as C^H V C"
    velocity = 1j * (
        bond_t
        - xbar_t * eigenvalues[None, :]
        + eigenvalues[:, None] * delta_t
    )
    velocity = 0.5 * (velocity + velocity.conj().T)
    weights = np.abs(velocity) ** 2
    return velocity, weights, "bond/xbar/delta composite from gen_reference_ops.py"


def _parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact generalized-eigenproblem reference for RSATB MSD(E,t) and D(E,t). "
            "Run with one MPI rank only."
        )
    )
    parser.add_argument("--config", required=True, help="RSATB input JSON")
    parser.add_argument("--output", help="output .npz (required for a normal run)")
    parser.add_argument("--name", help="metadata label; defaults to config stem")
    parser.add_argument("--axes", default="xyz", help="axes to calculate, default: xyz")
    parser.add_argument("--dt-fs", type=float)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--emin", type=float)
    parser.add_argument("--emax", type=float)
    parser.add_argument("--n-energy", type=int)
    parser.add_argument("--n-kpm", type=int)
    parser.add_argument("--grid-from-npz", help="reuse energy and time_fs from an RSATB MSD NPZ")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check-only", action="store_true")
    modes.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--memory-limit-gb", type=float, default=64.0)
    parser.add_argument("--save-eigenvalues", action="store_true")
    parser.add_argument("--save-intermediate", action="store_true")
    parser.add_argument("--save-eigenvectors", action="store_true")
    parser.add_argument("--verify-input", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not (args.check_only or args.estimate_only) and not args.output:
        parser.error("--output is required unless --check-only or --estimate-only is used")
    if args.save_eigenvectors and not args.save_intermediate:
        parser.error("--save-eigenvectors requires --save-intermediate")
    if args.grid_from_npz and any(
        value is not None
        for value in (args.dt_fs, args.n_steps, args.emin, args.emax, args.n_energy)
    ):
        parser.error(
            "--grid-from-npz cannot be combined with time or explicit energy-grid overrides"
        )
    return args


def _build_and_validate(
    config_path: Path, logger: logging.Logger
) -> Tuple[Dict[str, Any], Any, int, int, Dict[str, Any], sp.csr_matrix, sp.csr_matrix]:
    try:
        from mpi4py import MPI
        from rsatb.inputs.build import build_calculation, build_structure
        from rsatb.inputs.check import load_and_resolve
    except ImportError as error:
        raise RuntimeError(
            "compiled RSATB environment is required; activate the environment containing "
            "rsatb, _rsatb and mpi4py"
        ) from error

    world_size = MPI.COMM_WORLD.Get_size()
    if world_size != 1:
        raise RuntimeError(
            f"exact dense reference requires one MPI rank, but MPI_COMM_WORLD has {world_size}"
        )

    resolved, issues = load_and_resolve(str(config_path))
    for issue in issues:
        logger.warning("RSATB input issue: %s", issue)
    errors = [issue for issue in issues if getattr(issue, "is_error", True)]
    if errors:
        raise ValueError("RSATB input validation failed:\n" + "\n".join(map(str, errors)))

    serial = copy.deepcopy(resolved)
    serial.setdefault("run", {})["npar"] = 1
    serial["run"].setdefault("preconditioner", {})["operator_form"] = "solve"
    k_index = _gamma_index(serial)

    structure = build_structure(serial["structure"])
    base_atoms = int(structure.n_atom)
    supercell = serial["structure"].get("supercell", [1, 1, 1])
    natoms = base_atoms * int(np.prod(supercell))
    calc, _, _ = build_calculation(serial, log_level="silent")
    cpp = calc._cpp_obj

    overlap_dump = cpp.dump_operator_coo("S", 0, k_index)
    norb = _infer_norb_from_overlap_dump(overlap_dump)
    overlap = coo_matrix_from_dump(
        *overlap_dump, shape=(norb, norb), label="S"
    )
    hamiltonian = _dump_sparse(cpp, "H", 0, k_index, norb)
    if hamiltonian.shape != overlap.shape:
        raise ValueError(f"H shape {hamiltonian.shape} != S shape {overlap.shape}")
    h_error = _validate_hermitian(hamiltonian, "H")
    s_error = _validate_hermitian(overlap, "S")
    overlap_diagonal = overlap.diagonal().real
    if overlap_diagonal.size != norb or np.any(overlap_diagonal <= 0.0):
        raise ValueError("Overlap matrix has a non-positive diagonal entry")

    sparse_info = {
        "H_shape": list(hamiltonian.shape),
        "S_shape": list(overlap.shape),
        "H_nnz": int(hamiltonian.nnz),
        "S_nnz": int(overlap.nnz),
        "H_sparsity": float(hamiltonian.nnz / (norb * norb)),
        "S_sparsity": float(overlap.nnz / (norb * norb)),
        "max_H_hermiticity_error": h_error,
        "max_S_hermiticity_error": s_error,
        "gamma_k_index": k_index,
        "effective_npar": 1,
        "effective_operator_form": "solve",
    }
    logger.info("natoms=%d norb=%d Gamma k-index=%d", natoms, norb, k_index)
    logger.info("H shape=%s nnz=%d Hermiticity=%.6e", hamiltonian.shape, hamiltonian.nnz, h_error)
    logger.info("S shape=%s nnz=%d Hermiticity=%.6e", overlap.shape, overlap.nnz, s_error)
    return serial, calc, natoms, norb, sparse_info, hamiltonian, overlap


def _print_estimate(
    natoms: int,
    sparse_info: Mapping[str, Any],
    estimate: Mapping[str, Any],
) -> None:
    print(f"natoms             = {natoms}")
    print(f"norb               = {estimate['norb']}")
    print(f"H nnz              = {sparse_info['H_nnz']}")
    print(f"S nnz              = {sparse_info['S_nnz']}")
    print(f"H sparsity         = {sparse_info['H_sparsity']:.6e}")
    print(f"S sparsity         = {sparse_info['S_sparsity']:.6e}")
    print(f"one complex128 NxN = {estimate['estimated_dense_H_GiB']:.3f} GiB")
    print(f"estimated peak     = {estimate['estimated_peak_memory_GiB']:.3f} GiB")
    print(f"complexity         = {estimate['complexity']}")
    print(f"STATUS: {estimate['status']}")


def run(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_arguments(argv)
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    axes = parse_axes(args.axes)
    if args.memory_limit_gb <= 0.0:
        raise ValueError("--memory-limit-gb must be positive")

    if args.output:
        paths = output_paths(Path(args.output))
        ensure_output_targets_available(
            paths["npz"], args.force, include_intermediate=args.save_intermediate
        )
    else:
        base = Path.cwd() / f"{config_path.stem}_exact_msd_diffusion.npz"
        paths = output_paths(base)
    logger = _configure_logger(paths["log"])
    logger.info("start exact MSD/D reference")
    logger.info("config=%s", config_path)
    logger.info("axes=%s", "".join(axes))
    logger.info("BLAS=%s", json.dumps(_blas_metadata(), default=str))

    checksum_info: Dict[str, Any] = {"requested": False}
    if args.verify_input:
        checksum_info = _verify_checksum_file(config_path, logger)

    resolved, calc, natoms, norb, sparse_info, h_sparse, s_sparse = _build_and_validate(
        config_path, logger
    )
    if args.grid_from_npz:
        preview_energy, preview_times = _load_grid_from_npz(
            Path(args.grid_from_npz).resolve()
        )
        params = run_parameters_from_benchmark_grid(
            preview_energy, preview_times, explicit_n_kpm=args.n_kpm
        )
        provisional_n_energy = preview_energy.size
    else:
        params = resolve_run_parameters(
            resolved,
            axes=axes,
            dt_fs=args.dt_fs,
            n_steps=args.n_steps,
            n_kpm=args.n_kpm,
        )
        provisional_n_energy = args.n_energy if args.n_energy is not None else params.n_kpm
    if args.n_energy is not None:
        provisional_n_energy = args.n_energy
    estimate = estimate_resources(norb, int(provisional_n_energy))
    logger.info("memory estimate=%s", json.dumps(estimate, sort_keys=True))
    _print_estimate(natoms, sparse_info, estimate)

    if args.check_only:
        print("INPUT CHECK: PASS")
        logger.info("INPUT CHECK: PASS; no dense conversion or diagonalization")
        return 0
    if args.estimate_only:
        logger.info("estimate-only complete; no dense conversion or diagonalization")
        return 0
    if estimate["estimated_peak_memory_GiB"] > args.memory_limit_gb and not args.force:
        raise MemoryError(
            f"estimated peak {estimate['estimated_peak_memory_GiB']:.3f} GiB exceeds "
            f"--memory-limit-gb {args.memory_limit_gb:.3f}; use a suitable limit or --force"
        )

    spectral_bounds = tuple(map(float, calc.energy_bounds()))
    energy, times, center, half_width, n_kpm, grid_source = _resolve_grids(
        args, params, spectral_bounds
    )
    logger.info(
        "energy grid source=%s n=%d window=[%.9g, %.9g] n_kpm=%d",
        grid_source,
        energy.size,
        center - half_width,
        center + half_width,
        n_kpm,
    )
    logger.info("time grid n=%d dt_first=%.9g final=%.9g fs", times.size, times[1] - times[0], times[-1])

    logger.info("dense conversion start")
    hamiltonian = h_sparse.toarray()
    overlap = s_sparse.toarray()
    del h_sparse, s_sparse
    logger.info("generalized diagonalization start")
    diagonalization_start = time.monotonic()
    eigenvalues, eigenvectors = solve_generalized_eigenproblem(
        hamiltonian, overlap
    )
    del hamiltonian, overlap
    diagonalization_seconds = time.monotonic() - diagonalization_start
    logger.info(
        "generalized diagonalization end seconds=%.3f eigenvalue range=[%.9g, %.9g] eV",
        diagonalization_seconds,
        eigenvalues[0],
        eigenvalues[-1],
    )
    if eigenvalues[0] < center - half_width or eigenvalues[-1] > center + half_width:
        logger.warning(
            "eigenvalues extend outside broadening window; endpoint clipping follows the reference formula"
        )

    cpp = calc._cpp_obj
    k_index = int(sparse_info["gamma_k_index"])
    logger.info("Jackson-Chebyshev broadening start")
    broadened_delta = delta_matrix(
        energy, eigenvalues, center, half_width, n_kpm
    )
    logger.info("Jackson-Chebyshev broadening end")
    density_reference: Optional[np.ndarray] = None
    valid_reference: Optional[np.ndarray] = None
    msd_by_axis: Dict[str, np.ndarray] = {}
    diffusion_by_axis: Dict[str, np.ndarray] = {}
    intermediate: Dict[str, np.ndarray] = {}
    velocity_implementations: Dict[str, str] = {}
    if args.save_eigenvectors:
        intermediate["eigenvectors"] = eigenvectors

    for axis in axes:
        logger.info("axis %s start", axis)
        axis_start = time.monotonic()
        velocity, weights, implementation = _velocity_for_axis(
            cpp,
            AXIS_INDEX[axis],
            k_index,
            norb,
            eigenvalues,
            eigenvectors,
        )
        velocity_implementations[axis] = implementation
        logger.info("axis %s max_abs_velocity=%.9g", axis, np.max(np.abs(velocity)))
        density, valid, msd, diffusion = compute_msd_diffusion(
            eigenvalues,
            weights,
            energy,
            times,
            center_eV=center,
            half_width_eV=half_width,
            n_kpm=n_kpm,
            broadened_delta=broadened_delta,
        )
        if density_reference is None:
            density_reference = density
            valid_reference = valid
        else:
            np.testing.assert_allclose(density, density_reference, rtol=1e-13, atol=1e-14)
            np.testing.assert_array_equal(valid, valid_reference)
        msd_by_axis[axis] = msd
        diffusion_by_axis[axis] = diffusion
        if args.save_intermediate:
            intermediate[f"velocity_{axis}"] = velocity
            intermediate[f"velocity_weight_{axis}"] = weights
        del velocity, weights
        logger.info("axis %s end seconds=%.3f", axis, time.monotonic() - axis_start)

    arrays: Dict[str, np.ndarray] = {
        "energy_eV": energy,
        "time_fs": times,
        "spectral_density_per_eV": density_reference,
        "valid_energy_mask": valid_reference,
        "norb": np.array(norb, dtype=np.int64),
        "natoms": np.array(natoms, dtype=np.int64),
    }
    for axis in axes:
        arrays[f"msd_{axis}_A2"] = msd_by_axis[axis]
        arrays[f"diffusion_{axis}_A2_per_fs"] = diffusion_by_axis[axis]
    if set(axes) == set(AXIS_INDEX):
        msd_total = sum(msd_by_axis[axis] for axis in ("x", "y", "z"))
        diffusion_iso = sum(
            diffusion_by_axis[axis] for axis in ("x", "y", "z")
        ) / 3.0
        arrays["msd_total_A2"] = msd_total
        arrays["diffusion_iso_A2_per_fs"] = diffusion_iso
        consistency_values = np.abs(
            diffusion_iso
            - np.gradient(msd_total, times, axis=0, edge_order=1) / 6.0
        )
        finite_consistency = consistency_values[np.isfinite(consistency_values)]
        consistency = (
            float(np.max(finite_consistency)) if finite_consistency.size else None
        )
    else:
        consistency = None
    if args.save_eigenvalues or args.save_intermediate:
        arrays["eigenvalues_eV"] = eigenvalues

    metadata: Dict[str, Any] = {
        "name": args.name or config_path.stem,
        "config": str(config_path),
        **_input_paths(resolved),
        "natoms": natoms,
        "norb": norb,
        "dt_fs": float(times[1] - times[0]),
        "n_steps": int(times.size - 1),
        "parameter_sources": params.sources,
        "emin_eV": float(energy[0]),
        "emax_eV": float(energy[-1]),
        "broadening_window_emin_eV": float(center - half_width),
        "broadening_window_emax_eV": float(center + half_width),
        "n_energy": int(energy.size),
        "n_kpm": int(n_kpm),
        "energy_grid_source": grid_source,
        "grid_from_npz": str(Path(args.grid_from_npz).resolve()) if args.grid_from_npz else None,
        "axes": list(axes),
        "method": "exact diagonalization",
        "generalized_eigenproblem": True,
        "generalized_eigenproblem_equation": "H C = S C E",
        "velocity_operator": velocity_implementations,
        "energy_broadening": "RSATB Jackson-Chebyshev delta kernel",
        "msd_definition": "make_ref_curves.py::ref_msd pair sum",
        "diffusion_definition": "D_a(E,t) = 0.5 dMSD_a(E,t)/dt",
        "density_tolerance": DENSITY_TOL,
        "units": {
            "energy": "eV",
            "time": "fs",
            "msd": "Angstrom^2",
            "diffusion": "Angstrom^2/fs",
            "spectral_density": "1/eV",
        },
        "array_shapes": make_shape_metadata(energy, times, msd_by_axis),
        "sparse_checks": sparse_info,
        "resource_estimate": estimate,
        "memory_limit_GiB": args.memory_limit_gb,
        "force": bool(args.force),
        "checksum_verification": checksum_info,
        "eigenvalue_min_eV": float(eigenvalues[0]),
        "eigenvalue_max_eV": float(eigenvalues[-1]),
        "diagonalization_seconds": diagonalization_seconds,
        "isotropic_consistency_max_abs_A2_per_fs": (
            consistency
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "rsatb": _rsatb_version(),
            "script_git_commit": _git_commit(Path(__file__).resolve().parent),
        },
        "blas": _blas_metadata(),
        "single_mpi_rank_required": True,
        "atom_order_modified": False,
    }

    logger.info("writing output=%s", paths["npz"])
    _atomic_save_npz(paths["npz"], arrays)
    _atomic_write_json(paths["json"], metadata)
    if args.save_intermediate:
        if paths["intermediate"].exists() and not args.force:
            raise FileExistsError(
                f"intermediate output exists: {paths['intermediate']}; use --force"
            )
        _atomic_save_npz(paths["intermediate"], intermediate)
        logger.info("wrote intermediate=%s", paths["intermediate"])
    logger.info("finished output=%s metadata=%s", paths["npz"], paths["json"])
    return 0


def main() -> int:
    try:
        return run()
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError, MemoryError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
