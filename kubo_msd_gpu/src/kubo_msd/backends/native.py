"""Explicit, fail-closed adapter for the independent CPU stochastic extension."""
from importlib import import_module
import numpy as np
import scipy.sparse as sp

from ..contracts import MetricSolveReport
from ..errors import NumericalError
from ..units import HBAR_EV_FS


def native_requested(config):
    return (config.backend == "cpu" and config.solver == "stochastic-kpm"
            and config.cpu_kernel == "cpp")


def require_extension():
    try:
        return import_module("kubo_msd._native")
    except (ImportError, OSError) as exc:
        raise NumericalError(
            "CPU cpp kernel requires building the native extension: "
            "python -m pip install --no-build-isolation -e . "
            "(requires a C++17/OpenMP compiler and pybind11); "
            "select --cpu-kernel reference for the Python/SciPy implementation"
        ) from exc


def extension_info():
    return dict(require_extension().build_info())


def _call(function, *args):
    try:
        return function(*args)
    except (ValueError, RuntimeError) as exc:
        failure = NumericalError(str(exc))
        if hasattr(exc, "diagnostics"):
            failure.diagnostics = exc.diagnostics
        raise failure from exc


def csr(matrix, threads, extension):
    host = sp.csr_matrix(matrix, dtype=np.complex128, copy=True)
    host.sum_duplicates()
    host.sort_indices()
    return extension.CSR(host.indptr, host.indices, host.data, host.shape[0], threads)


class NativeMetric:
    def __init__(self, host, diagonal, blocks, config):
        self.extension = require_extension()
        self.S = csr(host, config.omp_threads, self.extension)
        self.core = self.extension.Metric(self.S, diagonal, blocks, config.omp_threads)

    def solve(self, B, config):
        rtol = config.rtol if hasattr(config, "rtol") else config.metric_rtol
        values, report = _call(
            self.core.solve, B, max(rtol, 0), config.metric_atol,
            config.metric_maxiter)
        return values, MetricSolveReport(
            np.asarray(report["converged"], dtype=bool),
            np.asarray(report["iterations"], dtype=np.int64),
            np.asarray(report["true_residual"], dtype=float),
            np.asarray(report["relative_residual"], dtype=float),
            np.asarray(report["breakdown"], dtype=bool))

    def sample(self, xi, coeff, center, half):
        return _call(self.core.sample, xi, coeff, center, half)

    def diagnostics(self):
        values = dict(self.core.diagnostics())
        info = dict(self.extension.build_info())
        info.update({k: values.pop(k) for k in
                     ("omp_threads_requested", "omp_threads_observed")})
        return {**values, "native": info}


class NativeOperators:
    def __init__(self, ops, config):
        extension = ops.metric.native.extension
        self.H = csr(ops.host_H, config.omp_threads, extension)
        self.S = ops.metric.native.S
        definitions = []
        for axis in "xyz":
            if axis in ops.velocity:
                kind, matrices = 0, [ops.velocity[axis]]
            elif axis in ops.position:
                kind, matrices = 1, [ops.position[axis]]
            elif axis in ops.composite:
                kind, matrices = 2, ops.composite[axis]
            else:
                kind, matrices = -1, []
            definitions.append((kind, [csr(m, config.omp_threads, extension)
                                       for m in matrices]))
        self.core = extension.Engine(
            self.H, ops.metric.native.core, definitions, config.metric_rtol,
            config.metric_atol, config.metric_maxiter, HBAR_EV_FS)

    def h_action(self, B):
        return _call(self.core.h_action, B)

    def v_cov(self, direction, B):
        return _call(self.core.v_cov, B, list(direction))

    def series(self, B, coeff, center, half, directions=()):
        return _call(self.core.series, B, coeff, center, half,
                     [list(d) for d in directions])

    def chain(self, B, length, center, half):
        return _call(self.core.chain, B, length, center, half)

    def moments(self, B, length, center, half):
        return _call(self.core.moments, B, length, center, half)
