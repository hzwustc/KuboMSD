"""Independent dense/reference checks for the compiled CPU stochastic path."""
from dataclasses import replace
import importlib.util

import numpy as np
from numpy.testing import assert_allclose
import pytest
import scipy.linalg as la
import scipy.sparse as sp

from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.metric.solve import MetricSolver
from kubo_msd.metric.sampling import MetricSampler
from kubo_msd.operators.velocity import build_operators


def config(**kw):
    values = dict(energy_eV=[-.3, .1, .4], time_fs=[0, .07, .17],
                  solver="stochastic-kpm", kernel="jackson", axes=("x",),
                  spectral_bounds=(-2, 2), n_moments=20, metric_rtol=2e-13,
                  sampling_tol=1e-12, segment_fs=.06, include_vac=False,
                  cpu_kernel="cpp")
    values.update(kw)
    return RunConfig(**values)


def model(kind="velocity", n=9):
    rng = np.random.default_rng(903)
    A = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    S = np.eye(n) + .02*(A.conj().T @ A)
    H = (A + A.conj().T)/12
    X = np.diag(np.linspace(-.5, .8, n)).astype(complex)
    X += .03*(A+A.conj().T)
    if kind == "velocity":
        return ModelBundle(H, S, velocity={"x": X})
    if kind == "position":
        return ModelBundle(H, S, position={"x": X})
    r = np.linspace(0, 2, n)
    return ModelBundle(H, S, composite={"x": (
        H*(r[None, :]-r[:, None]), X-r[:, None]*S,
        X-S*r[None, :])})


def test_compiled_kernel_is_available_and_selected():
    # Catches an absent extension or silently routing cpp to the reference path.
    assert importlib.util.find_spec("kubo_msd._native") is not None, (
        "build the independently implemented C++ stochastic extension")
    ops = build_operators(model(), config(omp_threads=2))
    d = ops.metric.diagnostics()
    assert d["kernel"] == "cpp"
    assert d["native"]["omp_threads_requested"] == 2
    assert 1 <= d["native"]["omp_threads_observed"] <= 2
    assert d["native"]["openmp"]


@pytest.mark.parametrize("preconditioner", ["diagonal", "block-jacobi"])
@pytest.mark.parametrize("threads", [1, 2])
def test_native_pcg_preserves_scaled_rhs_and_true_residual(preconditioner, threads):
    rng = np.random.default_rng(97)
    A = rng.normal(size=(11, 11)) + 1j*rng.normal(size=(11, 11))
    S = A.conj().T@A + 2*np.eye(11)
    B = (rng.normal(size=(11, 4)) + 1j*rng.normal(size=(11, 4)))
    B *= [0, 1e-180, 1, 1e180]
    cfg = config(preconditioner=preconditioner, omp_threads=threads)
    solver = MetricSolver(S, cfg, atom_slices=[(0, 3), (3, 7), (7, 11)])
    assert solver.diagnostics().get("kernel") == "cpp"
    out, report = solver.solve_s(B)
    for j in (1, 2, 3):
        scale = np.max(np.abs(B[:, j]))
        assert_allclose(out[:, j]/scale, la.solve(S, B[:, j]/scale),
                        rtol=3e-11, atol=2e-12)
        residual = la.norm(B[:, j]/scale-S@(out[:, j]/scale))
        assert residual/la.norm(B[:, j]/scale) <= 1.1*cfg.metric_rtol
    assert_allclose(out[:, 0], 0, atol=0)
    assert report.converged.all()
    assert np.all(report.relative_residual <= cfg.metric_rtol)


def test_native_pcg_atol_and_failure_report():
    S = np.array([[1, .2j], [-.2j, 2]])
    solver = MetricSolver(S, config(metric_maxiter=1))
    assert solver.diagnostics().get("kernel") == "cpp"
    with pytest.raises(ValueError) as failed:
        solver.solve_s(np.array([[0, 1], [0, 1]], complex))
    diag = failed.value.diagnostics
    assert diag["converged"] == [True, False]
    assert diag["iterations"] == [0, 1]
    assert diag["relative_residual"][1] > solver.config.metric_rtol
    cfg = config(metric_atol=1e-100)
    out, report = MetricSolver(S, cfg).solve_s(np.full((2, 1), 1e-180))
    assert_allclose(out, 0, atol=0)
    assert report.converged.all()


def test_native_rejects_nonfinite_rhs():
    solver = MetricSolver(np.eye(2), config())
    assert solver.diagnostics().get("kernel") == "cpp"
    with pytest.raises(ValueError, match="finite"):
        solver.solve_s(np.array([[1], [np.nan]]))


def test_native_sampling_matches_reference_covariance():
    S = model().S
    native = MetricSampler(MetricSolver(S, config()), config())
    cfg = config(cpu_kernel="reference")
    reference = MetricSampler(MetricSolver(S, cfg), cfg)
    assert native.metric.diagnostics().get("kernel") == "cpp"
    B = np.eye(len(S), dtype=complex)
    transformed = native.transform(B)
    assert_allclose(transformed, reference.transform(B), rtol=2e-12, atol=2e-13)
    assert_allclose(transformed@transformed.conj().T, la.inv(S),
                    rtol=2e-11, atol=2e-12)


@pytest.mark.parametrize("kind", ["velocity", "position", "composite"])
def test_native_velocity_and_fused_recursions_match_reference(kind):
    from kubo_msd.solvers.chebyshev import chain, projected_quadratic
    from kubo_msd.solvers.stepwise import propagate, stepwise_samples
    from kubo_msd.units import HBAR_EV_FS
    m = model(kind)
    cfg = config()
    cpp = build_operators(m, cfg)
    ref = build_operators(m, replace(cfg, cpu_kernel="reference"))
    assert cpp.metric.diagnostics().get("kernel") == "cpp"
    rng = np.random.default_rng(18)
    B = rng.normal(size=(9, 3))+1j*rng.normal(size=(9, 3))
    assert_allclose(cpp.apply_v_cov("x", B), ref.apply_v_cov("x", B),
                    rtol=2e-11, atol=2e-12)
    assert_allclose(chain(B, cpp, cfg, 12), chain(B, ref, cfg, 12),
                    rtol=2e-10, atol=2e-11)
    assert_allclose(projected_quadratic(B, cpp, cfg),
                    projected_quadratic(B, ref, cfg), rtol=2e-10, atol=2e-11)
    actual = propagate(B, .17, cpp, cfg)[0]
    expected = la.expm(-1j*la.solve(m.S, m.H_cov)*.17/HBAR_EV_FS)@B
    assert_allclose(actual, expected, rtol=2e-10, atol=2e-11)
    assert_allclose(propagate(actual, -.17, cpp, cfg)[0], B,
                    rtol=2e-10, atol=2e-11)
    d1, n1, _ = stepwise_samples(cpp, cfg, B)
    d2, n2, _ = stepwise_samples(ref, cfg, B)
    assert_allclose(d1, d2, rtol=2e-10, atol=2e-11)
    assert_allclose(n1["msd_A2/commutator"], n2["msd_A2/commutator"],
                    rtol=2e-9, atol=2e-11)


def test_native_exact_keeps_reference_metric():
    ops = build_operators(model(), replace(config(), solver="exact"))
    assert ops.metric.diagnostics().get("kernel") == "reference"
    assert getattr(ops, "native", None) is None


def test_missing_native_extension_is_actionable(monkeypatch):
    from kubo_msd.backends import native
    def missing(_name):
        raise ImportError("controlled missing build")
    monkeypatch.setattr(native, "import_module", missing)
    with pytest.raises(ValueError, match="build.*extension|extension.*build"):
        build_operators(model(), config())


def test_native_scalar_rhs_is_rejected_without_crashing():
    # A scalar must be shape-checked before C++ reads shape[0].
    import subprocess
    import sys
    code = """
import numpy as np
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver
c=RunConfig(energy_eV=[0],time_fs=[0],solver='stochastic-kpm',kernel='jackson')
s=MetricSolver(np.eye(2),c)
try:
    s.solve_s(np.array(1))
except ValueError:
    pass
else:
    raise AssertionError('scalar RHS accepted')
"""
    done = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=10)
    assert done.returncode == 0, done.stdout+done.stderr


def test_native_true_residual_refresh_and_noncontiguous_rhs():
    n = 96
    S = sp.diags([-np.ones(n-1), 2.03*np.ones(n), -np.ones(n-1)],
                 [-1, 0, 1], format="csr", dtype=complex)
    rng = np.random.default_rng(88)
    B = (rng.normal(size=(n, 6))+1j*rng.normal(size=(n, 6)))[:, ::2]
    solver = MetricSolver(S, config(metric_rtol=1e-11, metric_maxiter=1000))
    out, report = solver.solve_s(B)
    assert solver.diagnostics()["true_residual_refreshes"] >= 1
    assert_allclose(S@out, B, rtol=1e-9, atol=1e-9)
    assert np.all(report.relative_residual <= 1e-11)


def test_native_parallel_spmm_matches_scipy_and_thread_count():
    # Work exceeds the native OpenMP threshold, unlike the tiny dense fixtures.
    n = 600
    off = np.linspace(.02, .04, n-1)*(1+1j)
    S = sp.diags([off.conj(), 1.2*np.ones(n), off], [-1, 0, 1], format="csr")
    rng = np.random.default_rng(99)
    B = rng.normal(size=(n, 8))+1j*rng.normal(size=(n, 8))
    a = MetricSolver(S, config(omp_threads=1))
    b = MetricSolver(S, config(omp_threads=2))
    assert_allclose(b.native.S.matmul(B), S@B, rtol=2e-15, atol=2e-15)
    assert_allclose(a.solve_s(B)[0], b.solve_s(B)[0], rtol=0, atol=0)


def test_native_fused_failure_retains_per_column_diagnostics():
    from kubo_msd.solvers.stepwise import propagate
    ops = build_operators(model(), config(metric_maxiter=1))
    with pytest.raises(ValueError) as failed:
        propagate(np.ones((9, 2), dtype=complex), .2, ops, ops.config)
    assert failed.value.diagnostics["converged"] == [False, False]
    assert failed.value.diagnostics["iterations"] == [1, 1]


@pytest.mark.parametrize("kernel", ["reference", "cpp"])
@pytest.mark.parametrize("include_metric_rtol", [False, True])
def test_metric_override_rtol_takes_precedence(kernel, include_metric_rtol):
    from types import SimpleNamespace
    values = dict(rtol=.5, metric_atol=0., metric_maxiter=1)
    if include_metric_rtol:
        values["metric_rtol"] = 1e-13
    solver = MetricSolver(np.array([[1., .2], [.2, 2.]]),
                          config(cpu_kernel=kernel))
    out, report = solver.solve_s(np.ones((2, 1)), SimpleNamespace(**values))
    # One Jacobi-preconditioned CG iteration, derived directly from p=(1, .5).
    assert_allclose(out[:, 0], [15/17, 15/34], rtol=1e-14, atol=1e-14)
    assert report.converged.tolist() == [True]
    assert report.iterations.tolist() == [1]
    assert 1e-12 < report.relative_residual[0] < .5


@pytest.mark.parametrize("kernel", ["reference", "cpp"])
def test_metric_finite_extreme_complex_rhs_preserves_representable_solution(kernel):
    solver = MetricSolver(np.eye(2), config(cpu_kernel=kernel))
    B = np.array([[complex(1.3e308, 1.3e308)], [1.+0j]])
    out, report = solver.solve_s(B)
    assert np.isfinite(out).all()
    assert_allclose(out[0].real/1.3e308, [1.], rtol=1e-14)
    assert_allclose(out[0].imag/1.3e308, [1.], rtol=1e-14)
    assert_allclose(out[1], [1.], rtol=1e-14, atol=0)
    assert report.converged.tolist() == [True]
    assert np.isfinite(report.relative_residual).all()


@pytest.mark.parametrize("kernel", ["reference", "cpp"])
def test_metric_rejects_unrepresentable_solution(kernel):
    from kubo_msd.errors import NumericalError
    solver = MetricSolver(.25*np.eye(2), config(cpu_kernel=kernel))
    B = np.array([[1e308, 1.], [1., 1.]], dtype=complex)
    with pytest.raises(NumericalError, match="finite|representable|overflow") as failed:
        solver.solve_s(B)
    assert failed.value.diagnostics["converged"] == [False, True]
