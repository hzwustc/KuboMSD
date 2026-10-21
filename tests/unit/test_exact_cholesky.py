import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import pytest
from numpy.testing import assert_allclose
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


def test_exact_full_cholesky_preserves_metric_equation_and_avoids_ritz(monkeypatch):
    import kubo_msd.metric.setup as setup
    def no_ritz(*args, **kwargs):
        raise AssertionError("Cholesky already verifies positive definiteness")
    monkeypatch.setattr(setup, "spd_bounds", no_ritz)
    rng = np.random.default_rng(91)
    A = rng.normal(size=(24, 24)) + 1j*rng.normal(size=(24, 24))
    S = A.conj().T@A + .2*np.eye(24)
    rhs = rng.normal(size=(24, 4)) + 1j*rng.normal(size=(24, 4))
    rhs[:, 0] = 0
    cfg = RunConfig([0], [0], preconditioner="full-cholesky", metric_maxiter=2, metric_rtol=1e-12)
    metric = MetricSolver(S, cfg, atom_slices=[(0, 12), (12, 24)])
    result, report = metric.solve_s(rhs)
    assert_allclose(result, la.solve(S, rhs), rtol=2e-12, atol=1e-12)
    assert np.max(report.iterations) <= 2
    assert np.all(report.relative_residual <= cfg.metric_rtol)
    assert metric.s_lower > 0
    assert metric.bounds_report["positive_definiteness"] == "Cholesky factorization"
    assert metric.bounds_report["certified"] is False  # condition estimate is not a certified bound
    assert metric.atom_slices == [(0, 12), (12, 24)]


def test_full_cholesky_budget_is_checked_before_dense_allocation(monkeypatch):
    cfg = RunConfig([0], [0], preconditioner="full-cholesky", memory_limit_gb=1e-6)
    def no_dense(*args, **kwargs):
        raise AssertionError("allocation preceded memory guard")
    monkeypatch.setattr(sp.csr_matrix, "toarray", no_dense)
    with pytest.raises(ValueError, match="memory"):
        MetricSolver(sp.eye(100), cfg)


def test_full_cholesky_rejects_indefinite_overlap():
    cfg = RunConfig([0], [0], preconditioner="full-cholesky")
    with pytest.raises(ValueError, match="positive definite"):
        MetricSolver([[1., 2.], [2., 1.]], cfg)


@pytest.mark.parametrize("options", [dict(backend="gpu"), dict(backend="hybrid"),
                                    dict(solver="stochastic-kpm", kernel="jackson")])
def test_full_cholesky_is_an_explicit_cpu_exact_option(options):
    with pytest.raises(ValueError, match="CPU EXACT"):
        RunConfig([0], [0], preconditioner="full-cholesky", **options)


def test_full_cholesky_exact_transport_and_factor_memory_budget():
    from dataclasses import replace
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.solvers.exact import iter_exact_blocks
    S = np.array([[1., .2j], [-.2j, 1.3]])
    H = np.array([[.2, .1], [.1, -.5]])
    X = np.array([[0., 1j], [-1j, 2.]])
    model = ModelBundle(H, S, position={"x": X})
    cfg = RunConfig([0.], [0., .1], axes=("x",), metric_rtol=1e-12)
    reference = list(iter_exact_blocks(model, build_operators(model, cfg), cfg))[0]
    full = replace(cfg, preconditioner="full-cholesky")
    result = list(iter_exact_blocks(model, build_operators(model, full), full))[0]
    assert_allclose(result.arrays["msd_A2/pair_analytic"],
                    reference.arrays["msd_A2/pair_analytic"], rtol=1e-12, atol=1e-14)
    extra = result.metadata["memory_estimate_bytes"] - reference.metadata["memory_estimate_bytes"]
    assert extra >= S.size*np.dtype(np.complex128).itemsize
    budget = (result.metadata["memory_estimate_bytes"] + reference.metadata["memory_estimate_bytes"])/2/1024**3
    limited = replace(full, memory_limit_gb=budget)
    with pytest.raises(ValueError, match="EXACT working-memory"):
        list(iter_exact_blocks(model, build_operators(model, limited), limited))
