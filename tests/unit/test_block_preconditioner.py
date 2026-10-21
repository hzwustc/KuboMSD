import numpy as np
import scipy.linalg as la
from numpy.testing import assert_allclose
from kubo_msd.config import RunConfig
from kubo_msd.metric.solve import MetricSolver


def test_cpu_block_jacobi_reuses_cholesky_without_generic_refactorization(monkeypatch):
    rng = np.random.default_rng(12)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    S = a.conj().T @ a + np.eye(8)
    rhs = rng.normal(size=(8, 3)) + 1j * rng.normal(size=(8, 3))
    cfg = RunConfig([0.], [0.], preconditioner="block-jacobi")
    metric = MetricSolver(S, cfg, atom_slices=[(0, 3), (3, 8)])
    expected = np.concatenate([la.solve(S[:3, :3], rhs[:3]), la.solve(S[3:, 3:], rhs[3:])])
    def forbidden(*args, **kwargs):
        raise AssertionError("a saved Cholesky factor must not be refactorized by generic solve")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    assert_allclose(metric.precondition(rhs), expected, rtol=1e-13, atol=1e-13)
