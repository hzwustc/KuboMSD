"""Eigenpair identities must agree with an independent dense metric inverse."""
import numpy as np
import scipy.linalg as la
import pytest
from numpy.testing import assert_allclose
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.operators.velocity import build_operators
from kubo_msd.units import HBAR_EV_FS


@pytest.mark.parametrize("kind", ["velocity", "position", "composite"])
def test_projected_velocity_matches_dense_metric_equation_without_solves(kind, monkeypatch):
    from kubo_msd.solvers.exact import eigen_velocity_block

    rng = np.random.default_rng(731)
    n = 7
    def herm():
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return a + a.conj().T
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    S = a.conj().T @ a + np.eye(n)
    H = herm()
    parts = {}
    velocities = {}
    for axis in ("x", "y"):
        X = herm()
        if kind == "velocity":
            parts[axis] = X
            velocities[axis] = X
        elif kind == "position":
            parts[axis] = X
            velocities[axis] = 1j * (H @ la.solve(S, X) - X @ la.solve(S, H)) / HBAR_EV_FS
        else:
            # Arbitrary anti-Hermitian bond and delta=xbar^H are more general
            # than a finite position commutator, and preserve periodic physics.
            bond = 1j * herm()
            xbar = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            delta = xbar.conj().T
            parts[axis] = (bond, xbar, delta)
            velocities[axis] = 1j * (bond - xbar @ la.solve(S, H)
                                      + H @ la.solve(S, delta)) / HBAR_EV_FS
    cfg = RunConfig([0.], [0.], axes=("u",), axes_vectors=[[.6, .8, 0.]],
                    metric_rtol=1e-13)
    model = ModelBundle(H, S, **{kind: parts})
    ops = build_operators(model, cfg)
    eps, C = la.eigh(H, S)
    left, right = np.array([0, 3, 6]), np.array([1, 2, 4, 5])
    expected = C[:, left].conj().T @ (.6 * velocities["x"] + .8 * velocities["y"]) @ C[:, right]
    def forbidden(*args, **kwargs):
        raise AssertionError("EXACT eigenpair projection must not iterate on S")
    monkeypatch.setattr(ops, "solve_s", forbidden)
    actual = eigen_velocity_block(ops, "u", C[:, left], C[:, right], eps[left], eps[right])
    assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
