import numpy as np
import scipy.linalg as la
import pytest
from numpy.testing import assert_allclose
from kubo_msd.solvers.exact import solve_generalized


@pytest.mark.parametrize("backend", ["cpu", "hybrid"])
@pytest.mark.parametrize("imaginary", [0., 1e-30, .1])
def test_scipy_preserves_exact_real_or_complex_problem(backend, imaginary, monkeypatch):
    H = np.array([[.1, .2+1j*imaginary], [.2-1j*imaginary, .8]])
    S = np.array([[1., .2], [.2, 1.4]], complex)
    original = la.eigh
    expected = original(H, S, driver="gvd")[0]
    calls = []
    def observed(a, b, **kwargs):
        calls.append((a.dtype.kind, b.dtype.kind))
        assert a.dtype.kind == b.dtype.kind == ("f" if imaginary == 0 else "c")
        assert_allclose(a, H, atol=0, rtol=0)
        assert_allclose(b, S, atol=0, rtol=0)
        return original(a, b, **kwargs)
    monkeypatch.setattr(la, "eigh", observed)
    eps, C, report = solve_generalized(H, S, backend)
    assert len(calls) == 1
    assert_allclose(eps, expected, rtol=1e-12, atol=1e-14)
    assert_allclose(C.conj().T@S@C, np.eye(2), atol=1e-13)
    assert report["max_normalized_residual"] < 1e-12


@pytest.mark.parametrize("imaginary", [1e-30, .1])
def test_complex_overlap_keeps_complex_scipy_problem(imaginary, monkeypatch):
    H = np.diag([.2, .7])
    S = np.array([[1., .1+1j*imaginary], [.1-1j*imaginary, 1.4]])
    original = la.eigh
    def observed(a, b, **kwargs):
        assert a.dtype.kind == b.dtype.kind == "c"
        assert b[0, 1].imag == imaginary
        return original(a, b, **kwargs)
    monkeypatch.setattr(la, "eigh", observed)
    eps, C, report = solve_generalized(H, S)
    assert_allclose(H@C, (S@C)*eps, atol=1e-13)
