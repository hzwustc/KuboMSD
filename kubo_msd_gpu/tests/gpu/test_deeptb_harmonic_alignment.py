"""The dftio H basis is m=-l..l without Condon-Shortley signs."""
import json
from pathlib import Path
import h5py
import numpy as np
import scipy.linalg as la
import pytest
from gpu_fixtures import gaussian
from kubo_msd.backends.gpu_context import create_gpu_context
from kubo_msd.operators.gpu.input import parse_build_input
from kubo_msd.operators.gpu.builder import build_gpu_operators
from kubo_msd.operators.gpu.contracts import GpuBuildOptions
from kubo_msd.units import BOHR_A


def test_deeptb_h_and_spdf_integrals_share_basis(tmp_path):
    orbital = gaussian()
    text = f"Element X\nMesh {len(orbital.radius_bohr)}\ndr 0.02\n"
    for l, z, values in orbital.radial:
        text += f"Type L N\n0 {l} {z}\n" + "\n".join(map(str, values)) + "\n"
    (tmp_path/"X.orb").write_text(text)
    displacement = np.array([2.1, -1.2, 1.4])
    (tmp_path/"POSCAR").write_text(
        "Gaussian spdf alignment fixture\n1\n20 0 0\n0 20 0\n0 0 20\nX\n2\nCartesian\n"
        "0 0 0\n2.1 -1.2 1.4\n")
    n = orbital.n_orbitals
    rng = np.random.default_rng(710)
    h = rng.normal(size=(2*n, 2*n))
    h = h+h.T
    with h5py.File(tmp_path/"hamiltonians.h5", "w") as f:
        g = f.create_group("0")
        for i in range(2):
            for j in range(2):
                g.create_dataset(f"{i}_{j}_0_0_0", data=h[i*n:(i+1)*n, j*n:(j+1)*n])
    data = {"version": 1, "structure": {"file": "POSCAR"},
            "model": {"type": "lcao", "hamiltonian": {"path": "hamiltonians.h5", "format": "deeptb"},
                      "orbitals": {"dir": ".", "names": ["X.orb"]}}}
    path = tmp_path/"input.json"
    path.write_text(json.dumps(data))
    ctx = create_gpu_context(0, 1024**3)
    options = GpuBuildOptions(table_cache_dir=tmp_path/"tables")
    target = build_gpu_operators(parse_build_input(path), options, ctx).model
    s = ctx.to_host(target.S.toarray())
    # Analytic normalized same-exponent Gaussian s-p overlap, p=(y,z,x).
    d = displacement/BOHR_A
    expected = -np.sqrt(.5)*d[[1, 2, 0]]*np.exp(-.5*np.dot(d, d)/2)
    np.testing.assert_allclose(s[0, n+1:n+4].real, expected, rtol=2e-5, atol=1e-8)
    np.testing.assert_array_equal(ctx.to_host(target.H_cov.toarray()), h)
    # Independently specified signed permutations also cover d and f, and all
    # three Cartesian dipoles. H and its bond factors stay in the input basis.
    data["model"]["hamiltonian"]["orbital_convention"] = "abacus-real"
    path.write_text(json.dumps(data))
    source = build_gpu_operators(parse_build_input(path), options, ctx).model
    blocks = [
        np.eye(1),
        np.diag([-1, 1, -1])@np.eye(3)[[2, 0, 1]],
        np.diag([1, -1, 1, -1, 1])@np.eye(5)[[4, 2, 0, 1, 3]],
        np.diag([-1, 1, -1, 1, -1, 1, -1])@np.eye(7)[[6, 4, 2, 0, 1, 3, 5]],
    ]
    u = la.block_diag(*blocks, *blocks)
    np.testing.assert_allclose(s, u@ctx.to_host(source.S.toarray())@u.T, atol=1e-12, rtol=1e-10)
    for axis in "xyz":
        np.testing.assert_array_equal(ctx.to_host(target.composite[axis][0].toarray()),
                                      ctx.to_host(source.composite[axis][0].toarray()))
        for k in (1, 2):
            np.testing.assert_allclose(ctx.to_host(target.composite[axis][k].toarray()),
                u@ctx.to_host(source.composite[axis][k].toarray())@u.T, atol=1e-12, rtol=1e-10)
    assert target.cache_identity != source.cache_identity


def test_pre_alignment_raw_gpu_cache_requires_rebuild(tmp_path):
    from dataclasses import replace
    from test_device_metric_setup import setup_model
    from kubo_msd.operators.gpu.cache import save_operator_cache, load_operator_cache_gpu
    ctx, model, _ = setup_model(3)
    old = replace(model, provenance={"input_format": "legacy-lcao-deeptb", "builder_used": "gpu"})
    save_operator_cache(old, tmp_path/"old", ctx)
    with pytest.raises(ValueError, match="basis.*rebuild"):
        load_operator_cache_gpu(tmp_path/"old", ctx)
