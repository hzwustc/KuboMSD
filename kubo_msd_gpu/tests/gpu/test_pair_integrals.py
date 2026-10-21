from pathlib import Path
import numpy as np
import pytest
from kubo_msd.backends.gpu_context import create_gpu_context
from kubo_msd.operators.gpu.contracts import GpuBuildOptions
from gpu_fixtures import parsed,batch,mixed_orbitals


def test_off_node_spdf_pairs():
    from kubo_msd.operators.gpu.tables import build_tables_gpu
    from kubo_msd.operators.gpu.pairs import evaluate_pairs_gpu
    p=parsed();ctx=create_gpu_context(0,1024**3); t=build_tables_gpu(p,GpuBuildOptions(),ctx)
    ref=np.load(Path(__file__).parents[1]/'data/cpu_spdf_tables.npz')
    b=batch(ref['displacements'],p.orbitals)
    got=evaluate_pairs_gpu(t,b,ctx)
    np.testing.assert_allclose(ctx.to_host(got.overlap),ref['S'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(ctx.to_host(got.dipole),ref['D'],rtol=1e-10,atol=1e-12)
    # Same mathematics across legal batches, exact cutoff and nonzero onsite s-p.
    split=[evaluate_pairs_gpu(t,b.batch(i,i+1),ctx) for i in range(len(b.pair_ids))]
    np.testing.assert_allclose(ctx.to_host(ctx.xp.concatenate([a.overlap for a in split])),ref['S'],rtol=1e-12,atol=1e-13)
    assert abs(ctx.to_host(got.dipole)[2,1])>.3
    assert np.max(abs(ctx.to_host(got.overlap)[-256:]))==0


def test_off_node_mixed_zeta_spdf_pairs():
    from kubo_msd.operators.gpu.tables import build_tables_gpu
    from kubo_msd.operators.gpu.pairs import evaluate_pairs_gpu
    p=parsed(mixed_orbitals()); ctx=create_gpu_context(0,1024**3)
    ref=np.load(Path(__file__).parents[1]/'data/cpu_mixed_pairs.npz')
    t=build_tables_gpu(p,GpuBuildOptions(),ctx)
    b=batch(ref['displacements'],p.orbitals,ref['species'])
    got=evaluate_pairs_gpu(t,b,ctx)
    S=ctx.to_host(got.overlap);D=ctx.to_host(got.dipole)
    np.testing.assert_allclose(S,ref['S'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(D,ref['D'],rtol=1e-10,atol=1e-12)
    # First two pairs exchange centers, preserving the non-collinear direction.
    a,c=tuple(b.shapes[0])
    s=S[b.offsets[0]:b.offsets[1]].reshape(a,c)
    sr=S[b.offsets[1]:b.offsets[2]].reshape(c,a).T
    d=D[:,b.offsets[0]:b.offsets[1]].reshape(3,a,c)
    dr=D[:,b.offsets[1]:b.offsets[2]].reshape(3,c,a).transpose(0,2,1)
    np.testing.assert_allclose(sr,s,rtol=1e-10,atol=1e-12)
    # Center exchange is a physical identity of the continuous integrals. Separate
    # interpolants for S and D incur their finite table error in this identity.
    scale=max(1.,sum(o.cutoff_A for o in p.orbitals.values()))
    assert np.max(abs(dr-d+b.displacement_A[0,:,None,None]*s))/scale <= 1e-5/5
    assert a!=c


def test_duplicate_pairs_fail():
    from kubo_msd.operators.gpu.tables import build_tables_gpu
    from kubo_msd.operators.gpu.pairs import evaluate_pairs_gpu
    p=parsed(); ctx=create_gpu_context(0,1024**3);t=build_tables_gpu(p,GpuBuildOptions(),ctx)
    b=batch([[0,0,0],[1,0,0]],p.orbitals);b.pair_ids[:]=1
    with pytest.raises(ValueError,match='unique'): evaluate_pairs_gpu(t,b,ctx)
