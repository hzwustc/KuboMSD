from dataclasses import replace
import numpy as np
import scipy.sparse as sp
import pytest
from numpy.testing import assert_allclose
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.operators.velocity import build_operators
from kubo_msd.solvers.exact import iter_exact_blocks


@pytest.mark.parametrize('kind',['velocity','position','composite'])
@pytest.mark.parametrize('complex_values',[False,True])
def test_dense_projection_matches_sparse_all_states_and_custom_axis(kind,complex_values):
    rng=np.random.default_rng(126)
    def matrix():
        return rng.normal(size=(9,9))+(1j*rng.normal(size=(9,9)) if complex_values else 0)
    a=matrix();H=(a+a.conj().T)/2
    a=matrix();S=np.eye(9)+a.conj().T@a
    parts={}
    for axis in 'xy':
        x=matrix()
        if kind=='composite':
            b=matrix();parts[axis]=(b-b.conj().T,x,x.conj().T)
            if not complex_values:parts[axis]=tuple(sp.csr_matrix(v) for v in parts[axis])
        else:parts[axis]=x+x.conj().T
    model=ModelBundle(H,S,**{kind:parts})
    cfg=RunConfig([-.2,.1],[0,.01,.1],axes=('u',),axes_vectors=[[.6,.8,0]],
                  metric_rtol=1e-12,pair_block=4)
    sparse=list(iter_exact_blocks(model,build_operators(model,cfg),cfg))[0]
    dense_cfg=replace(cfg,exact_projection_kernel='dense')
    ops=build_operators(model,dense_cfg)
    dense=list(iter_exact_blocks(model,ops,dense_cfg))[0]
    for key in sparse.arrays:
        assert_allclose(dense.arrays[key],sparse.arrays[key],rtol=2e-11,atol=1e-12,equal_nan=True)
    count=6 if kind=='composite' else 2
    assert dense.metadata['memory_estimate_bytes']-sparse.metadata['memory_estimate_bytes'] >= count*9*9*16
    budget=(dense.metadata['memory_estimate_bytes']+sparse.metadata['memory_estimate_bytes'])/2/2**30
    with pytest.raises(ValueError,match='EXACT working-memory'):
        list(iter_exact_blocks(model,ops,replace(dense_cfg,memory_limit_gb=budget)))


@pytest.mark.parametrize('kw',[{'backend':'gpu'},{'backend':'hybrid'},{'solver':'stochastic-kpm','kernel':'jackson'}])
def test_dense_projection_cpu_exact_only(kw):
    with pytest.raises(ValueError,match='CPU EXACT'):
        RunConfig([0],[0],exact_projection_kernel='dense',**kw)
