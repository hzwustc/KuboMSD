import numpy as np
import pytest
from numpy.testing import assert_allclose
from kubo_msd.operators.lcao_integrals import OrbitalSet


def fixture():
    r=np.linspace(0,4,401); a=2/np.pi**.25*np.exp(-r*r/2)
    orb=OrbitalSet('X',r,[(0,0,a)],[1.],{})
    structure=dict(cell_A=np.array([[4.,0.,0.],[.9,4.,0.],[.2,.3,4.]]),positions_A=np.array([[0.,0.,0.],[1.2,.3,.2]]),symbols=['X','X'])
    h={(0,0,0,0,0):np.array([[-1.]]),(1,1,0,0,0):np.array([[1.]]),(0,1,0,0,0):np.array([[.2]])}
    return structure,{'X':orb},h


def test_complete_build_matches_reference_and_order():
    from kubo_msd.operators.builder import BuilderOptions,build_operators
    s,o,h=fixture()
    fast=build_operators(s,o,h,BuilderOptions(operator_builder='cpp'))
    ref=build_operators(s,o,h,BuilderOptions(operator_builder='reference'))
    assert fast.orbital_order==ref.orbital_order
    assert fast.diagnostics['builder']=='cpp'
    for a,b in [(fast.H_cov,ref.H_cov),(fast.S,ref.S)]+[(a,b) for q in 'xyz' for a,b in zip(fast.composite[q],ref.composite[q])]:
        assert_allclose(a.toarray(),b.toarray(),atol=1e-5)
    assert fast.S.nnz > fast.H_cov.nnz-2
    assert np.linalg.eigvalsh(fast.S.toarray()).min()>0


def test_native_neighbors_equal_enumeration_unwrapped():
    from kubo_msd.operators.builder import pair_tasks
    from itertools import product
    s,o,h=fixture();s['positions_A'][1]+=s['cell_A'][0]*3
    keys,d,p=pair_tasks(s,o)
    expected=set()
    for i in range(2):
        for j in range(i,2):
            for R in product(range(-6,7),repeat=3):
                if i==j and R<(0,0,0):continue
                if np.linalg.norm(s['positions_A'][j]+np.array(R)@s['cell_A']-s['positions_A'][i])<2*o['X'].cutoff_A:
                    expected.add((i,j,*R))
    assert set(map(tuple,keys))==expected
    assert any(k[0]==k[1] and tuple(k[2:])!=(0,0,0) for k in keys)


def test_budget_and_unknown_options_rejected():
    from kubo_msd.operators.builder import BuilderOptions,build_operators
    s,o,h=fixture()
    with pytest.raises((ValueError,RuntimeError)):build_operators(s,o,h,BuilderOptions(memory_limit_bytes=32))
    with pytest.raises(TypeError):BuilderOptions(silent_fallback=True)
