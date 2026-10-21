import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_pair_reference import gaussian
from kubo_msd.units import BOHR_A


def tables(orbitals):
    from kubo_msd.operators.tables import prepare_tables
    return prepare_tables(orbitals,tolerance=1e-5)

def evaluate(t,a,b,d,threads=1):
    from kubo_msd import _operators_native as native
    return native.evaluate_pairs(t,dict(species=np.array([[a,b]],np.int64),displacement_A=np.array([d],float),pair_ids=np.array([9],np.int64)),threads)

def test_fast_gaussian_unequal_and_cutoff():
    a,b=gaussian(.5),gaussian(.8,element='B'); t=tables({'A':a,'B':b})
    d=np.array([.413,-.291,.187]); got=evaluate(t,0,1,d)
    s=(2*np.sqrt(.5*.8)/1.3)**1.5*np.exp(-.5*.8/1.3*np.dot(d,d)/BOHR_A**2)
    assert_allclose(got['S'],[s],atol=2e-7)
    assert_allclose(got['D'][:,0],.8/1.3*d*s,atol=2e-7)
    assert_allclose(evaluate(t,0,1,[30,0,0])['S'],0,atol=0)
    assert got['pair_ids'].tolist()==[9]

def test_fast_pdf_realspace_exchange_and_onsite():
    from kubo_msd.operators.reference import evaluate_reference_pair
    a=gaussian(lmax=3); t=tables({'X':a}); d=np.array([.21,-.32,.17])
    got=evaluate(t,0,0,d); n=a.n_orbitals
    s,D,r=evaluate_reference_pair(a,a,d,{'tolerance':1e-6})
    assert r['status']=='PASS'
    assert_allclose(got['S'].reshape(n,n),s,atol=2e-7)
    assert_allclose(got['D'].reshape(3,n,n),D,atol=2e-7)
    rev=evaluate(t,0,0,-d)
    assert_allclose(rev['S'].reshape(n,n),s.T,atol=2e-7)
    assert_allclose(rev['D'].reshape(3,n,n),D.transpose(0,2,1)-d[:,None,None]*s.T,atol=2e-7)
    z=evaluate(t,0,0,[0,0,0]); assert abs(z['D'].reshape(3,n,n)[2,0,1])>.3

def test_parallel_and_payload_rejections():
    t=tables({'X':gaussian()})
    from kubo_msd import _operators_native as native
    p=dict(species=np.zeros((9,2),np.int64),displacement_A=np.random.default_rng(7).normal(size=(9,3)),pair_ids=np.arange(9,dtype=np.int64))
    one=native.evaluate_pairs(t,p,1)
    for nt in (2,4):
        other=native.evaluate_pairs(t,p,nt)
        assert_allclose(one['D'],other['D'],rtol=1e-12,atol=1e-13)
    p['species'][0,0]=10
    with pytest.raises((ValueError,RuntimeError)): native.evaluate_pairs(t,p,2)


def test_finite_support_onsite_uses_exact_radial_product_rule():
    # Truncating at 3 bohr deliberately leaves a nonzero boundary value. A
    # finite-k transform must not create an anti-Hermitian onsite dipole.
    from kubo_msd.operators.lcao_integrals import OrbitalSet
    from scipy.integrate import quad
    from scipy.interpolate import CubicSpline
    g=gaussian(lmax=3)
    o=OrbitalSet('X',g.radius_bohr[:301],[(l,z,f[:301]) for l,z,f in g.radial],[1.]*4,{})
    result=evaluate(tables({'X':o}),0,0,[0.,0.,0.]);n=o.n_orbitals
    s=result['S'].reshape(n,n);D=result['D'].reshape(3,n,n)
    assert_allclose(D,D.transpose(0,2,1),rtol=1e-13,atol=1e-14)
    f=CubicSpline(o.radius_bohr,o.radial[0][2])
    expected=quad(lambda r:r*r*f(r)**2,0,3,epsabs=1e-13,epsrel=1e-13,
                  points=o.radius_bohr,limit=1000)[0]
    assert_allclose(s[0,0],expected,rtol=1e-12,atol=1e-13)
