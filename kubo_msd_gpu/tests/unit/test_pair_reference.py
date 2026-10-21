import numpy as np
from numpy.testing import assert_allclose
from kubo_msd.operators.lcao_integrals import OrbitalSet
from kubo_msd.units import BOHR_A

def gaussian(alpha=0.5, lmax=0, element='X'):
    from scipy.special import gamma
    r=np.linspace(0,12,1201)
    radial=[]
    for l in range(lmax+1):
        c=np.sqrt(2*(2*alpha)**(l+1.5)/gamma(l+1.5))
        radial.append((l,0,c*r**l*np.exp(-alpha*r*r)))
    return OrbitalSet(element,r,radial,[1.]*(lmax+1),{})

def test_reference_gaussian_and_unconverged():
    from kubo_msd.operators.reference import evaluate_reference_pair
    a,b=gaussian(.5),gaussian(.8)
    d=np.array([.41,-.29,.18])
    s,D,report=evaluate_reference_pair(a,b,d,{'tolerance':1e-6})
    expected=(2*np.sqrt(.5*.8)/1.3)**1.5*np.exp(-.5*.8/1.3*np.dot(d,d)/BOHR_A**2)
    assert_allclose(s,[[expected]],atol=2e-8)
    assert_allclose(D[:,0,0],.8/1.3*d*expected,atol=2e-8)
    assert report['status']=='PASS'
    _,_,bad=evaluate_reference_pair(a,b,d,{'max_refinements':0})
    assert bad['status']=='REFERENCE_UNCONVERGED'

def test_reference_exchange_and_onsite_sp():
    from kubo_msd.operators.reference import evaluate_reference_pair
    a=gaussian(lmax=1); d=np.array([.3,.2,-.5])
    s,D,_=evaluate_reference_pair(a,a,d,{})
    st,Dt,_=evaluate_reference_pair(a,a,-d,{})
    assert_allclose(st,s.T,atol=2e-7)
    assert_allclose(Dt,D.transpose(0,2,1)-d[:,None,None]*s.T,atol=2e-7)
    s,D,_=evaluate_reference_pair(a,a,[0,0,0],{})
    assert abs(D[2,0,1]) > .3
