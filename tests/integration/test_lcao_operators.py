import numpy as np
from numpy.testing import assert_allclose

def make_orb(path):
    # Normalized isotropic Gaussian exp(-r^2/2): radial R = 2/pi^(1/4) exp(-r^2/2).
    r=np.arange(1001)*.01
    path.write_text('Element X\nRadius Cutoff(a.u.) 10\nLmax 0\nNumber of Sorbital--> 1\nSUMMARY END\nMesh 1001\ndr 0.01\nType L N\n0 0 0\n'+'\n'.join(map(str,2/np.pi**.25*np.exp(-r*r/2))))

def test_two_center_gaussian_overlap_dipole_translation(tmp_path):
    from kubo_msd.operators.lcao_integrals import read_orb, two_center
    from kubo_msd.units import BOHR_A
    make_orb(tmp_path/'x.orb'); orb=read_orb(tmp_path/'x.orb')
    displacement=np.array([.6,.2,.1])*BOHR_A
    expected=np.exp(-np.sum((displacement/BOHR_A)**2)/4)
    S,X,diag=two_center(orb,orb,displacement,{'radial_points':80,'polar_points':24,'azimuthal_points':48})
    assert_allclose(S,[[expected]],rtol=2e-8)
    assert_allclose(X[:,0,0],displacement/2*expected,rtol=2e-8,atol=1e-10)
    assert_allclose(orb.norms,[1],atol=1e-9)

def test_abacus_real_harmonics_phase_and_order():
    from kubo_msd.operators.lcao_integrals import real_harmonics
    v=real_harmonics(1,np.eye(3)); a=np.sqrt(3/(4*np.pi))
    assert_allclose(v,[[0,-a,0],[0,0,-a],[a,0,0]],atol=1e-15)
