import numpy as np
from numpy.testing import assert_allclose
from test_pair_reference import gaussian
from kubo_msd.operators.reference import direct_pair

def test_aligned_reference_phi_is_exact_for_pdf():
    o=gaussian(lmax=3);d=[.28,-.41,.19]
    a=direct_pair(o,o,d,dict(radial_points=80,polar_points=32,azimuthal_points=16))
    b=direct_pair(o,o,d,dict(radial_points=80,polar_points=32,azimuthal_points=48))
    assert_allclose(a[0],b[0],atol=2e-14);assert_allclose(a[1],b[1],atol=2e-14)
