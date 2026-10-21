import numpy as np
from numpy.testing import assert_allclose

def test_ratio_uses_sums_and_actual_unequal_counts():
    from kubo_msd.observables.statistics import estimate_ratio
    num=np.array([[2.],[90.]]); dos=np.array([[1.],[30.]]); counts=np.array([1,3])
    value,stderr,status=estimate_ratio(num,dos,counts)
    assert_allclose(value,[92/31]); assert stderr is not None
    assert status['reliable_interval'] is False

def test_correlated_numerator_denominator_errors_cancel():
    from kubo_msd.observables.statistics import estimate_ratio
    dos=np.arange(1.,9.)[:,None]
    value,stderr,status=estimate_ratio(3*dos,dos,np.ones(8,int))
    assert_allclose(value,3); assert_allclose(stderr,0,atol=1e-13)
    assert status['reliable_interval']

def test_one_block_no_fake_zero_stderr():
    from kubo_msd.observables.statistics import estimate_ratio
    _,stderr,status=estimate_ratio(np.ones((1,2)),np.ones((1,2)),[3])
    assert stderr is None and not status['reliable_interval']
