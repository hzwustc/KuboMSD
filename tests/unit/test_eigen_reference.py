import numpy as np
import pytest


def test_full_eigen_reference_rejects_changed_vectors_and_eigenvalues():
    from benchmarks.metric_eigen_reference import verify_eigenpairs
    from kubo_msd.contracts import ModelBundle
    import scipy.sparse as sp
    model=ModelBundle(sp.diags([2.,6]),sp.diags([2.,3]))
    C=np.diag([1/np.sqrt(2),1/np.sqrt(3)])
    report=verify_eigenpairs(model,np.array([1.,2]),C)
    assert report["status"]=="PASS"
    for eps,vec in [(np.array([1.,3]),C),(np.array([1.,2]),C*1.1)]:
        with pytest.raises(ValueError):
            verify_eigenpairs(model,eps,vec)


def test_comparison_masks_are_exact_and_do_not_use_boolean_subtraction():
    from benchmarks.metric_eigen_reference import differences
    checks=differences({"mask":np.array([True,False])},{"mask":np.array([True,True])})
    assert checks["mask"]["passed"] is False
    assert checks["mask"]["max_absolute_difference"]==1

