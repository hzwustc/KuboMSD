import numpy as np
from numpy.testing import assert_allclose

def test_inverse_sqrt_transform_complete_basis_and_sample_ids():
    from kubo_msd.metric.solve import MetricSolver
    from kubo_msd.metric.sampling import MetricSampler, random_phases
    from kubo_msd.config import RunConfig
    S=np.array([[1,.2j],[-.2j,2]])
    cfg=RunConfig(energy_eV=[0],time_fs=[0],sampling_tol=1e-12)
    ms=MetricSolver(S,cfg); sampler=MetricSampler(ms,cfg)
    C=sampler.transform(np.eye(2,dtype=complex))
    assert_allclose(C@C.conj().T,np.linalg.inv(S),rtol=1e-11,atol=1e-12)
    whole=random_phases(2,7,[0,1,2,3]); parts=np.column_stack([random_phases(2,7,[j]) for j in range(4)])
    assert_allclose(whole,parts,atol=0,rtol=0)
