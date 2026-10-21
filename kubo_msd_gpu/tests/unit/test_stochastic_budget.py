import numpy as np
import pytest
from kubo_msd.config import RunConfig
from kubo_msd.contracts import ModelBundle
from kubo_msd.operators.velocity import build_operators
from kubo_msd.solvers.stochastic import iter_stochastic_blocks


def test_bootstrap_workspace_is_included_in_root_budget():
    cfg = RunConfig(np.linspace(-1, 1, 1000), [0., .01],
                    solver="stochastic-kpm", kernel="jackson", cpu_kernel="reference",
                    axes=("x",), nsamples=2, statistics_block=1, sample_batch=1,
                    n_moments=4, include_vac=False, spectral_bounds=(-2., 2.),
                    memory_limit_gb=.005)
    model = ModelBundle(np.array([[0., .5], [.5, 0.]]), np.eye(2),
                        position={"x": np.diag([0., 1.])})
    ops = build_operators(model, cfg)
    with pytest.raises(ValueError, match="bootstrap"):
        list(iter_stochastic_blocks(ops, cfg, "stepwise"))
