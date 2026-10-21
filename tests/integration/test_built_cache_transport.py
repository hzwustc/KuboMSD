import numpy as np
from kubo_msd.dispatch import run_input
from kubo_msd.config import RunConfig
from kubo_msd.io.results import read_results


def test_built_cache_cpu_exact_and_stochastic(built_operator_cache,tmp_path):
    for solver in ('exact','stochastic-kpm'):
        cfg=RunConfig(np.linspace(-2,2,9),[0.,.01,.02],axes=('x','y','z'),solver=solver,kernel='jackson',spectral_bounds=(-4,4),n_moments=16,nsamples=16,statistics_block=2,metric_rtol=1e-12)
        run_input(built_operator_cache,cfg,tmp_path/solver)
        result=read_results(tmp_path/solver)
        assert result['metadata']['run_status']=='complete'
        assert any('msd' in k for k in result['arrays'])
