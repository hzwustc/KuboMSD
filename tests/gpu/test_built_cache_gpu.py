import numpy as np
import pytest
from dataclasses import replace
from numpy.testing import assert_allclose
from kubo_msd.config import RunConfig
from kubo_msd.dispatch import run_input
from kubo_msd.io.results import read_results

pytestmark=pytest.mark.gpu

@pytest.mark.parametrize('solver,backends',[('exact',['cpu','hybrid','gpu']),('stochastic-kpm',['cpu','gpu'])])
def test_one_built_cache_all_backends(built_operator_cache,tmp_path,solver,backends):
    cfg=RunConfig(np.linspace(-2,2,9),[0.,.01,.02],axes=('x','y','z'),solver=solver,kernel='jackson',spectral_bounds=(-4,4),n_moments=16,nsamples=16,statistics_block=2,metric_rtol=1e-12)
    results=[]
    for backend in backends:
        run_input(built_operator_cache,replace(cfg,backend=backend),tmp_path/backend)
        results.append(read_results(tmp_path/backend))
    for result in results[1:]:
        for key,value in results[0]['arrays'].items():
            assert_allclose(result['arrays'][key],value,rtol=1e-9,atol=1e-11,equal_nan=True,err_msg=key)
