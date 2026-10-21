import os
import numpy as np
import pytest
from dataclasses import replace
from numpy.testing import assert_allclose
from test_build_mpi import launch_build
from kubo_msd.config import RunConfig
from kubo_msd.dispatch import run_input
from kubo_msd.io.results import read_results

pytestmark=[pytest.mark.gpu,pytest.mark.mpi]

def test_mpi_cache_consumed_after_launcher_exits(tmp_path):
    if os.environ.get('KUBOMSD_TEST_GPU_MPI')!='1':pytest.skip('set KUBOMSD_TEST_GPU_MPI=1')
    p=launch_build(tmp_path/'build',3,2)
    assert p.returncode==0,p.stdout+p.stderr
    for solver in ('exact','stochastic-kpm'):
        cfg=RunConfig(np.linspace(-2,2,9),[0.,.01,.02],axes=('x','y','z'),solver=solver,kernel='jackson',spectral_bounds=(-4,4),n_moments=16,nsamples=16,statistics_block=2,metric_rtol=1e-12)
        for backend in ('cpu','gpu'):
            run_input(tmp_path/'build/operators/model.json',replace(cfg,backend=backend),tmp_path/(solver+backend))
        cpu,gpu=[read_results(tmp_path/(solver+b)) for b in ('cpu','gpu')]
        for k,v in cpu['arrays'].items():assert_allclose(gpu['arrays'][k],v,rtol=1e-9,atol=1e-11,equal_nan=True,err_msg=k)
