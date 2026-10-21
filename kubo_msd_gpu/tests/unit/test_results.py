import json
import numpy as np
import pytest
from numpy.testing import assert_allclose

def test_partial_failure_complete_roundtrip_and_signed_values(tmp_path):
    from kubo_msd.io.results import ResultsWriter,read_results
    from kubo_msd.config import RunConfig
    from kubo_msd.contracts import TransportBlock,RunSummary
    cfg=RunConfig(energy_eV=[0,1],time_fs=[0,1,2],axes=('x',),output_npz=True)
    msd=np.array([[[0.,0],[1,2],[.1,1.]]]); D=np.array([[[0.,0],[1,2],[-.5,-1.]]])
    for mode in ('failure','complete'):
        out=tmp_path/mode
        try:
            with ResultsWriter(out,cfg) as w:
                w.append(TransportBlock(0,cfg.time_fs,{'msd_A2/pair_analytic':msd,'diffusion_A2_per_fs/pair_analytic':D,'numerator_msd_A2/pair_analytic':msd*2,'valid/pair_analytic':np.ones_like(msd,bool),'reason_code/pair_analytic':np.zeros_like(msd,dtype='uint32')},{'dos_states_per_eV':np.array([2.,2.])},{'primary_estimator':'pair_analytic','test_nan':float('nan')}))
                if mode=='failure': raise RuntimeError('injected interruption')
                w.finalize(RunSummary(str(out),'complete','exact','cpu'))
        except RuntimeError: pass
        if mode=='failure':
            with pytest.raises(ValueError): read_results(out)
            assert read_results(out,allow_partial=True)['metadata']['run_status']=='failed'
        else:
            r=read_results(out); assert_allclose(r['arrays']['diffusion_x_A2_per_fs'],D[0]); assert r['metadata']['test_nan'] is None
            with np.load(out/'results.npz',allow_pickle=False) as z: assert_allclose(z['diffusion_x_A2_per_fs'],D[0])
        with pytest.raises(FileExistsError): ResultsWriter(out,cfg)

def test_noncontiguous_append_does_not_complete(tmp_path):
    from kubo_msd.io.results import ResultsWriter
    from kubo_msd.config import RunConfig
    from kubo_msd.contracts import TransportBlock,RunSummary
    cfg=RunConfig(energy_eV=[0],time_fs=[0,1],axes=('x',))
    with pytest.raises(ValueError):
        with ResultsWriter(tmp_path/'out',cfg) as w:
            w.append(TransportBlock(1,[1],{}))
