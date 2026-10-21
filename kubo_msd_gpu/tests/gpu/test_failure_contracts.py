import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from test_device_metric_setup import setup_model,runtime
from kubo_msd.config import RunConfig
from kubo_msd.gpu_cli import main


def test_gpu_failure_has_no_cpu_fallback(tmp_path,monkeypatch):
    import kubo_msd.gpu_dispatch as dispatch
    def fail(*a,**k):raise RuntimeError('injected CUDA capability failure')
    monkeypatch.setattr(dispatch,'create_gpu_context',fail)
    out=tmp_path/'out';path=Path(__file__).parents[2]/'examples/two_level/model.json'
    assert main(['run','--input',str(path),'--out',str(out),'--emin','-1','--emax','1'])==2
    assert not out.exists()


def test_wrong_device_rejected():
    ctx,model,_=setup_model(2)
    with pytest.raises(ValueError,match='device'): replace(model,device_id=ctx.device_id+1)


def test_cache_atomic_failure_and_corruption(tmp_path,monkeypatch):
    import kubo_msd.operators.gpu.cache as cache
    ctx,model,_=setup_model(2);original=cache.os.rename
    def fail(*a,**k):raise OSError('injected publish failure')
    monkeypatch.setattr(cache.os,'rename',fail)
    with pytest.raises(OSError): cache.save_operator_cache(model,tmp_path/'bad',ctx)
    assert not (tmp_path/'bad').exists() and not (tmp_path/'bad.lock').exists()
    monkeypatch.setattr(cache.os,'rename',original)
    cache.save_operator_cache(model,tmp_path/'good',ctx)
    p=tmp_path/'good/model.json';d=json.loads(p.read_text());d['n_orbitals']=999;p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='checksum'):cache.load_operator_cache_gpu(p,ctx)


def test_device_unrepresentable_solution_fails():
    from kubo_msd.metric.gpu_setup import DeviceMetricSolver
    ctx,model,_=setup_model(2)
    cfg=RunConfig([0],[0],backend='gpu')
    m=DeviceMetricSolver(ctx.sparse.eye(2,dtype=ctx.xp.complex128,format='csr')*.25,cfg,ctx)
    with pytest.raises(ValueError,match='representable'):m.solve_s(ctx.asarray(np.array([[1e308],[1.]],complex)))


def test_result_write_failure_not_complete(tmp_path,monkeypatch):
    from kubo_msd.io.results import ResultsWriter
    def fail(self,block):raise OSError('injected output error')
    monkeypatch.setattr(ResultsWriter,'append',fail)
    out=tmp_path/'out';path=Path(__file__).parents[2]/'examples/two_level/model.json'
    assert main(['run','--input',str(path),'--out',str(out),'--emin','-1','--emax','1','--n-energy','3','--tmax-fs','.01'])==2
    assert not (out/'results.h5').exists()
    assert json.loads((out/'metadata.json').read_text())['run_status']=='failed'


def test_late_metadata_failure_not_complete(tmp_path,monkeypatch):
    from kubo_msd.io.results import ResultsWriter
    original=ResultsWriter._save_meta
    def fail(self):
        if self.done:raise OSError('late metadata failure')
        return original(self)
    monkeypatch.setattr(ResultsWriter,'_save_meta',fail)
    out=tmp_path/'out';path=Path(__file__).parents[2]/'examples/two_level/model.json'
    result=main(['run','--input',str(path),'--out',str(out),'--emin','-1','--emax','1','--n-energy','3','--tmax-fs','.01'])
    # There should be no extra fallible metadata write after successful finalize.
    assert result==0
    assert json.loads((out/'metadata.json').read_text())['run_status']=='complete'


@pytest.mark.parametrize('message',['CUDA out of memory','NVRTC compilation failure'])
def test_gpu_build_failures_publish_nothing(tmp_path,monkeypatch,message):
    from kubo_msd.backends.gpu_context import GPUContext
    def fail(*args,**kwargs):raise RuntimeError(message)
    monkeypatch.setattr(GPUContext,'kernel',fail)
    out=tmp_path/'bad-cache'
    path=Path(__file__).parents[2]/'examples/raw_lcao/input.json'
    assert main(['prepare','--input',str(path),'--out',str(out),'--mode','build'])==2
    assert not out.exists()
