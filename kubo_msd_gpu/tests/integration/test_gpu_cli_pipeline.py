from pathlib import Path
import json
import pytest
from kubo_msd.gpu_cli import main

@pytest.mark.parametrize('solver',['exact','stochastic-kpm'])
def test_run_native_pipeline(tmp_path,solver):
    inp=Path(__file__).parents[2]/'examples/nonorthogonal/model.json'
    out=tmp_path/solver
    args=['run','--input',str(inp),'--out',str(out),'--solver',solver,'--kernel','jackson',
          '--spectral-min','-4','--spectral-max','4','--emin','-.2','--emax','.2','--n-energy','5',
          '--tmax-fs','.05','--dt-fs','.03','--n-moments','12','--nsamples','3','--statistics-block','2','--no-vac']
    assert main(args)==0
    meta=json.loads((out/'metadata.json').read_text())
    assert meta['run_status']=='complete' and meta['builder_used']=='cache'
    assert meta['gpu_pipeline']['capabilities']['float64_complex128_smoke']=='PASS'


def test_prepare_audit_and_pilot_no_ready(tmp_path):
    inp=Path(__file__).parents[2]/'examples/raw_lcao/input.json'
    for mode in ('audit','pilot'):
        out=tmp_path/mode
        assert main(['prepare','--input',str(inp),'--out',str(out),'--mode',mode])==0
        assert not (out/'CACHE_READY').exists()
        report=json.loads((out/'prepare_report.json').read_text())
        assert report['mode']==mode and not report['complete_operator_cache']
