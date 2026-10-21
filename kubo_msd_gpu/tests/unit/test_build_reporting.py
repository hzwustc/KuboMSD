import json
import pytest
from kubo_msd.operators.progress import BuildReport

def test_stage_success_and_failure(tmp_path):
    report = BuildReport('cpp', tmp_path/'progress.json')
    with report.stage('read_inputs'):
        sum(range(1000))
    assert report.timings['read_inputs'] > 0
    with pytest.raises(RuntimeError):
        with report.stage('pair_evaluation'):
            raise RuntimeError('pair 7')
    state=json.loads((tmp_path/'progress.json').read_text())
    assert state['status']=='FAIL'
    assert 'pair 7' in state['error']
    assert state['timings']['pair_evaluation'] > 0
