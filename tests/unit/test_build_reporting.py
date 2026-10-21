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


def test_progress_refresh_during_long_native_stage(tmp_path,monkeypatch):
    import time
    import kubo_msd.operators.progress as progress
    monkeypatch.setattr(progress,'HEARTBEAT_SECONDS',.02)
    report=BuildReport('cpp',tmp_path/'progress.json')
    with report.stage('table_build_or_load'):
        first=json.loads(report.progress_path.read_text())['elapsed_s']
        time.sleep(.08)
        current=json.loads(report.progress_path.read_text())
        assert current['elapsed_s']>first+.02
        assert current['phase']=='table_build_or_load'
    assert 'software' in report.details['environment']
