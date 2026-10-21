import json
import os
from pathlib import Path
import subprocess
import sys

def test_benchmark_reports_measured_times_and_errors(tmp_path):
    root=Path(__file__).resolve().parents[2]
    env=dict(os.environ,PYTHONPATH=str(root/'src'),OPENBLAS_NUM_THREADS='1')
    result=subprocess.run([sys.executable,str(root/'benchmarks/benchmark_transport.py'),'--quick','--out',str(tmp_path/'benchmark')],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    data=json.loads((tmp_path/'benchmark/report.json').read_text())
    assert len(data['runs'])==3
    assert all(r['wall_seconds']>0 for r in data['runs'])
    assert data['claim']=='small-model measurements only; no material/GPU speedup claim'
