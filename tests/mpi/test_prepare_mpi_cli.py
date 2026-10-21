import json
import subprocess
import os
import sys
from pathlib import Path
import pytest
from test_transport_mpi import launcher


def test_cli_mpi_audit_pilot_build(tmp_path):
    # Fixture generated once by the outer process; the command's root alone reads it.
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'integration/test_prepare_cli.py'
    spec=importlib.util.spec_from_file_location('raw_fixture',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    inp=m.raw_input(tmp_path)
    for mode in ('audit','pilot','build'):
        cmd=[launcher(),'-n','3',*m.command(inp,tmp_path/mode,mode,extra=['--build-distribution','mpi','--operator-builder','cpp-omp','--threads','2','--pair-chunk-size','1'])]
        r=subprocess.run(cmd,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},capture_output=True,text=True,timeout=90)
        assert r.returncode==0,r.stdout+r.stderr
    assert (tmp_path/'build/CACHE_READY').is_file()
