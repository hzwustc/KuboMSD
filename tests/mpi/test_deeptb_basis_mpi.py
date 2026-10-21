import importlib.util
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
from test_transport_mpi import launcher


def test_mpi_deeptb_sp_assembly_matches_serial(tmp_path):
    path=Path(__file__).resolve().parents[1]/'integration/test_deeptb_basis.py'
    spec=importlib.util.spec_from_file_location('basis_fixture',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    inp,data,*_=m.fixture(tmp_path)
    from kubo_msd.io.legacy_input import load_legacy
    from kubo_msd.io.model import load_model
    from kubo_msd.operators.builder import BuilderOptions
    serial=load_legacy(inp,data,builder_options=BuilderOptions())
    result=subprocess.run([launcher(),'-n','2',sys.executable,'-m','kubo_msd.operators.prepare',
        '--input',str(inp),'--out',str(tmp_path/'mpi'),'--mode','build',
        '--build-distribution','mpi','--operator-builder','cpp-omp','--threads','2'],
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    mpi=load_model(tmp_path/'mpi/model.json')
    for a,b in [(mpi.S,serial.S)]+[(a,b) for axis in 'xyz' for a,b in zip(mpi.composite[axis],serial.composite[axis])]:
        np.testing.assert_array_equal(a.toarray(),b.toarray())
