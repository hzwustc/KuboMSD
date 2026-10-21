import os
import subprocess
import sys
from pathlib import Path
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
from test_transport_mpi import launcher


def launch_build(path,ranks=2,threads=1,mode='ok'):
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    return subprocess.run([launcher(),'-n',str(ranks),sys.executable,str(Path(__file__).with_name('build_driver.py')),str(path),mode,str(threads)],env=env,capture_output=True,text=True,timeout=90)


@pytest.mark.parametrize('ranks',[1,2,3,4])
@pytest.mark.parametrize('threads',[1,2])
def test_mpi_build_matches_serial_and_covers_once(tmp_path,ranks,threads):
    got=launch_build(tmp_path,ranks,threads)
    assert got.returncode==0,got.stdout+got.stderr
    diag=json.loads((tmp_path/'diagnostics.json').read_text())
    assert diag['details']['mpi']['world_size']==ranks
    ids=[i for r in diag['details']['mpi']['ranks'] for i in r['pair_ids']]
    assert sorted(ids)==[0,1,2]
    from kubo_msd.operators.builder import BuilderOptions,build_operators
    from kubo_msd.operators.lcao_integrals import OrbitalSet
    r=np.linspace(0,4,401);orb=OrbitalSet('X',r,[(0,0,2/np.pi**.25*np.exp(-r*r/2))],[1.],{})
    structure=dict(cell_A=np.eye(3)*12,positions_A=np.array([[0.,0.,0.],[1.2,.3,.2]]),symbols=['X','X'])
    h={(0,0,0,0,0):np.array([[-1.]]),(1,1,0,0,0):np.array([[1.]]),(0,1,0,0,0):np.array([[.2]])}
    serial=build_operators(structure,{'X':orb},h,BuilderOptions())
    expected=dict(H=serial.H_cov.toarray(),S=serial.S.toarray(),**{a+str(i):m.toarray() for a,p in serial.composite.items() for i,m in enumerate(p)})
    with np.load(tmp_path/'matrices.npz') as f:
        for k,v in expected.items():assert_allclose(f[k],v,rtol=1e-12,atol=1e-13)
