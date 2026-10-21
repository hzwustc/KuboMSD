import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

def model(tmp_path):
    np.savez(tmp_path/'h.npz',matrix=np.array([[0.,1.],[1.,0.]]))
    d={'format':'kubomsd-model-v1','n_orbitals':2,'basis':'orthogonal','kpoint':[0,0,0],'energy_unit':'eV','length_unit':'angstrom','orbital_order':['a:s','b:s'],'hamiltonian':{'path':'h.npz','storage':'dense','shape':[2,2],'dtype':'float64','index_base':0,'representation':'covariant'},'velocity_source':{'type':'position','coordinates':[[0,0,0],[1,0,0]]}}
    (tmp_path/'model.json').write_text(json.dumps(d)); return tmp_path/'model.json'

def command(args):
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[2]/'src'))
    return subprocess.run([sys.executable,'-m','kubo_msd',*map(str,args)],env=env,text=True,capture_output=True)

@pytest.mark.parametrize('method',['exact','stepwise','spectral','auto'])
def test_cli_end_to_end_and_existing_directory(tmp_path,method):
    p=model(tmp_path); out=tmp_path/'out'; args=['run','--input',p,'--emin',-.5,'--emax',.5,'--n-energy',5,'--tmax-fs',.2,'--dt-fs',.1,'--axes','x','--out',out,'--npz']
    if method!='exact': args+=['--solver','stochastic-kpm','--kernel','jackson','--stochastic-method',method,'--n-moments',8,'--nsamples',8,'--statistics-block',1]
    result=command(args); assert result.returncode==0,result.stderr
    meta=json.loads((out/'metadata.json').read_text()); assert meta['run_status']=='complete'
    assert (out/'results.h5').exists() and (out/'results.npz').exists()
    assert command(args).returncode!=0

def test_cli_unknown_parameters_and_missing_gpu_fail(tmp_path):
    assert command(['run','--tau',10]).returncode!=0
    r=command(['run','--input',model(tmp_path),'--emin',-.5,'--emax',.5,'--backend','gpu','--out',tmp_path/'gpu'])
    if r.returncode!=0: assert 'GPU' in r.stderr or 'CuPy' in r.stderr
