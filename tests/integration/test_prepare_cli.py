import json
import os
import subprocess
import sys
from pathlib import Path
import h5py
import numpy as np
import pytest


def raw_input(root):
    r=np.arange(801)*.01
    (root/'X.orb').write_text('Element X\nMesh 801\ndr .01\nType L N\n0 0 0\n'+'\n'.join(map(str,2/np.pi**.25*np.exp(-r*r/2))))
    (root/'POSCAR').write_text('test\n1\n12 0 0\n0 12 0\n0 0 12\nX\n2\nCartesian\n0 0 0\n1.2 .3 .2\n')
    with h5py.File(root/'H.h5','w') as f:
        g=f.create_group('0');g['0_0_0_0_0']=[[-1.]];g['1_1_0_0_0']=[[1.]];g['0_1_0_0_0']=[[.2]]
    d=dict(version=1,structure={'file':'POSCAR'},model=dict(type='lcao',hamiltonian={'path':'H.h5','format':'deeptb'},orbitals={'names':['X.orb']}))
    p=root/'input.json';p.write_text(json.dumps(d));return p


def command(p,out,mode='build',extra=()):
    return [sys.executable,'-m','kubo_msd.operators.prepare','--input',str(p),'--out',str(out),'--mode',mode,'--expect-atoms','2','--expect-orbitals','2',*extra]


def test_cli_audit_pilot_build_and_direct_optional_loading(tmp_path):
    p=raw_input(tmp_path)
    for mode in ('audit','pilot','build'):
        r=subprocess.run(command(p,tmp_path/mode,mode),capture_output=True,text=True,timeout=90)
        assert r.returncode==0,r.stdout+r.stderr
    from kubo_msd.io.model import load_model
    from kubo_msd.operators.builder import BuilderOptions
    a=load_model(p,builder_options=BuilderOptions());b=load_model(tmp_path/'build/model.json')
    np.testing.assert_array_equal(a.S.data,b.S.data)
    r=subprocess.run(command(p,tmp_path/'build'),capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stderr


def test_concurrent_cache_publication_and_corrupt_model(tmp_path):
    p=raw_input(tmp_path);out=tmp_path/'operators'
    cmd=command(p,out,extra=['--table-cache',str(tmp_path/'tables')])
    jobs=[subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(2)]
    for job in jobs:
        stdout,stderr=job.communicate(timeout=90);assert job.returncode==0,stdout+stderr
    from kubo_msd.io.model import load_model
    with h5py.File(out/'operators.h5','r+') as f:f['S/data'][0]+=1
    with pytest.raises(ValueError,match='hash'):load_model(out/'model.json')


def test_pilot_rejects_zero_sample_request(tmp_path):
    p=raw_input(tmp_path)
    r=subprocess.run(command(p,tmp_path/'pilot','pilot',extra=['--pilot-pairs-per-type','0']),capture_output=True,text=True,timeout=30)
    assert r.returncode!=0
    assert not (tmp_path/'pilot/pilot.json').exists()
