import importlib.util
import json
import numpy as np
import h5py
import os
import sys
import subprocess
from pathlib import Path
from test_cli import command
from test_lcao_operators import make_orb

def test_raw_poscar_orb_hamiltonian_without_rsatb_or_mpi(tmp_path):
    for name in ('rsatb','_rsatb'): assert importlib.util.find_spec(name) is None
    make_orb(tmp_path/'X.orb')
    (tmp_path/'POSCAR').write_text('Synthetic Gaussian orbital fixture\n1\n20 0 0\n0 20 0\n0 0 20\nX\n2\nCartesian\n0 0 0\n1 0 0\n')
    with h5py.File(tmp_path/'hamiltonians.h5','w') as f:
        g=f.create_group('0'); g['0_0_0_0_0']=[[0.]]; g['1_1_0_0_0']=[[.5]]; g['0_1_0_0_0']=[[.2]]
    d={'version':1,'structure':{'file':'POSCAR'},'model':{'type':'lcao','spin':'none','hamiltonian':{'path':'hamiltonians.h5','format':'deeptb'},'orbitals':{'names':['X.orb'],'dir':'.'},'integrals':{'radial_points':40,'polar_points':12,'azimuthal_points':24,'tolerance':1e-5,'max_refinements':2}}}
    (tmp_path/'input.json').write_text(json.dumps(d))
    args=['run','--input',tmp_path/'input.json','--emin',-.1,'--emax',.5,'--n-energy',3,'--sigma-ev',.2,'--tmax-fs',.2,'--dt-fs',.1,'--axes','x','--out',tmp_path/'out']
    # MPI is an optional installed extra in v1.1. A serial run must work even
    # when all imports of MPI and RSATB are actively blocked.
    script = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('mpi4py', 'rsatb', '_rsatb'):
            raise ImportError('blocked dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from kubo_msd.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src')}
    r=subprocess.run([sys.executable,'-c',script,*map(str,args)],env=env,text=True,capture_output=True)
    assert r.returncode==0,r.stderr
    metadata=json.loads((tmp_path/'out/metadata.json').read_text()); assert metadata['provenance']['integrals']['max_scaled_refinement_difference']<1e-5
