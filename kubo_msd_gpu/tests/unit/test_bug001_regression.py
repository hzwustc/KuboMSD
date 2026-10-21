import argparse
import importlib.util
import hashlib
from pathlib import Path
import sys
import numpy as np

def test_original_bug_reproduced_and_new_scale_fixed():
    path=Path(__file__).parents[1]/'fixtures/exact_original.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='677b876ddaf0d9ee0e6565dad9db32fb22e45302733b175e8dcd98f50a9be888'
    spec=importlib.util.spec_from_file_location('kubomsd_reference_original',path); old=importlib.util.module_from_spec(spec); sys.modules[spec.name]=old; spec.loader.exec_module(old)
    args=argparse.Namespace(grid_from_npz=None,emin=-.5,emax=.5,n_energy=11,n_kpm=64)
    params=old.RunParameters(dt_fs=.1,n_steps=2,n_kpm=64,sources={})
    E,t,c,a,L,source=old._resolve_grids(args,params,(-10.,20.))
    eps=np.array([-8.,-2.,0.,2.,15.]); wrong=old.delta_matrix(E,eps,c,a,L)
    assert (c,a)==(0.,.5) and np.allclose(wrong[:,0],wrong[:,1])
    from kubo_msd.config import RunConfig
    from kubo_msd.kernels.energy import energy_weights
    cfg=RunConfig(energy_eV=E,time_fs=t,kernel='jackson',n_moments=64,spectral_bounds=(-10,20))
    new=energy_weights(E,eps,cfg)
    np.testing.assert_allclose(new,old.delta_matrix(E,eps,5,15,L),rtol=1e-11,atol=1e-13)
    assert not np.allclose(new[:,0],new[:,1])
