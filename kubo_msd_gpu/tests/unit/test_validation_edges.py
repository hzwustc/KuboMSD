from dataclasses import replace
import numpy as np
import pytest
from numpy.testing import assert_allclose

def test_nonhermitian_composite_rejected_before_any_solver():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.config import RunConfig
    from kubo_msd.operators.velocity import build_operators
    m=ModelBundle(np.diag([-.2,.3]),np.eye(2),composite={'x':(np.eye(2),np.zeros((2,2)),np.zeros((2,2)))})
    cfg=RunConfig(energy_eV=[0],time_fs=[0,.1],axes=('x',),solver='stochastic-kpm',kernel='jackson')
    with pytest.raises(ValueError,match='Hermitian'): build_operators(m,cfg)

def test_spectral_coarse_integration_and_tiny_segments_still_execute():
    from test_propagation import setup_ops
    from kubo_msd.solvers.spectral import spectral_samples
    ops,cfg=setup_ops(); cfg=replace(cfg,integration_dt_fs=.2,segment_fs=.01)
    d,a,meta=spectral_samples(ops,cfg,np.eye(2,dtype=complex))
    assert np.all(np.isfinite(a['vac_A2_per_fs2/vac_integral']))

def test_result_reader_preserves_all_estimator_masks(tmp_path):
    from kubo_msd.config import RunConfig
    from kubo_msd.dispatch import run_input
    from kubo_msd.io.results import read_results
    from test_model_io import write_model
    write_model(tmp_path)
    cfg=RunConfig(energy_eV=[-.5,0,.5],time_fs=[0,.2,.4],axes=('x',),output_npz=True)
    run_input(tmp_path/'model.json',cfg,tmp_path/'out')
    for path in (tmp_path/'out',tmp_path/'out/results.npz'):
        data=read_results(path)['arrays']
        for key in ('valid/pair_analytic','valid/msd_gradient','reason_code/pair_analytic','reason_code/msd_gradient'): assert key in data

def test_low_dos_validity_does_not_depend_on_other_output_energies():
    from kubo_msd.contracts import ModelBundle
    from kubo_msd.config import RunConfig
    from kubo_msd.operators.velocity import build_operators
    from kubo_msd.solvers.exact import iter_exact_blocks
    m=ModelBundle(np.diag([0,1.]),np.eye(2),velocity={'x':np.ones((2,2))})
    c=RunConfig(energy_eV=[.5,1,2],time_fs=[0,.1],axes=('x',),sigma_eV=.1)
    ref=list(iter_exact_blocks(m,build_operators(m,c),c))[0]
    c=replace(c,energy_eV=[2,10]); new=list(iter_exact_blocks(m,build_operators(m,c),c))[0]
    assert_allclose(ref.static['dos_states_per_eV'][-1],new.static['dos_states_per_eV'][0])
    assert ref.arrays['valid/pair_analytic'][0,1,-1]==new.arrays['valid/pair_analytic'][0,1,0]

def test_requested_wrong_stochastic_bounds_fail():
    from test_propagation import setup_ops
    from kubo_msd.metric.bounds import estimate_spectral_bounds
    ops,cfg=setup_ops()
    with pytest.raises(ValueError): estimate_spectral_bounds(ops,replace(cfg,spectral_bounds=(-.1,.1)))

def test_orbitals_import_without_removed_scipy_sph_harm(monkeypatch):
    import scipy.special
    import importlib
    import kubo_msd.operators.lcao_integrals as integrals
    monkeypatch.delattr(scipy.special,'sph_harm',raising=False)
    importlib.reload(integrals)
    assert_allclose(integrals.real_harmonics(0,np.eye(3)),1/np.sqrt(4*np.pi))

def test_spectral_no_vac_omits_only_optional_output():
    from test_propagation import setup_ops
    from kubo_msd.solvers.spectral import spectral_samples
    ops,cfg=setup_ops()
    _,a,_=spectral_samples(ops,replace(cfg,include_vac=False),np.eye(2,dtype=complex))
    assert 'vac_A2_per_fs2/vac_integral' not in a
    assert 'msd_A2/vac_integral' in a and 'diffusion_A2_per_fs/vac_integral' in a
