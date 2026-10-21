import numpy as np
import pytest

@pytest.mark.parametrize('kw', [dict(sigma_eV=-1),dict(solver='stochastic-kpm',backend='hybrid',kernel='jackson'),dict(solver='stochastic-kpm'),dict(solver='stochastic-kpm',kernel='jackson',initial_state_selection='window'),dict(energy_eV=[1,0]),dict(time_fs=[1,2]),dict(time_fs=[0,np.nan]),dict(nsamples=0),dict(axes=('x','x'))])
def test_reject_invalid_configs(kw):
    from kubo_msd.config import RunConfig
    args=dict(energy_eV=[0,1],time_fs=[0,1]); args.update(kw)
    with pytest.raises(ValueError): RunConfig(**args)

def test_time_grid_includes_exact_last_short_interval():
    from kubo_msd.config import resolve_time_grid
    np.testing.assert_allclose(resolve_time_grid(1,.3),[0,.3,.6,.9,1])

def test_config_freezes_arrays():
    from kubo_msd.config import RunConfig
    c=RunConfig(energy_eV=[0,1],time_fs=[0,1])
    with pytest.raises(ValueError): c.energy_eV[0]=2
