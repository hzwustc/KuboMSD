import pytest

@pytest.mark.parametrize('method',['auto','stepwise','spectral'])
def test_auto_is_fixed_and_explicit_is_preserved(method):
    from kubo_msd.dispatch import choose_stochastic_method
    from kubo_msd.config import RunConfig
    c=RunConfig(energy_eV=[0],time_fs=[0],solver='stochastic-kpm',kernel='jackson',stochastic_method=method)
    d=choose_stochastic_method(c,{'n_orbitals':10},None)
    assert d['method']==('stepwise' if method=='auto' else method)
    assert d['rule_version'] and d['reason']
