import subprocess
import sys
import pytest

@pytest.mark.parametrize('damage',["state['nspecies']=0", "state['step']=float('nan')", "state['channel_keys'][3]=9", "state['channel_keys']=state['channel_keys'][:0];state['values']=state['values'][:0]"])
def test_restore_rejects_malformed_state_without_native_crash(damage):
    code='''import numpy as np
from kubo_msd import _operators_native as n
from kubo_msd.operators.tables import orbital_payload
from kubo_msd.operators.lcao_integrals import OrbitalSet
r=np.linspace(0,2,101);o=OrbitalSet('X',r,[(0,0,np.exp(-r*r))],[1.],{})
t=n.build_tables(orbital_payload({'X':o}),dict(kmax=8.,dk=.2,distance_step=.1,radial_order=4))
state=t.state()
'''+damage+'''
try:n.restore_tables(state)
except (ValueError,RuntimeError):raise SystemExit(0)
raise SystemExit(8)
'''
    p=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr+f' return code {p.returncode}'


def test_cached_payload_memory_checked_before_numpy_load(tmp_path,monkeypatch):
    import numpy as np
    from kubo_msd import _operators_native as n
    from kubo_msd.operators.tables import orbital_payload
    from kubo_msd.operators.lcao_integrals import OrbitalSet
    from kubo_msd.operators import cache
    r=np.linspace(0,2,1001);o=OrbitalSet('X',r,[(0,0,np.exp(-r*r))],[1.],{})
    table=n.build_tables(orbital_payload({'X':o}),dict(kmax=8.,dk=.2,distance_step=1.,radial_order=4))
    cache._write_table(tmp_path,table,{})
    def premature(*args,**kwargs):raise AssertionError('NPZ arrays loaded before memory preflight')
    monkeypatch.setattr(np,'load',premature)
    with pytest.raises(ValueError,match='budget'):cache._read_table(tmp_path,1024)


def test_direct_restore_counts_radial_payload_memory():
    import numpy as np
    from kubo_msd import _operators_native as n
    from kubo_msd.operators.tables import orbital_payload
    from kubo_msd.operators.lcao_integrals import OrbitalSet
    r=np.linspace(0,2,1001);o=OrbitalSet('X',r,[(0,0,np.exp(-r*r))],[1.],{})
    table=n.build_tables(orbital_payload({'X':o}),dict(kmax=8.,dk=.2,distance_step=1.,radial_order=4))
    state=table.state();state['settings']['memory_limit_bytes']=1024
    with pytest.raises(ValueError,match='budget'):n.restore_tables(state)
