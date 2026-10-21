import pytest
from test_build_mpi import launch_build

@pytest.mark.parametrize('mode',['root-input','worker','missing','duplicate','corrupt','root-write','cleanup'])
def test_application_failure_coordinated(tmp_path,mode):
    result=launch_build(tmp_path,3,2,mode)
    assert result.returncode!=0
    for r in range(3): assert (tmp_path/f'failure-{r}.txt').is_file(),result.stderr
    assert not list(tmp_path.rglob('CACHE_READY'))
