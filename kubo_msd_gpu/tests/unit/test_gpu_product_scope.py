import importlib
import pytest


def entry():
    spec = importlib.util.find_spec('kubo_msd.gpu_cli')
    assert spec is not None, 'missing GPU product entry'
    return importlib.import_module('kubo_msd.gpu_cli')

@pytest.mark.parametrize('extra', [['--device','0,1'], ['--backend','cpu'], ['--transport-distribution','mpi']])
def test_rejects_mpi_and_device_list_before_output(tmp_path, extra):
    cli=entry(); out=tmp_path/'out'
    args=['run','--input','missing.json','--out',str(out),'--emin','-1','--emax','1']+extra
    try: status=cli.main(args)
    except SystemExit as e: status=e.code
    assert status!=0 and not out.exists()


def test_one_task_with_cpu_threads_is_allowed(monkeypatch):
    cli=entry(); monkeypatch.setenv('SLURM_NTASKS','32'); monkeypatch.setenv('OMP_NUM_THREADS','16')
    cli.validate_scope(0,'gpu','serial')

@pytest.mark.parametrize('name', ['OMPI_COMM_WORLD_SIZE','PMI_SIZE','PMIX_SIZE','MV2_COMM_WORLD_SIZE','WORLD_SIZE'])
def test_real_launcher_rejected(monkeypatch,name):
    cli=entry(); monkeypatch.setenv(name,'2')
    with pytest.raises(ValueError): cli.validate_scope(0,'gpu','serial')


def test_active_slurm_non_mpi_step_rejected(monkeypatch):
    monkeypatch.setenv('SLURM_STEP_NUM_TASKS', '2')
    monkeypatch.setenv('SLURM_PROCID', '1')
    monkeypatch.setenv('SLURM_STEP_ID', '0')
    with pytest.raises(ValueError, match='single process'):
        entry().validate_scope(0)
