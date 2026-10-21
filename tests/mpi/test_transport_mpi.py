import os
from pathlib import Path
import shutil
import subprocess
import sys
import json
import numpy as np
import pytest
from numpy.testing import assert_allclose
from kubo_msd.config import RunConfig
from kubo_msd.dispatch import run_input
from kubo_msd.io.results import read_results

pytestmark = pytest.mark.mpi


def launcher():
    if os.environ.get("KUBOMSD_TEST_MPI") != "1":
        pytest.skip("set KUBOMSD_TEST_MPI=1 to run real local MPI subprocess tests")
    path = shutil.which("mpiexec") or str(Path(sys.executable).parent/"mpiexec")
    assert Path(path).is_file(), "MPI acceptance requested but mpiexec missing"
    return path


def model_fixture(path):
    rng = np.random.default_rng(11)
    n = 5
    a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
    s = np.eye(n) + .03*(a+a.conj().T)
    h = np.diag(np.linspace(-1, 1, n)) + .04*(a+a.conj().T)
    x = np.diag(np.arange(n, dtype=float)).astype(complex)
    matrices = {}
    for name, value in [("H", h), ("S", s), ("X", x)]:
        np.savez(path/f"{name}.npz", matrix=np.asarray(value, dtype=complex))
        matrices[name] = dict(path=f"{name}.npz", shape=[n, n], dtype="complex128",
                              storage="dense", index_base=0, representation="covariant")
    model = dict(format="kubomsd-model-v1", n_orbitals=n, basis="nonorthogonal",
                 kpoint=[0, 0, 0], energy_unit="eV", length_unit="angstrom",
                 orbital_order=list(range(n)), hamiltonian=matrices["H"], overlap=matrices["S"],
                 velocity_source=dict(type="position", matrices={"x": matrices["X"]}))
    p = path/"model.json"
    p.write_text(json.dumps(model))
    return p


def launch(nrank, model, out, mode, timeout=90):
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]/"src"),
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    if mode == "gpu-collision":
        env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    return subprocess.run(
        [launcher(), "-n", str(nrank), sys.executable, str(Path(__file__).with_name("worker.py")),
         str(model), str(out), mode], capture_output=True, text=True, env=env, timeout=timeout)


@pytest.mark.parametrize("nrank,mode", [(1, "stepwise"), (2, "stepwise"),
                                     (3, "spectral"), (4, "empty"),
                                     (2, "native-stepwise"), (3, "native-spectral"),
                                     (2, "native-adaptive"), (2, "native-cholesky"),
                                     (2, "native-ic0")])
def test_real_mpi_matches_serial_raw_statistics(tmp_path, nrank, mode):
    launcher()
    model = model_fixture(tmp_path)
    cfg = RunConfig(
        np.linspace(-2, 2, 9), [0, .03, .07],
        solver="stochastic-kpm", kernel="jackson", axes=("x",),
        n_moments=16, nsamples=3 if mode == "empty" else 17, statistics_block=2, sample_batch=3,
        time_block=2, spectral_bounds=(-4., 4.), integration_dt_fs=.02,
        include_vac=True, stochastic_method="spectral" if "spectral" in mode else "stepwise",
        cpu_kernel="reference")
    run_input(model, cfg, tmp_path/"serial")
    result = launch(nrank, model, tmp_path/"parallel", mode)
    assert result.returncode == 0, result.stdout + result.stderr
    serial, parallel = [read_results(tmp_path/name) for name in ("serial", "parallel")]
    assert serial["arrays"].keys() == parallel["arrays"].keys()
    for key, expected in serial["arrays"].items():
        assert_allclose(parallel["arrays"][key], expected, rtol=2e-9, atol=2e-11,
                        equal_nan=True, err_msg=key)
    meta = parallel["metadata"]["parallel"]
    assert meta["world_size"] == nrank
    assert sum(len(r["sample_ids"]) for r in meta["ranks"]) == cfg.nsamples
    assert sorted(s for r in meta["ranks"] for s in r["sample_ids"]) == list(range(cfg.nsamples))
    if mode.startswith("native"):
        assert all(r["metric"]["kernel"] == "cpp" for r in meta["ranks"])
    if mode=="native-cholesky":
        assert all(r["metric"]["solver"]=="cholesky" for r in meta["ranks"])
    if mode=="native-ic0":
        assert all(r["metric"]["resolved_preconditioner"]=="ic0" for r in meta["ranks"])
    if mode=="native-adaptive":
        assert all(r["metric"]["true_residual_checks"]>0 for r in meta["ranks"])


def test_rank_failure_is_seen_by_every_rank_without_hanging(tmp_path):
    result = launch(3, "unused.json", tmp_path, "failure")
    assert result.returncode != 0
    for rank in range(3):
        assert "rank 1" in (tmp_path/f"failed-{rank}.txt").read_text()


def test_missing_input_and_root_output_error_exit_all_ranks(tmp_path):
    result = launch(2, tmp_path/"absent.json", tmp_path/"not-created", "stepwise")
    assert result.returncode != 0
    assert not (tmp_path/"not-created"/"results.h5").exists()
    model = model_fixture(tmp_path)
    result = launch(2, model, tmp_path, "stepwise")
    assert result.returncode != 0
    assert not (tmp_path/"results.h5").exists()


def test_numerical_failure_never_publishes_complete_result(tmp_path):
    launcher()
    model = model_fixture(tmp_path)
    result = launch(3, model, tmp_path/"failed-solve", "numeric-failure")
    assert result.returncode != 0
    assert "MPI stage" in result.stderr
    assert not (tmp_path/"failed-solve"/"results.h5").exists()


def test_new_metric_initialization_failure_exits_all_ranks(tmp_path):
    launcher()
    model=model_fixture(tmp_path)
    result=launch(2,model,tmp_path/"partition-failed","partition-failure",timeout=30)
    assert result.returncode != 0
    assert "initialize operators" in result.stderr
    assert not (tmp_path/"partition-failed"/"results.h5").exists()


def test_rank_cleanup_failure_does_not_hang_or_mark_complete(tmp_path):
    launcher()
    model = model_fixture(tmp_path)
    result = launch(2, model, tmp_path/"cleanup-failed", "cleanup-failure", timeout=15)
    assert result.returncode != 0
    assert "scratch cleanup failure" in result.stderr
    meta = json.loads((tmp_path/"cleanup-failed"/"metadata.json").read_text())
    assert meta["run_status"] == "failed"
    assert not (tmp_path/"cleanup-failed"/"results.h5").exists()


@pytest.mark.parametrize("mode", ["config-mismatch", "input-mismatch", "append-failure"])
def test_inconsistent_rank_inputs_and_root_mid_write_fail_cleanly(tmp_path, mode):
    launcher()
    model = model_fixture(tmp_path)
    result = launch(2, model, tmp_path/mode, mode, timeout=15)
    assert result.returncode != 0
    assert not (tmp_path/mode/"results.h5").exists()
    if mode == "append-failure":
        meta = json.loads((tmp_path/mode/"metadata.json").read_text())
        assert meta["run_status"] == "failed"
        assert meta["written_time_points"] > 0


@pytest.mark.gpu
def test_mpi_gpu_duplicate_assignment_is_rejected(tmp_path):
    launcher()
    if os.environ.get("KUBOMSD_TEST_GPU_MPI") != "1":
        pytest.skip("set KUBOMSD_TEST_GPU_MPI=1 for real GPU binding rejection check")
    from kubo_msd.backends.gpu import GPUBackend
    GPUBackend()
    model = model_fixture(tmp_path)
    result = launch(2, model, tmp_path/"duplicate-gpu", "gpu-collision", timeout=30)
    assert result.returncode != 0
    assert "distinct GPU per local rank" in result.stderr
    assert not (tmp_path/"duplicate-gpu"/"results.h5").exists()


@pytest.mark.gpu
def test_one_rank_gpu_mpi_matches_cpu_reference(tmp_path):
    launcher()
    if os.environ.get("KUBOMSD_TEST_GPU_MPI") != "1":
        pytest.skip("set KUBOMSD_TEST_GPU_MPI=1 for real MPI GPU transport")
    from kubo_msd.backends.gpu import GPUBackend
    GPUBackend()
    test_real_mpi_matches_serial_raw_statistics(tmp_path, 1, "gpu-single")
