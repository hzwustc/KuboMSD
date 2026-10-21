"""The timing artifact must compare identical work and expose numerical error."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_native_benchmark_records_real_end_to_end_and_kernel_results(tmp_path):
    root = Path(__file__).resolve().parents[2]
    run = subprocess.run(
        [sys.executable, str(root/"benchmarks/benchmark_native.py"),
         "--sizes", "12", "--nsamples", "4", "--repeats", "1",
         "--out", str(tmp_path/"native")],
        cwd=root, env=dict(os.environ, PYTHONPATH=str(root/"src")),
        text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stdout+run.stderr
    report = json.loads((tmp_path/"native/report.json").read_text())
    case = report["cases"][0]
    assert case["n_orbitals"] == 12
    assert case["end_to_end"]["reference"]["wall_seconds"] > 0
    assert case["end_to_end"]["cpp"]["wall_seconds"] > 0
    assert case["end_to_end"]["max_absolute_array_difference"] < 1e-8
    assert case["end_to_end"]["speedup_reference_over_cpp"] > 0
    for name in ("spmm", "pcg", "sampling", "chain", "projected", "propagate"):
        assert case["kernels"][name]["cpp_seconds"] > 0
        assert case["kernels"][name]["reference_seconds"] > 0
        assert case["kernels"][name]["max_absolute_difference"] < 1e-8
    assert report["scope"] == "synthetic sparse models only; material and server performance unverified"
