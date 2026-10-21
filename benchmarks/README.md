# Synthetic benchmarks

Run these tools from the repository root after installation. Use a fresh output path for each run.

```bash
python benchmarks/benchmark_transport.py --quick --out runs/transport_benchmark
python benchmarks/benchmark_native.py --sizes 32 --nsamples 8 --repeats 1 --out runs/native_benchmark
```

`benchmark_transport.py` compares complete exact, stepwise and spectral calculations on a synthetic model. Different accuracy settings are not a matched-accuracy speed comparison.

`benchmark_native.py` compares Python/reference and native CPU kernels on identical synthetic work, including numerical differences. `benchmark_backend_dataflow.py` and `audit_dataflow_transfers.py` require CUDA and examine backend initialization and explicit host-to-device conversions. Their counters do not include all transfers inside CUDA libraries.

`validate_gpu.py --out NEW_DIRECTORY` runs CPU/hybrid/GPU output comparisons on synthetic inputs. `validate_install.py --wheel FILE --out NEW_DIRECTORY` checks an installed wheel in a new environment; read its help for dependency options.

`metric_eigen_reference.py` supplies independent full-eigenbasis formulas used by the tests. It has no material-specific command-line driver.

Benchmark results depend on matrix structure, accuracy settings and hardware. These fixtures do not establish throughput for a large material calculation.
