# Validation

The public test suite uses synthetic inputs. It covers generalized eigenpairs, nonorthogonal basis transformations, velocity operators, direct matrix-exponential references, stochastic metric solves, trace normalization, time kernels, output serialization and native operator construction.

The independent eigenbasis reference retains all intermediate states. The Jackson spectral-scale regression uses a standalone cosine-series calculation and does not require an external transport package.

## CPU

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest tests/unit tests/integration -q
```

The test environment must have the compiled extensions installed. A source checkout without native extensions is insufficient for the complete CPU suite.

## GPU

```bash
python -m pytest tests/gpu --require-gpu -q
```

Without `--require-gpu`, tests requiring a missing CUDA device may skip. A skipped test does not validate a GPU path. The GPU suite compares actual device results against CPU or independent references; no CUDA emulation is used.

## MPI

```bash
KUBOMSD_TEST_MPI=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest tests/mpi -m 'not gpu' -q
```

MPI tests create local launcher subprocesses. They cover pair ownership, deterministic assembly, serial/parallel stochastic agreement and coordinated failure handling. Set up the launcher and mpi4py before opting in. GPU/MPI checks additionally require suitable devices and `KUBOMSD_TEST_GPU_MPI=1` where requested by the test.

## Release checks

Checks below were run on 2026-10-21 using Python 3.11.15, NumPy 1.26.4, SciPy 1.12.0, h5py 3.11.0 and threadpoolctl 3.7.0. Local command logs are kept with the release preparation record, outside the public repository.

| Check | Result |
| --- | --- |
| CPU unit and integration suite | 242 passed, 23.13 s |
| Native extensions | Both rebuilt with GCC 11.4 / OpenMP |
| GPU suite | 55 passed, 150.33 s; RTX 5090, CuPy 13.6, CUDA 12.8 |
| CPU MPI suite | 34 passed, 3 GPU cases deselected, 30.26 s; local MPICH launcher |
| Wheel installation | Installed in a separate virtual environment; both extensions imported from the installed package |
| Source distribution | Source archive built; a wheel rebuilt successfully from the archive |
| Documentation examples | Exact, stochastic, raw input audit, native cache construction and cached exact transport completed |
| Runtime source identity | 67 Python/C++ source files match the development version |

The separate wheel environment reused the base environment's NumPy/SciPy/h5py installations and installed threadpoolctl from a local wheel. It was not a network-clean dependency installation. MPI and GPU tests used local process communication and device access. GPU/MPI combined tests were not part of this release check.

## Scope

Material-specific validation queues, private structures/orbitals, large matrices and historical performance reports are excluded. The public results above do not establish convergence or performance for those systems. GitHub Actions is configured for CPU tests and distribution builds; the hosted workflow has not been executed as part of local preparation.

The checked source version remains `1.1.0.dev4`. Compatibility with untested operating systems, Python versions, compiler versions and cluster environments is not implied by this record.
