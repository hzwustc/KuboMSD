# Installation

## Requirements

- Linux, Python 3.10 or later.
- A C++17 compiler accepting `-fopenmp`, with its OpenMP runtime.
- NumPy ≥ 1.24, SciPy ≥ 1.10, h5py ≥ 3.8 and threadpoolctl ≥ 3.1.
- Build dependencies: setuptools ≥ 64, wheel and pybind11 ≥ 2.11, < 4.

The source build uses `-O3 -fopenmp`. It does not enable `-ffast-math` or `-march=native`. Windows and macOS builds have not been established; the current build configuration uses GCC-style flags.

## CPU

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test]'
kubomsd --version
kubomsd capabilities
```

For development, replace the installation command with `python -m pip install -e '.[test]'`. Runtime installations do not require pytest.

The installed commands are `kubomsd` and `kubomsd-prepare`. The equivalent module entry points are `python -m kubo_msd` and `python -m kubo_msd.operators.prepare`.

## GPU

The optional extra installs CuPy for CUDA 12:

```bash
python -m pip install '.[gpu,test]'
kubomsd capabilities
python -m pytest tests/gpu --require-gpu -q
```

A compatible NVIDIA driver and CUDA libraries must be available. Do not install several CuPy variants in one environment. Configure the CUDA runtime for the target machine; no machine-specific library paths are included here.

EXACT supports `--backend cpu`, `hybrid` and `gpu`. Hybrid diagonalizes on the CPU and performs transport work on the GPU. Stochastic KPM supports `cpu` and `gpu`. Operator construction remains on the CPU.

## MPI

Install one MPI implementation and a compatible mpi4py build, then install the package extra:

```bash
python -m pip install '.[mpi,test]'
mpiexec -n 2 python -c 'from mpi4py import MPI; print(MPI.COMM_WORLD.rank)'
```

The `mpi` extra does not select or configure a cluster MPI installation. The launcher, shared libraries and mpi4py must use the same implementation. MPI operator construction requires a shared output filesystem. MPI transport supports stochastic KPM, not distributed EXACT.

## Build distributions

```bash
python -m pip install build
python -m build
```

This creates a source archive and a platform-specific wheel under `dist/`. Build on the target platform when Python ABI, architecture, C++ runtime or OpenMP compatibility differs. Virtual environments and prebuilt extension files are not part of this source distribution.
