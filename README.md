# KuboMSD

[中文说明](README.zh-CN.md)

The standalone GPU edition (**1.0.4**) is in [`kubo_msd_gpu/`](kubo_msd_gpu/README.md). It provides GPU operator construction and single-GPU transport without the CPU C++ extensions. Install it in a separate virtual environment; both editions use the Python package name `kubo_msd`. The documentation below describes the root package.

KuboMSD computes energy-resolved electronic mean-square displacement (MSD), velocity autocorrelation and diffusion coefficients from a fixed single-particle Hamiltonian. It supports orthogonal and nonorthogonal bases at a single Γ point.

Version: **1.1.0.dev4**. The package is under development. The supported source-build environment is Linux with Python 3.10 or later and a C++17 compiler with OpenMP; the release checks use Python 3.11.

## Methods

- Full diagonalization of `H C = S C ε`, with CPU, hybrid CPU/GPU and single-GPU backends.
- Stochastic KPM with Jackson broadening: a stepwise commutator method and a spectral velocity-correlation method.
- Nonorthogonal metric solves using PCG or a reusable Cholesky factor. PCG supports diagonal, atom-block, spatial-block and IC(0) preconditioners.
- Construction of overlap and dipole matrices from numerical atomic orbitals and a POSCAR structure, with a DeePTB-format Hamiltonian. The native two-center integrals support s, p, d and f orbitals and multiple radial functions.
- OpenMP operator construction and stochastic kernels; optional MPI operator construction and distribution of stochastic samples.

This version does not calculate self-consistent electronic structure, total energies, atomic forces or structural relaxation. It does not include mobility postprocessing, multi-k integration, distributed exact diagonalization or electron–phonon scattering.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[test]'
kubomsd --version
kubomsd capabilities
```

The build compiles two C++ extensions and requires an OpenMP-capable compiler. On Debian/Ubuntu, the usual prerequisites are `build-essential` and `python3-dev`. See [installation](docs/INSTALL.md) for CUDA, MPI and build details.

## First calculation

The repository includes small synthetic examples that require no external material data. Run this command from the repository root after installation:

```bash
kubomsd run \
  --input examples/two_level/model.json \
  --out runs/two_level_exact \
  --solver exact --backend cpu \
  --emin -1.5 --emax 1.5 --n-energy 61 \
  --sigma-ev 0.1 --axes x \
  --tmax-fs 1 --dt-fs 0.05 --npz
```

The output directory must not already exist. A completed run writes `results.h5` and `metadata.json`; `--npz` adds `results.npz`.

Energies passed to the command are **absolute energies in eV**, in the same reference as the input Hamiltonian. `--efermi-ev` records a reference; it does not shift the input energy grid. Diffusion is stored in Å²/fs; multiply by 0.1 to obtain cm²/s.

## Documentation

| Topic | Document |
| --- | --- |
| Installation, CUDA and MPI dependencies | [Installation](docs/INSTALL.md) |
| Exact, stochastic, raw LCAO and MPI runs | [Usage](docs/USAGE.md) |
| Matrix schema, HDF5 blocks and units | [Input formats](docs/INPUT_FORMAT.md) |
| HDF5 datasets, masks and estimators | [Outputs](docs/OUTPUTS.md) |
| Definitions, normalization and convergence | [Numerical methods](docs/NUMERICS.md) |
| PCG, Cholesky and preconditioners | [Metric solvers](docs/STOCHASTIC_METRIC.md) |
| Orbital order and phase conventions | [Basis conventions](docs/HARMONIC_CONVENTIONS.md) |
| Two-center integral formulas | [Integral derivation](docs/TWO_CENTER_DERIVATION.md) |
| Tests and release checks | [Validation](docs/VALIDATION.md) |
| References and dependency provenance | [Sources](SOURCES.md) |

## Tests

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest tests/unit tests/integration -q
```

GPU tests require a working CUDA device. MPI tests require a launcher and an explicit opt-in. Commands and the checks performed on this source distribution are listed in [Validation](docs/VALIDATION.md).

Passing software tests does not establish convergence for a material calculation. Check the orbital basis, energy broadening, stochastic sampling, propagation tolerances, simulation time and finite-cell effects for the system being studied.

## License

KuboMSD and the standalone GPU edition are licensed under the [MIT License](LICENSE). Dependency licenses remain those of their respective projects. See [Sources](SOURCES.md).
