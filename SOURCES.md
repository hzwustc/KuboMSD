# References and source provenance

## Methods

- Z. Fan et al., *Linear Scaling Quantum Transport Methodologies*, Physics Reports 903, 1–69 (2021). [DOI](https://doi.org/10.1016/j.physrep.2020.12.001), [arXiv](https://arxiv.org/abs/1811.07387).
- Z. Fu et al., *Large-scale stochastic propagation method beyond the sequential approach*, Computer Physics Communications 322, 110070 (2026). [DOI](https://doi.org/10.1016/j.cpc.2026.110070), [arXiv](https://arxiv.org/abs/2510.17432). This is background for stochastic spectral propagation; the benchmarks reported in that paper are not KuboMSD benchmarks.
- NIST Digital Library of Mathematical Functions: [spherical-Bessel addition and plane-wave identities](https://dlmf.nist.gov/10.60) and [3-j symbols and Gaunt relations](https://dlmf.nist.gov/34.3).
- SciPy: [generalized Hermitian eigensolver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html).
- CuPy: [Hermitian eigensolver](https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.eigh.html).

The two-center formulas and normalization used by this implementation are documented in [Integral derivation](docs/TWO_CENTER_DERIVATION.md).

## Dependencies

NumPy, SciPy, h5py and threadpoolctl are installed as runtime dependencies. Pybind11, setuptools and wheel are build dependencies. Pytest, CuPy and mpi4py are optional test/backend dependencies. C++ and OpenMP runtimes are supplied by the compiler environment. These dependencies retain their own licenses; their source implementations are not vendored here.

The implementation was developed with mathematical and interface comparisons against RSATB, op2c and dftio. KuboMSD does not import, link or bundle these projects. The raw HDF5 adapter implements the relevant matrix layout; the harmonic conversion implements signed permutations documented in [Basis conventions](docs/HARMONIC_CONVENTIONS.md).

The repository examples are synthetic. Real material Hamiltonians, numerical orbitals, structures, calculation results and local validation queues are not included. A historical external-script test fixture is also excluded; the spectral-scale regression is covered by a small standalone test instead.

`SOURCE_SHA256SUMS` records the Python runtime and C++ source hashes. These sources match the development version used to prepare this distribution. Public packaging, documentation and portable test support were organized separately.

## License status

KuboMSD and the standalone GPU edition are licensed under the [MIT License](LICENSE). Third-party dependencies and materials retain their respective licenses. This file records provenance and references; it does not relicense third-party material.
