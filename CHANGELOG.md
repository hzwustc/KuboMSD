# Changelog

## 1.1.0.dev4

- Add adaptive PCG residual checks and reusable Cholesky metric solves for stochastic transport.
- Add spatial cluster-Jacobi and IC(0) preconditioners.
- Group GPU block triangular solves by block size and allow batched host status checks.
- Retain original-equation residual checks for every right-hand side.

The GitHub source distribution includes portable tests and synthetic examples. Local material datasets, benchmark queues, binary environments and development records are excluded. The Python runtime and C++ sources match the development directory byte for byte; packaging and user documentation are maintained separately.

## 1.1.0.dev3

- Add native Fourier–Bessel two-center overlap and dipole integrals, OpenMP construction and MPI pair distribution.
- Add reusable radial tables and complete operator caches.
- Correct orbital-order and phase conversion between ABACUS-style numerical orbitals and DeePTB-format Hamiltonians.
- Reduce repeated metric solves in exact eigenbasis velocity projection.

Operator caches made with the earlier basis convention must be rebuilt. Results from the dev1/dev2 raw DeePTB input path should be recomputed before use.
