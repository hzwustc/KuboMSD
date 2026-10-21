# Synthetic examples

All examples in this directory are small generated fixtures. They are not fitted material models.

| Directory | Content |
| --- | --- |
| `two_level` | Orthogonal two-level Hamiltonian and position matrices |
| `nonorthogonal` | Explicit nonorthogonal H/S model |
| `raw_lcao` | POSCAR, synthetic Gaussian numerical orbital and DeePTB-layout HDF5 blocks |

The two-level quick start is in the [README](../README.md). The raw example's orbital convention is explicitly `abacus-real`; do not copy this value to a different Hamiltonian without checking its basis order.

Matrix NPZ files store data under the `matrix` key. The synthetic `X.orb` is included so overlap/dipole construction can be tested without external orbital files. Keep these small inputs under version control; put larger material inputs outside the repository or in the ignored `data/` directory.
