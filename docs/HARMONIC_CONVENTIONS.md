# Orbital order and phases

Hamiltonian, overlap, dipole and velocity matrices must use the same orbital basis. Eigenproblem residuals alone cannot detect a consistent solve of matrices expressed in incompatible bases.

The integral engine uses ABACUS-style real Condon–Shortley spherical harmonics in the order

```text
0, cos(1), sin(1), cos(2), sin(2), ...
```

The raw DeePTB adapter transforms the generated overlap and dipole blocks to the declared Hamiltonian convention. For `deeptb-real`, the per-shell signed permutations are:

| l | Old indices in new row order, zero-based | Signs |
| --- | --- | --- |
| 0 | 0 | + |
| 1 | 2, 0, 1 | −, +, − |
| 2 | 4, 2, 0, 1, 3 | +, −, +, −, + |
| 3 | 6, 4, 2, 0, 1, 3, 5 | −, +, −, +, −, +, − |

With signed permutation U, each pair block transforms as `S_new = U_i S_old U_jᵀ` and `D_new[a] = U_i D_old[a] U_jᵀ`. H is already in its declared convention and is unchanged. Every zeta is retained and the atom order is unchanged. For example, the p shell changes from `(z, −x, −y)` to `(y, z, x)`.

Raw LCAO input defaults to `model.orbitals.convention="deeptb-real"`. Declare `"abacus-real"` only for Hamiltonians retaining that ordering. The direct Python builder defaults to `abacus-real`; the raw input parser supplies its convention explicitly.

Serial, OpenMP and MPI construction use the same transformation. The complete operator cache schema is `kubomsd-operators-v2`; incompatible older caches must be rebuilt. Radial tables keep the integral engine's internal harmonic convention.

Tests cover s/p/d/f shells, multiple zeta functions, rectangular complex blocks, Cartesian p functions, adapter behavior and invariance of eigenvalues and MSD under a consistent basis transformation.
