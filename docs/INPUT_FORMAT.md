# Input formats

All file paths in an input JSON are resolved relative to that JSON. The package reads inputs without updating them.

## Matrix model

[The two-level example](../examples/two_level/model.json) is a complete `kubomsd-model-v1` input. Required fields are:

| Field | Meaning |
| --- | --- |
| `format` | `kubomsd-model-v1` |
| `n_orbitals` | Positive matrix dimension |
| `basis` | `orthogonal` or `nonorthogonal` |
| `kpoint` | `[0, 0, 0]` |
| `energy_unit` | `eV`, `hartree` or `rydberg` |
| `length_unit` | `angstrom` or `bohr` |
| `orbital_order` | Unique identifiers, one per orbital |
| `hamiltonian` | Matrix declaration |
| `overlap` | Required for a nonorthogonal basis |
| `velocity_source` | Explicit velocity, position or composite representation |

An orthogonal model may omit overlap; the identity is then used. A supplied overlap must agree with the declared basis. H and S must be Hermitian and S must be positive definite for the supported solvers.

Each matrix declaration includes:

```json
{
  "path": "H.npz",
  "shape": [2, 2],
  "dtype": "complex128",
  "storage": "dense",
  "index_base": 0,
  "representation": "covariant"
}
```

Supported dtypes are `float64` and `complex128`. Matrix data must agree with the declaration. NPZ and HDF5 containers support dense data and CSR storage. Dense data use the key `matrix` by default; `key` overrides it. CSR stores `data`, `indices`, `indptr` and `shape`. In HDF5 these are datasets under the declared group. `index_base` is 0 or 1.

## Velocity source

- `type: "matrix"`: supply a `matrices` map by Cartesian axis and `unit: "angstrom/fs"` or `"eV*angstrom"`. The latter is converted using ℏ.
- `type: "position"`: supply covariant position matrices in the declared length unit. Orthogonal models may instead supply one three-component `coordinates` row per orbital.
- `type: "composite"`: supply `axes`, each containing `bond`, `xbar` and `delta` matrix declarations. This is the representation produced by the LCAO builder.

Atomic coordinates cannot replace position/dipole matrix elements in a nonorthogonal basis. Periodic LCAO transport also needs the image-dependent bond terms; a diagonal coordinate operator is not a general substitute.

## Raw LCAO model

[The raw LCAO example](../examples/raw_lcao/input.json) contains:

```json
{
  "version": 1,
  "structure": {"file": "POSCAR"},
  "model": {
    "type": "lcao",
    "spin": "none",
    "hamiltonian": {
      "path": "hamiltonians.h5",
      "format": "deeptb",
      "energy_unit": "eV",
      "index_base": 0
    },
    "orbitals": {
      "dir": "orbitals",
      "names": ["A.orb", "B.orb"],
      "convention": "deeptb-real"
    }
  }
}
```

This adapter supports scalar, spinless input at Γ. The names above are placeholders; the repository example uses a synthetic element X and declares its actual orbital convention.

The HDF5 file has a top-level group `0`. Each dataset is named `i_j_Rx_Ry_Rz`, where `i,j` are atom indices and the three integers identify the ket-cell translation. Its shape is the orbital count on atom i by the orbital count on atom j. A trailing dimension of length 2 may encode real and imaginary parts. Reverse Hermitian blocks are checked or completed by the loader.

Numerical orbitals use the supported ABACUS `.orb` text layout, with radial coordinates in Bohr. Supply one orbital set per element. The builder keeps the supplied radial normalization and cutoff. The native integral implementation supports angular momenta 0 through 3 and multiple zeta functions.

The default raw-adapter convention is `deeptb-real`. Use `abacus-real` only when the Hamiltonian actually uses that order and phase. See [Basis conventions](HARMONIC_CONVENTIONS.md).

The HDF5 format describes matrix storage, not how the Hamiltonian was generated. This package does not train or run a DeePTB prediction model. No pseudopotential is needed for the overlap/dipole construction because H is supplied separately.
