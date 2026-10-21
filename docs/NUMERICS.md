# Numerical methods

For a normalized direction a, the energy-resolved electronic MSD is

```text
MSD_a(E,t) = Tr[δ(E−H) (X_a(t)−X_a(0))²] / Tr[δ(E−H)].
```

With the real, symmetrized velocity correlation `C_a(E,t)`, the corresponding relations are

```text
D_a(E,t)   = ∫₀ᵗ C_a(E,u) du = ½ ∂MSD_a(E,t)/∂t,
MSD_a(E,t) = 2 ∫₀ᵗ (t−u) C_a(E,u) du.
```

These are operator expressions; in a nonorthogonal basis the metric must be included in their matrix representation. The program solves `S Y = B` and does not treat `S⁻¹H` as Euclidean-Hermitian.

The output direction average is the arithmetic mean over the explicitly selected orthonormal directions. For three Cartesian directions this gives `(Dx+Dy+Dz)/3`; for a film with two selected in-plane directions it gives their mean. No automatic volume normalization is applied.

## Exact solver

The complete generalized eigensystem obeys `H C = S C ε` and `C† S C = I`. CPU and hybrid calculations use SciPy; GPU calculations use a Cholesky transformation followed by a Hermitian eigensolve. Eigenpair residuals and metric orthogonality are checked.

Gaussian broadening uses `sigma_eV`. Jackson broadening uses a Chebyshev expansion over bounds covering the full eigenspectrum. An output window near a Fermi energy does not redefine this spectral scale.

Optional initial-state window selection limits only the outer states. All intermediate states remain in the pair sums, and the complete DOS denominator is retained. This approximation requires a buffer convergence check; it is disabled by default.

The composite velocity is

```text
v = i/ℏ (bond − xbar S⁻¹H + H S⁻¹delta).
```

In the generalized eigenbasis it is evaluated as

```text
v_nm = i/ℏ [(C†bond C)_nm − (C†xbar C)_nm ε_m + ε_n(C†delta C)_nm].
```

This removes repeated metric solves from the eigenbasis projection without replacing the nonorthogonal basis. Dense CPU projection changes the multiplication method and memory use, not the formula.

## Stochastic solvers

Full-space random samples are transformed for the overlap metric. The code does not prefilter them to the output energy window. A Jackson approximation represents the energy delta function.

- `stepwise` propagates an augmented velocity-driven Chebyshev recurrence and evaluates the commutator MSD.
- `spectral` builds polynomial velocity correlations and integrates VAC to obtain D and MSD.

Both retain raw trace statistics. The writer derives `msd_gradient` separately on the requested output time grid. The integration step, output step and propagation segment are different controls.

The stochastic method does not guarantee linear scaling for every configuration. Dense Cholesky metric solves require quadratic factor storage and cubic factorization work. Ill-conditioned S, high polynomial orders and large sample counts also affect cost.

## Convergence checks

Before interpreting a material result, vary the relevant controls independently:

1. Numerical orbital/basis quality and overlap/dipole integration accuracy.
2. Gaussian width or Jackson order, with a spectral interval containing all states.
3. Random sample count and independent seeds, including low-DOS uncertainty.
4. Original-equation metric tolerance and sampling/propagation truncation tolerances.
5. VAC integration step, output time spacing and total propagation time.
6. Supercell size and the physical adequacy of a Γ-only calculation.

Oscillatory or negative instantaneous D is retained. A short-time ballistic regime is not a diffusive plateau, and a numerical plateau alone does not establish a physical scattering mechanism.

The Hamiltonian is fixed during a run. Atomic motion, inelastic scattering, self-consistent charge updates and atomic forces are outside this model. Background and method references are listed in [Sources](../SOURCES.md).
