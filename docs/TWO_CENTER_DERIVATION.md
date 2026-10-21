# Independent two-center derivation and numerical ABI (algorithm v2-onsite)

All radial coordinates below are Bohr. Input orbitals are `f_a(r) Y_la,ma(rhat)`;
no renormalization or fitted overlap is performed. Real harmonics have ABACUS order
`0, cos(1), sin(1), ...` with the Condon–Shortley phase. This is the internal
integral convention; DeePTB H5 assembly additionally applies the signed orbital
permutation documented in [HARMONIC_CONVENTIONS.md](HARMONIC_CONVENTIONS.md).
In terms of complex
harmonics, cosine is `(Y_lm + (-1)^m Y_l,-m)/sqrt(2)` and sine is
`(Y_lm - (-1)^m Y_l,-m)/(i sqrt(2))`. Thus Y_11c = -sqrt(3/4pi) x/r,
Y_11s = -sqrt(3/4pi) y/r, Y_10 = sqrt(3/4pi) z/r.

## Fourier and angular normalization

Use F(k)=integral exp(-ik.r) f(r) dr, inverse `(2pi)^-3`.
The radial transform is `F_a,l(k)=integral_0^cut r^2 f_a(r) j_l(kr) dr`.
The transform of an orbital is `4pi (-i)^l F_a,l(k) Y_lm(khat)`.
The real Gaunt coefficient is `G(a,b,c)=integral Y_a Y_b Y_c dOmega`.
It is computed once by product Gauss–Legendre/trapezoidal angular integration
which is exact for the finite spherical-polynomial products in this ABI (up to
floating-point rounding). Triangle and even-parity rules are enforced.

For ket displacement d, expanding exp(-ik.d) gives

`S_ab(d) = 8 sum_LM i^(la-lb-L) G(la ma,lb mb,LM) Y_LM(dhat) T_ab,L(|d|)`

where `T_ab,L(R)=integral_0^infinity k^2 F_a,la(k) F_b,lb(k) j_L(kR) dk`.
The phase is real on allowed channels and equals `(-1)^((la-lb-L)/2)`.
The factor 8 comes from `(4pi)^3/(2pi)^3`. At R=0 only L=0 survives;
`8 Y_00 G(a,b,00)=2/pi delta_ab` for equal angular indices, as required
by radial Parseval. A displaced s–p Gaussian checks the phase independently.
Scalar L ranges to la+lb (6 for input f orbitals), not merely input lmax.

## Dipole without a midpoint approximation

Expand the coordinate multiplied bra exactly:

`x_q f_a(r) Y_la,ma = r f_a(r) sum_lambda,mu c(q,a;lambda,mu) Y_lambda,mu`.

The Cartesian coefficients are Gaunt coefficients with Y_1:
`c_x=-sqrt(4pi/3)G(la ma,11c,lambda mu)`,
`c_y=-sqrt(4pi/3)G(la ma,11s,lambda mu)`,
`c_z=+sqrt(4pi/3)G(la ma,10,lambda mu)`.
Only lambda=la-1,la+1 occurs. Evaluate the same overlap formula for radial
`r f_a` and angular lambda, then contract c. This gives D relative to the
bra center, in Bohr; multiply by BOHR_A once to return Angstrom.
For input l<=3, augmented bra lambda<=4 and vector L<=7 are required.
No derivative of a tabulated noisy radial function is required.
Onsite s–p and p–d dipoles are generally nonzero. The tables retain them.

Translation: `X=D+r_i S`. Center exchange, obtained by changing integration
variable r -> r+d, gives `S_ba(-d)=S_ab(d)^T` and
`D_ba(-d)=D_ab(d)^T-d S_ab(d)^T`. The merge step alone generates the reverse
image, preserving these identities without symmetrizing a faulty block.

## Quadrature, interpolation, endpoints, and error evidence

Orbitals are the original not-a-knot cubic splines on the .orb grid with zero
outside the supplied finite cutoff. C++ integrates these splines with composite
Gauss quadrature. Reciprocal integration uses a finite even Simpson grid;
kmax, dk, radial quadrature order, distance spacing and refinement checks are
recorded. Distance channels use local degree-five interpolation (six nodes),
including one-sided endpoint stencils; no extrapolation beyond the support.
At R=0 j_0=1, j_L>0=0. At or outside the sum of cutoffs S/D are exactly zero.
This support truncation is physical; convergence near the boundary is tested.

The runtime convergence checks compare independent reciprocal/radial settings
and withheld distance midpoints, with allocated targets epsilon/5 each. They
are sample-based evidence, not rigorous uniform error bounds. Independent
real-space reference convergence has a separate epsilon/5 budget; unresolved
references are REFERENCE_UNCONVERGED and cannot certify the fast path.
Actual parameter refinements and worst errors must be recorded; requesting a
smaller epsilon must never reuse a less accurate table under another label.

## Numerical payloads (C-contiguous arrays; schema 1)

- `radial_l`: int64[nrad], input 0..3, radial IDs ordered by species/l/zeta.
- `radial_species`: int64[nrad], species IDs follow the ordered species list.
- `grid_offsets`: int64[nrad+1], flattened radial grids `radius`: float64[ngrid], Bohr.
- `coeff_offsets`: int64[nrad+1], flattened `coefficients`: float64[4*(ngrid-nrad)],
  per interval [cubic,quadratic,linear,constant], values in original orbital units.
- `species_count`: number of species; radial IDs are not atom IDs.
- settings: `kmax` (Bohr^-1), `dk` (Bohr^-1), `distance_step` (Bohr),
  `radial_order` (Gauss nodes per spline interval), `memory_limit_bytes`.
- pair payload: `species`: int64[npair,2], `displacement_A`: float64[npair,3],
  `pair_ids`: int64[npair] (unique, nonnegative stable global IDs).
- result: `offsets`: int64[npair+1], S float64[nblockvalues],
  D float64[3,nblockvalues], row-major per pair; `pair_ids` retained verbatim.
- serialized table contains the original payload, resolved settings, distance
  channels, orbital mappings and convergence report; loading does not build tables.

All native counts/offsets use checked 64-bit indexing. C++ releases the GIL for
quadrature and pair loops. Table arrays are immutable after creation; scratch
belongs to each OpenMP iteration. MPI and Python/HDF5 are main-thread only.

Primary mathematical references (identities, not implementation source):
[NIST plane-wave expansion](https://dlmf.nist.gov/10.60.E7) and
[NIST spherical-harmonic/Gaunt relations](https://dlmf.nist.gov/34.3).
Implementation and the coordinate-multiplied-bra construction above are independent.

## Exact same-center endpoint

At R=0 only L=0 survives. Parseval reduces the radial channel to `(pi/2) * integral r^(2+power) fa(r) fb(r) dr`, with power 0 for overlap and 1 for the coordinate-multiplied bra. The union of both spline knot grids partitions this polynomial integrand into degrees at most 9. Six-point Gauss quadrature integrates each piece exactly up to floating-point roundoff. This endpoint replaces the finite-k endpoint before interpolation. It follows from the original integral; no matrix symmetrization or altered normalization is applied. The algorithm/cache ID was advanced to `fourier-bessel-real-gaunt-v2-onsite` so earlier finite-k endpoint tables are invalidated.
