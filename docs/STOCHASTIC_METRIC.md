# Nonorthogonal metric solvers

Every solve targets the original equation `S Y = B`. For each right-hand side j, the acceptance condition is

```text
||B_j − S Y_j||₂ ≤ max(metric_rtol ||B_j||₂, metric_atol).
```

Preconditioning changes the numerical solve, not the physical overlap matrix. The program does not shift S to make a failing preconditioner succeed.

## Options

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--metric-solver` | `pcg` | `pcg` or stochastic-only `cholesky` |
| `--metric-rtol` | `1e-10` | Relative original-equation residual tolerance |
| `--metric-atol` | `0` | Absolute residual tolerance |
| `--metric-maxiter` | `1000` | PCG iteration limit |
| `--metric-residual-policy` | `fixed` | `fixed` or `adaptive` |
| `--metric-check-interval` | `32` | Periodic true-residual interval |
| `--metric-gap-ratio` | `0.1` | Adaptive residual-gap threshold |
| `--metric-refine-maxiter` | `3` | Direct-solve refinement limit |
| `--metric-gpu-check-batch` | `1` | Host status read interval on GPU |
| `--preconditioner` | `diagonal` | PCG preconditioner |
| `--metric-ic-ordering` | `rcm` | `rcm` or `natural` |
| `--metric-ic-shift-max` | `0.01` | Maximum shift of the IC preconditioner only |

Adaptive PCG checks the true residual both periodically and when a recurrence first indicates convergence. Converged columns are frozen. False convergence or a large residual gap triggers a per-column restart within the original iteration budget.

## Preconditioners

- `diagonal`: inexpensive default.
- `block-jacobi`: atom blocks, using the model's atom slices.
- `cluster-jacobi`: spatial clusters containing whole atoms; requires a partition file.
- `ic0`: incomplete Cholesky with the original sparse pattern and a bounded preconditioner-shift sequence. A positive-definite S can still fail IC(0); failure is reported without switching solvers.

The native extension is required for IC(0) initialization, including GPU use. GPU atom/cluster factors are grouped by block size. Batched GPU status checks reduce host reads without weakening the final residual check.

For an operator cache built from a POSCAR structure, create a spatial partition from the repository root:

```bash
python scripts/prepare_metric_partition.py \
  --model runs/raw_operators/model.json \
  --structure examples/raw_lcao/POSCAR \
  --max-orbitals 128 --out runs/partition_128.json
```

Then add these options to a stochastic run:

```text
--preconditioner cluster-jacobi
--metric-partition-file runs/partition_128.json
--metric-residual-policy adaptive
```

The partition is bound to the model, overlap fingerprint, structure and orbital order. It cannot be reused for an unrelated cache.

## Direct Cholesky

Add `--metric-solver cholesky` to a stochastic CPU or GPU run. This factors S once and uses triangular solves, followed by original-equation residual checks and optional refinement.

Keep PCG-only options at their defaults: do not combine this mode with adaptive PCG, IC(0), spatial partitioning or a nondefault preconditioner. The default `preconditioner=diagonal` is an unused compatibility value in this mode. Direct solves report refinement counts separately from PCG iterations.

The full factor needs O(N²) storage and typically O(N³) setup. GPU factorization is performed on the device. Choose this route only when the factor and workspaces fit memory and its setup cost is justified.

`--preconditioner full-cholesky` is a different option, supported only by CPU EXACT. It supplies a full-factor preconditioner while preserving that path's residual checks.

## Diagnostics

The result metadata records the actual solver, preconditioner, residual checks, iteration/refinement statistics and initialization costs. A faster metric solve is not necessarily a faster end-to-end transport calculation. Compare identical input, samples, tolerances and output settings in separate processes.
