# Running calculations

Commands below assume an installed package and the repository root as the working directory. Output directories must be new. Values in the synthetic examples are demonstration settings, not convergence recommendations for materials.

## Exact diagonalization

```bash
kubomsd run \
  --input examples/two_level/model.json --out runs/two_level_exact \
  --solver exact --backend cpu --axes x \
  --emin -1.5 --emax 1.5 --n-energy 61 --sigma-ev 0.1 \
  --tmax-fs 1 --dt-fs 0.05 --npz
```

EXACT uses the complete generalized eigensystem. Gaussian broadening is the default. Jackson broadening is also available with `--kernel jackson --n-moments N`. The spectral scale must contain the complete spectrum, independently of the output energy window.

CPU EXACT can use `--exact-projection-kernel dense` when memory permits. This expands the required operators one output direction at a time and uses dense BLAS. The default is sparse projection. `--preconditioner full-cholesky` is a separate CPU EXACT option for metric operations outside eigenbasis projection.

`--backend hybrid` keeps diagonalization on the CPU; `--backend gpu` performs the generalized solve through a device Cholesky transformation. Both need CuPy and a CUDA device. Memory checks do not replace scheduler or operating-system limits.

## Stochastic KPM

```bash
kubomsd run \
  --input examples/two_level/model.json --out runs/two_level_stochastic \
  --solver stochastic-kpm --kernel jackson --stochastic-method stepwise \
  --backend cpu --cpu-kernel cpp --omp-threads 2 --axes x \
  --emin -1.5 --emax 1.5 --n-energy 61 --n-moments 64 \
  --spectral-min -2 --spectral-max 2 \
  --nsamples 32 --statistics-block 8 --sample-batch 8 \
  --tmax-fs 1 --dt-fs 0.05 --no-vac
```

`stepwise` computes the commutator MSD by propagation. `spectral` computes velocity correlations and their time integrals. `auto` currently selects `stepwise`; the decision is recorded before the run. `--no-vac` avoids the optional extra correlation calculation in stepwise mode.

Stochastic KPM requires Jackson broadening and full-space sampling. It does not support the hybrid backend or initial-state window selection. The spectral bounds, when supplied with `--spectral-min` and `--spectral-max`, must contain the entire spectrum with a margin.

`nsamples` is the number of random vectors; `sample-batch` controls processing batches, and `statistics-block` controls independent statistical groups. Changing a batching option does not create more samples. Check stochastic uncertainty rather than judging convergence from a single smooth curve.

Metric solver choices and examples are in [Metric solvers](STOCHASTIC_METRIC.md).

## Raw LCAO inputs

The raw input contains a POSCAR path, numerical orbitals and a DeePTB-format HDF5 Hamiltonian. See [Input formats](INPUT_FORMAT.md).

```bash
kubomsd-prepare \
  --input examples/raw_lcao/input.json --out runs/raw_audit --mode audit

kubomsd-prepare \
  --input examples/raw_lcao/input.json --out runs/raw_operators --mode build \
  --operator-builder cpp-omp --threads 2 --integral-tol 1e-5 \
  --table-cache runs/radial_tables --memory-limit-gib 2

kubomsd run \
  --input runs/raw_operators/model.json --out runs/raw_exact \
  --solver exact --backend cpu --axes x \
  --emin -0.5 --emax 1 --n-energy 31 --sigma-ev 0.1 \
  --tmax-fs 0.2 --dt-fs 0.02
```

`audit` checks input metadata and does not build the complete Hamiltonian. `pilot` checks sampled pairs against an independent reference. `build` writes the complete operator cache, including `model.json`, `operators.h5`, `build_report.json` and `CACHE_READY`.

Radial tables can be reused across structures with compatible orbitals and numerical settings. A full operator cache is tied to the structure, orbitals, Hamiltonian, basis convention and construction settings. Thread count and transport backend do not change its physical identity. Incompatible existing caches are rejected.

## MPI

MPI construction distributes atom pairs. Every rank must see the input and output files:

```bash
mpiexec -n 2 kubomsd-prepare \
  --input examples/raw_lcao/input.json --out runs/raw_operators_mpi --mode build \
  --build-distribution mpi --operator-builder cpp-omp --threads 1 \
  --table-cache runs/radial_tables_mpi --memory-limit-gib 2
```

The memory budget is per rank. Root collects the complete sparse result; this is not distributed storage of the final Hamiltonian. The construction protocol does not provide fault-tolerant MPI recovery.

MPI transport distributes stochastic statistical blocks:

```bash
mpiexec -n 2 kubomsd run \
  --input runs/raw_operators_mpi/model.json --out runs/raw_stochastic_mpi \
  --solver stochastic-kpm --kernel jackson --stochastic-method stepwise \
  --transport-distribution mpi --backend cpu --omp-threads 1 --axes x \
  --emin -0.5 --emax 1 --n-energy 31 --n-moments 32 \
  --spectral-min -2 --spectral-max 2 \
  --nsamples 16 --statistics-block 4 --sample-batch 4 \
  --tmax-fs 0.2 --dt-fs 0.02 --no-vac
```

Build raw LCAO inputs first. MPI transport accepts the matrix-model schema and rejects raw input adapters. EXACT remains a single-process calculation even when its input cache was built with MPI. For GPU MPI, each rank needs a separate visible device; this is sample distribution, not one Hamiltonian split across GPUs.

## Grids, directions and threads

- `--energy-grid energies.json` accepts a JSON list of absolute energies and replaces `--emin/--emax`.
- `--time-grid times.json` accepts a strictly increasing JSON list beginning at zero and replaces `--tmax-fs/--dt-fs`.
- `--axes x,y` computes two directions. Custom orthonormal direction vectors can be supplied with `--axes-vectors directions.json`.
- `--efermi-ev VALUE --efermi-source TEXT` records the reference energy. It does not shift input energies or alter occupations.
- For a film, choose the physical in-plane directions explicitly. Vacuum volume is not a transport normalization.

Avoid thread oversubscription. For a native stochastic run, a useful starting configuration is `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and an explicit `--omp-threads` within the allocation. CPU EXACT uses BLAS, so set BLAS threads for that calculation separately. MPI ranks multiplied by threads per rank must fit the allocated CPUs.

The complete option list is available through `kubomsd run --help` and `kubomsd-prepare --help`.
