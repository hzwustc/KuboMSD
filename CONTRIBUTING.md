# Development

Install an editable checkout with test dependencies:

```bash
python -m pip install -e '.[test]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
  python -m pytest tests/unit tests/integration -q
```

Changes to C++ files require rebuilding the extensions with the editable installation command. CPU tests run in GitHub Actions. CUDA and MPI checks are described in [Validation](docs/VALIDATION.md).

For a numerical change, include a small reproducible case and a reference that does not repeat the implementation being tested. Keep energy units, basis conventions and estimator names explicit. Compare raw numerators and denominators when changing stochastic estimators; a similar normalized curve can hide a normalization error.

Bug reports should include the package version, Python/NumPy/SciPy versions, backend, command, error message and a minimal input. Attach `metadata.json` when possible after reviewing its input paths and environment information. Do not include private material data or large operator caches unless they are needed and can be shared.

New GPU or MPI paths need tests that execute those paths on the corresponding hardware. A skipped test is not evidence that a backend works.

No open-source license has been selected. Resolve contribution and distribution terms with the maintainer before submitting code intended for wider reuse.
