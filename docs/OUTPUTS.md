# Results

A completed calculation produces `results.h5` and `metadata.json`. `--npz` additionally exports `results.npz` for small results. The HDF5 file is first written as `results.partial.h5`; the final name is published only after finalization.

Always inspect `run_status` and the validity masks. A partial output, a failed metric solve or a low-DOS flag must not be treated as a completed physical result.

## Units and dimensions

| Quantity | Unit |
| --- | --- |
| `energy_eV` | eV, absolute input reference |
| `time_fs` | fs |
| DOS | states/eV/system |
| MSD | Å² |
| Diffusion | Å²/fs |
| Velocity autocorrelation | Å²/fs² |

Transport arrays are ordered `(axis, time, energy)`. Statistical numerator block sums use `(block, axis, time, energy)`. DOS and grid datasets have their own lower-dimensional shapes. Output axes and `axes_vectors` identify the directions; inspect dataset names rather than assuming an estimator from its shape.

The DOS spin multiplier is 1. A factor of two is not inserted automatically. Multiplying diffusion by 0.1 converts Å²/fs to cm²/s.

## Estimators

Results retain estimator names, including `msd_A2/commutator`, `msd_A2/pair_analytic`, `diffusion_A2_per_fs/vac_integral` and `diffusion_A2_per_fs/msd_gradient`, according to the selected solver. The MSD-gradient estimator uses the actual output time grid. The VAC-integral estimator uses the correlation integration path; their agreement depends on the numerical grids.

The writer also records aliases for the primary estimator. These do not replace the explicitly named arrays. Negative or oscillating instantaneous diffusion values are retained.

Stochastic output includes raw DOS/numerator statistics, block sums and uncertainty information. Preserve these when combining samples or evaluating low-DOS regions. Do not average normalized curves without accounting for their denominators.

## Metadata and validity

`metadata.json` records the resolved configuration, selected method, file hashes, units, backend details, metric diagnostics and completion state. It may include absolute input paths and local environment details; review it before sharing a material calculation.

Reason flags are bit fields:

| Value | Name |
| --- | --- |
| 1 | `LOW_DOS` |
| 2 | `UNCERTAIN_TRACE` |
| 4 | `NO_SELECTED_STATES` |
| 8 | `TRUNCATION_UNVERIFIED` |
| 16 | `INTEGRATION_UNVERIFIED` |
| 32 | `INSUFFICIENT_BLOCKS` |

The mapping is included in each result's metadata. Invalid normalized values may be NaN. Do not replace them with zeros when plotting or fitting.

## Inspect an output

```python
import json
from pathlib import Path
import h5py

folder = Path("runs/two_level_exact")
metadata = json.loads((folder / "metadata.json").read_text())
assert metadata["run_status"] == "complete"
with h5py.File(folder / "results.h5", "r") as results:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    results.visititems(show)
```
