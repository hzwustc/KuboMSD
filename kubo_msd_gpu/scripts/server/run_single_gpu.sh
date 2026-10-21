#!/usr/bin/env bash
set -euo pipefail
: "${KUBOMSD_ROOT:?absolute product path}"
: "${KUBOMSD_ENV:?absolute isolated virtual environment path}"
: "${INPUT:?raw input.json or completed model.json}"
: "${OUT:?new output directory}"
: "${MEMORY_GB:?explicit per-case logical GPU memory budget}"
: "${EMIN:?output energy minimum in eV}"
: "${EMAX:?output energy maximum in eV}"
solver="${1:?exact or stochastic-kpm}"
case "$solver" in exact|stochastic-kpm) ;; *) exit 2;; esac
source "$KUBOMSD_ENV/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES is provided by Slurm; device 0 means this allocation's GPU.
args=(run --input "$INPUT" --out "$OUT" --solver "$solver" --device 0
  --backend gpu --transport-distribution serial --operator-builder gpu
  --memory-limit-gb "$MEMORY_GB" --device-memory-limit-gb "$MEMORY_GB"
  --emin "$EMIN" --emax "$EMAX" --n-energy "${N_ENERGY:-33}"
  --tmax-fs "${TMAX_FS:-0.1}" --dt-fs "${DT_FS:-0.05}"
  --axes "${AXES:-x,y,z}" --integral-tol "${INTEGRAL_TOL:-1e-5}"
  --build-pair-batch "${PAIR_BATCH:-128}")
if [[ -n "${TABLE_CACHE:-}" ]]; then args+=(--table-cache "$TABLE_CACHE"); fi
if [[ "$solver" == stochastic-kpm ]]; then
  args+=(--kernel jackson --stochastic-method "${METHOD:-auto}"
    --n-moments "${N_MOMENTS:-128}" --nsamples "${NSAMPLES:-8}"
    --sample-batch "${SAMPLE_BATCH:-2}" --statistics-block "${STATISTICS_BLOCK:-2}"
    --seed "${SEED:-1729}")
fi
if [[ -n "${SPECTRAL_MIN:-}" || -n "${SPECTRAL_MAX:-}" ]]; then
  args+=(--spectral-min "${SPECTRAL_MIN:?}" --spectral-max "${SPECTRAL_MAX:?}")
fi
cd "$KUBOMSD_ROOT"
exec "$KUBOMSD_ENV/bin/kubomsd-gpu" "${args[@]}"
