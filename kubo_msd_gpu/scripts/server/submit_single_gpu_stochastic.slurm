#!/usr/bin/env bash
#SBATCH --job-name=kubomsd-stoch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=kubomsd-stoch-%j.log
set -euo pipefail
exec bash "${KUBOMSD_ROOT:?set KUBOMSD_ROOT}/scripts/server/run_single_gpu.sh" stochastic-kpm
