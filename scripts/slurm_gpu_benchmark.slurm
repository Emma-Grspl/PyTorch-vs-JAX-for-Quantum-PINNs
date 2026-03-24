#!/bin/bash
#SBATCH --job-name=quantum-pinn-benchmark
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
cd "$PROJECT_DIR"

if [ -f /etc/profile ]; then
  source /etc/profile
fi

if [ -n "${MODULESHOME:-}" ]; then
  module purge || true
  # Adapte ces modules a ton cluster.
  module load cuda || true
fi

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"

./project1/bin/python scripts/hpc_run_benchmark.py \
  --device cuda \
  --frameworks pytorch jax \
  --repeats 5 \
  --warmup 1
