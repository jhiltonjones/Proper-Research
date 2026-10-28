#!/bin/bash
#SBATCH --job-name=magbeam_bench
#SBATCH --partition=amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=magbeam_bench_%j.out
#SBATCH --error=magbeam_bench_%j.err

set -euo pipefail

# Activate the same environment used for the project, for example:
# source "$HOME/miniforge3/etc/profile.d/conda.sh"
# conda activate your_environment

# Keep BLAS serial while finite-difference columns are parallelised explicitly.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run from the repository root.
cd "${SLURM_SUBMIT_DIR}"

# Deterministic serial baseline.
python compare_forward_jacobian_before_after.py \
  --variant contact \
  --jacobian-mode fast \
  --num-poses 20 \
  --repeats 3 \
  --workers 1 \
  --result-detail contact \
  --strict \
  --output-dir "benchmark_contact_serial_${SLURM_JOB_ID}"

# Explicit sensitivity-column parallelism. Compare this against the serial run;
# do not assume four workers will be faster for a small N_nodes problem.
python compare_forward_jacobian_before_after.py \
  --variant contact \
  --jacobian-mode fast \
  --num-poses 20 \
  --repeats 3 \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --result-detail contact \
  --strict \
  --output-dir "benchmark_contact_workers${SLURM_CPUS_PER_TASK}_${SLURM_JOB_ID}"
