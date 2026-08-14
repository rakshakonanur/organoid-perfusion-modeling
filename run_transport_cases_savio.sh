#!/usr/bin/env bash
#SBATCH --job-name=transport-coupling
#SBATCH --account=fc_biome
#SBATCH --partition=savio4_htc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00
#SBATCH --output=transport-coupling-%j.out
#SBATCH --error=transport-coupling-%j.err
## Optional email notifications (uncomment both lines and set your address):
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=YOUR_BERKELEY_EMAIL

set -euo pipefail

# This workflow launches a single Python process rather than an MPI job step.
# Prevent the Conda MPI runtime from detecting Savio's incompatible PMIx context.
while IFS= read -r mpi_env_var; do
  [[ -n "${mpi_env_var}" ]] && unset "${mpi_env_var}"
done < <(compgen -A variable PMI || true)
unset SLURM_MPI_TYPE SLURM_GTIDS

# Submit this job while your existing Conda environment is active. Slurm exports
# that environment by default. Alternatively, set CONDA_ENV_PREFIX before sbatch.
ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_PREFIX:-}}"
if [[ -z "${ENV_PREFIX}" || ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "No active Conda environment found." >&2
  echo "Activate it before sbatch, or export CONDA_ENV_PREFIX=/path/to/the/environment." >&2
  exit 1
fi

CODE_DIR="${SAVIO_CODE_DIR:-/global/home/users/${USER}/coupled-multi-organoid-model}"
SCRATCH_DIR="${SAVIO_SCRATCH_DIR:-/global/scratch/users/${USER}/coupled-multi-organoid-model}"
TRIAL_DIR="${SCRATCH_DIR}/src/prep/scaled-screening/081226"
RESULTS_DIR="${SCRATCH_DIR}/src/Results/paper-results/transport-comparison/core-targeting"
PYTHON_BIN="${ENV_PREFIX}/bin/python"
SVZERODSOLVER_BIN="${ENV_PREFIX}/bin/svzerodsolver"

if [[ ! -d "${CODE_DIR}" ]]; then
  echo "Missing code directory: ${CODE_DIR}" >&2
  exit 1
fi

cd "${CODE_DIR}"
mkdir -p "${RESULTS_DIR}"

if [[ ! -f "${TRIAL_DIR}/combined.in" ]]; then
  echo "Missing staged trial input: ${TRIAL_DIR}/combined.in" >&2
  exit 1
fi
if [[ ! -x "${SVZERODSOLVER_BIN}" ]]; then
  echo "Missing executable: ${SVZERODSOLVER_BIN}" >&2
  exit 1
fi

# The coupling driver launches one Darcy process at a time. On savio4_htc,
# requesting extra cores also provides more memory; prevent threaded libraries
# from oversubscribing those cores.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "Job ID:       ${SLURM_JOB_ID}"
echo "Node:         ${SLURMD_NODENAME:-unknown}"
echo "Code:         ${CODE_DIR}"
echo "Scratch:      ${SCRATCH_DIR}"
echo "Trial input:  ${TRIAL_DIR}"
echo "Python:       ${PYTHON_BIN}"
echo "Started:      $(date --iso-8601=seconds)"

cases=(
  "400um"
  "1000um"
  "3000um"
  "3000um-connected"
)

for case_name in "${cases[@]}"; do
  coupled_root="${RESULTS_DIR}/${case_name}"
  perm_region_root="${CODE_DIR}/files/stl/organoid-growth-domains/${case_name}"

  echo
  echo "============================================================"
  echo "Starting case: ${case_name}"
  echo "Coupled root:  ${coupled_root}"
  echo "Region root:   ${perm_region_root}"
  echo "============================================================"

  if [[ ! -d "${perm_region_root}" ]]; then
    echo "Missing permeability-region directory: ${perm_region_root}" >&2
    exit 1
  fi

  mkdir -p "${coupled_root}"

  "${PYTHON_BIN}" \
    "${CODE_DIR}/src/coupling/coupler_hybrid_algebraic.py" \
    --template-combined "${TRIAL_DIR}/combined.in" \
    --trial-dir "${TRIAL_DIR}" \
    --seed-run0 "${TRIAL_DIR}" \
    --coupled-root "${coupled_root}" \
    --n-organoids 4 \
    --start-run 1 \
    --n-ramp 1 \
    --max-iter 250 \
    --tol-p 5e-2 \
    --terminal-bc-mode resistance \
    --resistance-calibration-run 1 \
    --no-inner-resistance-0d-search \
    --svzerodsolver "${SVZERODSOLVER_BIN}" \
    --run-and-split "${CODE_DIR}/src/prep/run_and_split_svzerod.py" \
    --darcy-script "${CODE_DIR}/src/solves/darcy_mixed.py" \
    --darcy-output-mode minimal \
    --darcy-diagnostics-mode minimal \
    --scaled-screening-mode on \
    --scaled-pressure-offset-anchor arterial \
    --first-organoid-linear-profile \
    --concave-bc-mode dirichlet \
    --concave-profile-face-length 0.4 \
    --concave-profile-compartment-spacing 0.6 \
    --dy-step 0.6 \
    --coords-inlet -0.28 0.9 0.5375 \
    --coords-outlet 0.30 0.9 0.5375 \
    --perm-region-root "${perm_region_root}" \
    --perm-low 1e-11 \
    --perm-high 2e-9 \
    --darcy-stl-anisotropy-factor 5.0 \
    --perm-transition-width 0.02 \
    --hybrid-sides both \
    --hybrid-start-run 2 \
    --hybrid-run2-equalize-arterial-to-venous \
    --hybrid-opposite-side-referenced-error \
    --hybrid-pressure-scale-floor 1.0 \
    --hybrid-organoid-deltap-scaling \
    --hybrid-organoid-deltap-gamma 0.5 \
    --hybrid-organoid-deltap-max-log-shift 2.0 \
    --hybrid-organoid-deltap-ratio-lower 0.90 \
    --hybrid-organoid-deltap-ratio-upper 1.10 \
    --hybrid-max-log-step 0.45 \
    --hybrid-moderate-log-step 0.45 \
    --hybrid-large-log-step 0.45 \
    --hybrid-extreme-log-step 0.45 \
    --hybrid-venous-max-log-step 0.45 \
    --hybrid-venous-moderate-log-step 0.45 \
    --hybrid-venous-large-log-step 0.45 \
    --hybrid-venous-extreme-log-step 0.45 \
    --hybrid-q-sensitivity-control \
    --hybrid-q-sensitivity-start-run 6 \
    --hybrid-q-sensitivity-history-window 6 \
    --hybrid-q-sensitivity-min-logq-change 0.0 \
    --hybrid-q-sensitivity-min-error-change 0.0 \
    --hybrid-q-sensitivity-sign-consistency 0.0 \
    --hybrid-q-sensitivity-ema-alpha 0.5 \
    --hybrid-q-sensitivity-alpha 0.8 \
    --hybrid-q-sensitivity-active-only \
    --hybrid-alternate-sides \
    --hybrid-alternate-start-side venous \
    --hybrid-inactive-side-alpha-scale 0.1 \
    --hybrid-side-gauge-common-mode-control \
    --no-hybrid-side-gauge-common-mode-band-gate \
    --hybrid-arterial-side-gauge-common-mode-gain 0.45 \
    --hybrid-venous-side-gauge-common-mode-gain 0.45 \
    --hybrid-side-gauge-common-mode-max-log-step 0.20

  echo "Completed case: ${case_name}"
done

echo
echo "All four cases completed successfully at $(date --iso-8601=seconds)."
