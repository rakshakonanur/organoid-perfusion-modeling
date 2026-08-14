#!/usr/bin/env bash
set -euo pipefail

cases=(
  "400um"
  "1000um"
  "3000um"
  "3000um-connected"
)

for case_name in "${cases[@]}"; do
  coupled_root="src/Results/paper-results/transport-comparison/core-targeting-enlarged/${case_name}"
  perm_region_root="files/stl/organoid-growth-domains/${case_name}"

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

  /opt/miniconda3/envs/fenicsx-env/bin/python \
    src/coupling/coupler_hybrid_algebraic.py \
    --template-combined src/prep/scaled-screening/081226/combined.in \
    --trial-dir src/prep/scaled-screening/081226 \
    --seed-run0 src/prep/scaled-screening/081226 \
    --coupled-root "${coupled_root}" \
    --n-organoids 4 \
    --start-run 1 \
    --n-ramp 1 \
    --max-iter 250 \
    --tol-p 5e-2 \
    --terminal-bc-mode resistance \
    --resistance-calibration-run 1 \
    --no-inner-resistance-0d-search \
    --svzerodsolver /opt/miniconda3/envs/fenicsx-env/bin/svzerodsolver \
    --run-and-split src/prep/run_and_split_svzerod.py \
    --darcy-script src/solves/darcy_mixed.py \
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
    --hybrid-q-sensitivity-history-window 12 \
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
echo "All four cases completed successfully."