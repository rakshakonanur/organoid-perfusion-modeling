#!/usr/bin/env bash
set -uo pipefail

# Run the requested transport post-processing cases sequentially. Each 3D
# transport solve itself uses NPROCS MPI ranks (four by default).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/opt/miniconda3/envs/fenicsx-env/bin/python}"
MPIEXEC="${MPIEXEC:-/opt/miniconda3/envs/fenicsx-env/bin/mpiexec.hydra}"
NPROCS="${NPROCS:-4}"
DRY_RUN="${DRY_RUN:-0}"
THREADS_PER_RANK="${THREADS_PER_RANK:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
RUN_GROUPS="${RUN_GROUPS:-channel perfusion core}"

TRANSPORT_SCRIPT="${SCRIPT_DIR}/src/solves/3d_transport.py"
STL_BASE="${SCRIPT_DIR}/files/stl/organoid-growth-domains"
CHANNEL_BASE="${SCRIPT_DIR}/src/Results/paper-results/transport-comparison/channel"
PERFUSION_BASE="${SCRIPT_DIR}/src/Results/paper-results/perfusion-comparison/enlarged"
CORE_BASE="${SCRIPT_DIR}/src/Results/paper-results/transport-comparison/core-targeting-enlarged"

# Prevent threaded numerical libraries from oversubscribing each MPI rank.
export OMP_NUM_THREADS="${THREADS_PER_RANK}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_RANK}"
export MKL_NUM_THREADS="${THREADS_PER_RANK}"
export NUMEXPR_NUM_THREADS="${THREADS_PER_RANK}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
# adios4dolfinx can hit an off-by-one redistribution error when reading these
# 1D BP flow checkpoints with multiple ranks. The 3D mesh/solve remains fully
# distributed; only the much smaller 1D inputs are replicated on each rank.
export TRANSPORT_REPLICATE_1D_INPUTS="${TRANSPORT_REPLICATE_1D_INPUTS:-1}"

failures=()
completed=0
skipped=0

group_enabled() {
  [[ " ${RUN_GROUPS} " == *" $1 "* ]]
}

latest_run() {
  local case_dir="$1"
  local candidate base number
  local largest=-1
  local selected=""

  for candidate in "${case_dir}"/run_*; do
    [[ -d "${candidate}" ]] || continue
    base="${candidate##*/}"
    number="${base#run_}"
    [[ "${number}" =~ ^[0-9]+$ ]] || continue
    if (( number > largest )); then
      largest="${number}"
      selected="${candidate}"
    fi
  done

  [[ -n "${selected}" ]] || return 1
  printf '%s\n' "${selected}"
}

stl_name_for_case() {
  local case_name="$1"
  case "${case_name}" in
    *400um*)
      printf '%s\n' "400um"
      ;;
    *1000um*)
      printf '%s\n' "1000um"
      ;;
    *3000um-connected*|*3000um-connecred*)
      # The channel results tree currently spells "connected" as "connecred".
      printf '%s\n' "3000um-connected"
      ;;
    *3000um*)
      printf '%s\n' "3000um"
      ;;
    *)
      return 1
      ;;
  esac
}

require_files() {
  local missing=0
  local path
  for path in "$@"; do
    if [[ ! -e "${path}" ]]; then
      echo "[missing] ${path}" >&2
      missing=1
    fi
  done
  return "${missing}"
}

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${MPIEXEC}" -n "${NPROCS}" "${PY}" "${TRANSPORT_SCRIPT}" "$@"
    printf '\n'
    return 0
  fi

  "${MPIEXEC}" -n "${NPROCS}" "${PY}" "${TRANSPORT_SCRIPT}" "$@"
}

run_channel_case() {
  local case_dir="$1"
  local case_name="${case_dir##*/}"
  local run_root stl_name stl_file org

  # The channel comparison is defined at run_2. Fall back to the latest run
  # only if run_2 is absent.
  if [[ -d "${case_dir}/run_2" ]]; then
    run_root="${case_dir}/run_2"
  elif ! run_root="$(latest_run "${case_dir}")"; then
    echo "[skip] No run_* directories under ${case_dir}" >&2
    failures+=("${case_dir}: no run directory")
    return
  fi

  if ! stl_name="$(stl_name_for_case "${case_name}")"; then
    echo "[skip] Cannot determine STL for channel case ${case_name}" >&2
    failures+=("${case_dir}: unknown STL mapping")
    return
  fi

  org="${run_root}/organoid_1"
  stl_file="${STL_BASE}/${stl_name}/organoid-1.stl"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${run_root}/transport_critical_summary_organoid_1.json" ]]; then
    echo "[resume] Skipping completed case: ${run_root}"
    ((skipped += 1))
    return
  fi
  if ! require_files \
    "${org}/geometry/bioreactor.xdmf" \
    "${org}/geometry/mesh_tags.xdmf" \
    "${org}/out_darcy/u.xdmf" \
    "${stl_file}"; then
    failures+=("${run_root}: missing input")
    return
  fi

  echo
  echo "======================================================================"
  echo "Channel case: ${case_name}"
  echo "Run root:     ${run_root}"
  echo "STL:          ${stl_file}"
  echo "MPI ranks:    ${NPROCS}"
  echo "======================================================================"

  if run_command \
    --bioreactor-domain "${org}/geometry/bioreactor.xdmf" \
    --facet-file "${org}/geometry/mesh_tags.xdmf" \
    --vel-file "${org}/out_darcy/u.xdmf" \
    --skip-1d \
    --diffusion-region-path "${stl_file}" \
    --concave-exchange-mode combined \
    --c-in-value 2e-7 \
    --marker-32-concentration 2e-7 \
    --out-file "${run_root}/transport_c_organoid_1.xdmf" \
    --consumption-output-file "${run_root}/transport_consumption_organoid_1.xdmf" \
    --critical-output-file "${run_root}/transport_critical_mask_organoid_1.xdmf" \
    --critical-summary-file "${run_root}/transport_critical_summary_organoid_1.json" \
    --output-format xdmf \
    --T 5000 \
    --dt 50; then
    ((completed += 1))
  else
    failures+=("${run_root}: transport failed")
  fi
}

run_coupled_case() {
  local case_dir="$1"
  local stl_name="$2"
  local group_name="$3"
  local case_name="${case_dir##*/}"
  local run_root org stl_file

  if ! run_root="$(latest_run "${case_dir}")"; then
    echo "[skip] No run_* directories under ${case_dir}" >&2
    failures+=("${case_dir}: no run directory")
    return
  fi

  org="${run_root}/organoid_1"
  stl_file="${STL_BASE}/${stl_name}/organoid-1.stl"
  if [[ "${SKIP_EXISTING}" == "1" && -s "${run_root}/transport_critical_summary_organoid_1.json" ]]; then
    echo "[resume] Skipping completed case: ${run_root}"
    ((skipped += 1))
    return
  fi
  if ! require_files \
    "${org}/geometry/bioreactor.xdmf" \
    "${org}/geometry/mesh_tags.xdmf" \
    "${org}/geometry/tagged_branches_inlet.bp" \
    "${org}/geometry/tagged_branches_outlet.bp" \
    "${org}/geometry/flow_checkpoint_inlet.bp" \
    "${org}/geometry/flow_checkpoint_outlet.bp" \
    "${org}/out_darcy/u.xdmf" \
    "${org}/out_darcy/interface_bc.json" \
    "${stl_file}"; then
    failures+=("${run_root}: missing input")
    return
  fi

  echo
  echo "======================================================================"
  echo "${group_name} case: ${case_name}"
  echo "Run root:     ${run_root}"
  echo "STL:          ${stl_file}"
  echo "MPI ranks:    ${NPROCS}"
  echo "======================================================================"

  if run_command \
    --bioreactor-domain "${org}/geometry/bioreactor.xdmf" \
    --facet-file "${org}/geometry/mesh_tags.xdmf" \
    --mesh-inlet-file "${org}/geometry/tagged_branches_inlet.bp" \
    --mesh-outlet-file "${org}/geometry/tagged_branches_outlet.bp" \
    --flow-inlet-file "${org}/geometry/flow_checkpoint_inlet.bp" \
    --flow-outlet-file "${org}/geometry/flow_checkpoint_outlet.bp" \
    --vel-file "${org}/out_darcy/u.xdmf" \
    --interface-bc-file "${org}/out_darcy/interface_bc.json" \
    --diffusion-region-path "${stl_file}" \
    --concave-exchange-mode combined \
    --T 5000 \
    --dt 50 \
    --marker-32-concentration 2e-7 \
    --out-file "${run_root}/transport_c_organoid_1.xdmf" \
    --consumption-output-file "${run_root}/transport_consumption_organoid_1.xdmf" \
    --critical-output-file "${run_root}/transport_critical_mask_organoid_1.xdmf" \
    --critical-summary-file "${run_root}/transport_critical_summary_organoid_1.json" \
    --output-format xdmf \
    --coupled-1d-transport; then
    ((completed += 1))
  else
    failures+=("${run_root}: transport failed")
  fi
}

if [[ ! -x "${PY}" ]]; then
  echo "Python executable not found: ${PY}" >&2
  exit 2
fi
if [[ ! -x "${MPIEXEC}" ]]; then
  echo "MPI launcher not found: ${MPIEXEC}" >&2
  exit 2
fi
if [[ ! "${NPROCS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROCS must be a positive integer; received: ${NPROCS}" >&2
  exit 2
fi

echo "Transport script: ${TRANSPORT_SCRIPT}"
echo "Python:           ${PY}"
echo "MPI launcher:     ${MPIEXEC}"
echo "MPI ranks/solve:  ${NPROCS}"
echo "Case groups:      ${RUN_GROUPS}"
[[ "${DRY_RUN}" == "1" ]] && echo "Mode:             dry run"
[[ "${SKIP_EXISTING}" == "1" ]] && echo "Resume mode:      skip completed summaries"

# 1. Channel comparison: use run_2 when available and the matching STL.
if group_enabled channel; then
  for case_dir in "${CHANNEL_BASE}"/*; do
    [[ -d "${case_dir}" ]] || continue
    run_channel_case "${case_dir}"
  done
fi

# 2. Perfusion comparison: use each case's latest run and the 3000um STL.
if group_enabled perfusion; then
  for case_dir in "${PERFUSION_BASE}"/*; do
    [[ -d "${case_dir}" ]] || continue
    run_coupled_case "${case_dir}" "3000um" "Perfusion"
  done
fi

# 3. Core-targeting comparison: latest run and the corresponding STL.
if group_enabled core; then
  for stl_name in 400um 1000um 3000um 3000um-connected; do
    case_dir="${CORE_BASE}/${stl_name}"
    if [[ ! -d "${case_dir}" ]]; then
      echo "[skip] Missing core-targeting case directory: ${case_dir}" >&2
      failures+=("${case_dir}: missing case directory")
      continue
    fi
    run_coupled_case "${case_dir}" "${stl_name}" "Core-targeting"
  done
fi

echo
echo "======================================================================"
echo "Completed transport solves: ${completed}"
echo "Previously completed/skipped: ${skipped}"
if (( ${#failures[@]} > 0 )); then
  echo "Failures/skips: ${#failures[@]}" >&2
  for failure in "${failures[@]}"; do
    echo "  - ${failure}" >&2
  done
  exit 1
fi

echo "All requested transport solves completed successfully."
