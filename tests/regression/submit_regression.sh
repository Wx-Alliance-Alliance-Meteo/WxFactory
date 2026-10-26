#!/usr/bin/env bash
#
# submit_regression.sh — launch the full WxFactory regression suite as a set of
# parallel PBS jobs that all write to ONE shared output directory, plus a final
# aggregation job that produces the single report + gallery.
#
# What it does:
#   1. Sources an environment-loading script (module load, python environment,
#      environment variables) — see Usage below.
#   2. Generates a shared run-id (UTC timestamp + git sha) and the shared
#      output directory:  results/regression/<run-id>
#   3. Submits one regression_case.job per hardcoded group below. Each group
#      passes its own resources (select / walltime / job name) on the qsub
#      command line and its config list via REGRESSION_CASES.
#   4. Computes the "remainder" group: every discovered runnable config that is
#      not in a hardcoded group (catches newly added configs). Submitted only
#      when non-empty.
#   5. Submits the aggregation job with a PBS dependency on all of them; it
#      writes the single manifest / report.html / gallery entry.
#
# Usage:
#   ./submit_regression.sh [--dry-run] [env-script]
#
#   env-script  a script that is SOURCED to set up the environment (module
#               load, python environment, environment variables). It is also
#               forwarded to the jobs (WX_ENV_SCRIPT), which source it on the
#               compute node. Defaults to ${WX_ENV_SCRIPT} if set.
#
#   --dry-run   print the plan, submit nothing.
#
# To add a config to a group, add its .ini file name to the matching CASES
# array below. A config that is in no group is picked up by the remainder job.

set -euo pipefail

# This script lives in tests/regression/; the repo root is two levels up.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# --- Arguments ---------------------------------------------------------------
DRY_RUN=0
ENV_SCRIPT="${WX_ENV_SCRIPT:-}"
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        *) ENV_SCRIPT="${arg}" ;;
    esac
done
if [[ -z "${ENV_SCRIPT}" ]]; then
    echo "ERROR: no environment script given." >&2
    echo "       Usage: $0 [--dry-run] <env-script>" >&2
    echo "       (or set WX_ENV_SCRIPT to the script that is sourced to load the" >&2
    echo "        module / python environment / environment variables)" >&2
    exit 1
fi
if [[ ! -f "${ENV_SCRIPT}" ]]; then
    echo "ERROR: environment script not found: ${ENV_SCRIPT}" >&2
    exit 1
fi

# shellcheck disable=SC1090
. "${ENV_SCRIPT}"

CASE_JOB="${SCRIPT_DIR}/regression_case.job"
AGGREGATE_JOB="${SCRIPT_DIR}/regression_aggregate.job"

# --- Guard: do not launch if a regression run is already in flight ----------
# We must not flood the cluster with a second full suite while the previous one is still running.
# Every compute job is named wx_regression_*, so qstat is the source of truth (self-cleaning if a
# job dies). A flock serializes the check-and-submit so two concurrent launches
# cannot both pass the check. On a conflict we exit 1 so the CI job fails
# (allow_failure keeps the pipeline green) — a red job means "this pipeline did
# not generate results". Skipped for --dry-run.
if [[ "${DRY_RUN}" -ne 1 ]]; then
    LOCK_DIR="${REPO_ROOT}/results/regression"
    mkdir -p "${LOCK_DIR}"
    LOCK_FILE="${LOCK_DIR}/.launch.lock"
    exec 200>"${LOCK_FILE}"
    if ! flock -n 200; then
        echo "ERROR: another regression launch is in progress (could not acquire ${LOCK_FILE})." >&2
        exit 1
    fi
    if qstat -u "${USER}" 2>/dev/null | grep -q 'wx_regression_'; then
        echo "ERROR: a regression run is already in flight (found wx_regression_* jobs in qstat)." >&2
        echo "       Not launching a second one; the in-flight run will publish its own results." >&2
        exit 1
    fi
fi

# --- Shared run-id + output directory ---------------------------------------
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"
# Base dir for the per-run output. Overridable via WX_REGRESSION_ROOT. The compute jobs receive the full
# path via REGRESSION_OUTPUT_ROOT, so they write here automatically.
REGRESSION_BASE="${WX_REGRESSION_ROOT:-results/regression}"
OUTPUT_ROOT="${REGRESSION_BASE}/${RUN_ID}"

# --- Job groups --------------------------------------------------------------
# Each group: name, select spec, walltime, and its config file names.
GROUP_NAMES=(cartesian1 cartesian2 sw dcmip dcmip21 dcmip21_int dcmip200 dcmip200_ROSEXP)
GROUP_SELECTS=(
    "1:ncpus=24:mpiprocs=24:ompthreads=1:mem=100G:vntype=gpu:ngpus=1"
    "1:ncpus=24:mpiprocs=24:ompthreads=1:mem=100G:vntype=gpu:ngpus=1"
    "3:ncpus=48:mpiprocs=2:ompthreads=1:mem=200G:vntype=gpu:ngpus=2"
    "3:ncpus=48:mpiprocs=2:ompthreads=1:mem=200G:vntype=gpu:ngpus=2"
    "3:ncpus=48:mpiprocs=2:ompthreads=1:mem=200G:vntype=gpu:ngpus=2"
    "3:ncpus=48:mpiprocs=2:ompthreads=1:mem=200G:vntype=gpu:ngpus=2"
    "3:ncpus=48:mpiprocs=2:ompthreads=1:mem=200G:vntype=gpu:ngpus=2"
    "3:ncpus=48:mpiprocs=2:ompthreads=1:mem=200G:vntype=gpu:ngpus=2"
)
GROUP_WALLTIMES=("3:0:0" "1:0:0" "1:0:0" "6:0:0" "2:0:0" "2:0:0" "9:0:0" "9:0:0")
GROUP_CASES=(
    "cold_bubble.ini density_current.ini gresho_vortex.ini pill_bubble.ini"
    "gaussian_bubble.ini"
    "case1.ini case2.ini case5.ini case6.ini galewsky.ini matsuno.ini unsteady_zonal.ini vortex.ini"
    "dcmip11.ini dcmip12.ini dcmip13.ini dcmip201.ini dcmip201_ROSEXP.ini dcmip202_flat_ROSEXP.ini dcmip31.ini dcmip31_EPI2.ini dcmip31_ROSEXP.ini dcmip41.ini"
    "dcmip21.ini"
    "dcmip21_EPI2.ini dcmip21_ROSEXP.ini dcmip21_STRANG.ini"
    "dcmip200.ini"
    "dcmip200_ROSEXP.ini"
)
# Resources for the remainder job (only submitted when it is non-empty).
REMAINDER_SELECT="3:ncpus=48:mpiprocs=2:ompthreads=1:mem=200G:vntype=gpu:ngpus=2"
REMAINDER_WALLTIME="6:0:0"

# --- Remainder: discovered runnable configs not in any group -----------------
RUNNABLE="$(python tests/regression/run_regression.py --list-runnable)"

assigned=()
for cases in "${GROUP_CASES[@]}"; do
    for c in ${cases}; do
        assigned+=("${c}")
    done
done

remainder=()
for c in ${RUNNABLE}; do
    found=0
    for a in ${assigned[@]+"${assigned[@]}"}; do
        if [[ "${c}" == "${a}" ]]; then
            found=1
            break
        fi
    done
    if [[ "${found}" -eq 0 ]]; then
        remainder+=("${c}")
    fi
done

# --- Plan --------------------------------------------------------------------
echo "=== WxFactory regression submission ==="
echo "run-id:        ${RUN_ID}"
echo "output root:   ${OUTPUT_ROOT}"
for i in "${!GROUP_NAMES[@]}"; do
    echo "group ${GROUP_NAMES[$i]}: ${GROUP_CASES[$i]}"
done
if [[ ${#remainder[@]} -gt 0 ]]; then
    echo "group remainder: ${remainder[*]}"
else
    echo "group remainder: (empty — not submitted)"
fi
echo "aggregate job: ${AGGREGATE_JOB}"
echo "========================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[dry-run] would submit the groups above (one ${CASE_JOB} each), then the"
    echo "[dry-run] aggregation job with depend=afterany on all of them."
    exit 0
fi

# --- Register the run in the gallery (placeholder) ---------------------------
# Create an "in progress" entry so the run is visible immediately; the
# aggregation job's publish() later replaces the stub with the real report.
# (Done after the dry-run exit so --dry-run does not create a real entry.)
python tests/regression/run_regression.py --register-run "${RUN_ID}"

# --- Submit the case jobs, capturing their IDs -------------------------------
job_ids=()
group_labels=()
submit_group() {
    local name="$1" select="$2" walltime="$3" cases="$4"
    local -a case_arr
    read -r -a case_arr <<< "${cases}"
    echo "submitting ${name} (${#case_arr[@]} configs) ..."
    local out id
    # NOTE: qsub -v takes a SINGLE comma-separated list (see `man qsub`).
    # The cases value contains spaces, so it is wrapped in single quotes.
    out="$(qsub -l "select=${select}" -l "walltime=${walltime}" \
        -N "wx_regression_${name}" \
        -v "REGRESSION_CASES='${cases}',REGRESSION_OUTPUT_ROOT=${OUTPUT_ROOT},WX_ENV_SCRIPT='${ENV_SCRIPT}',WX_REPO_ROOT=${REPO_ROOT}" \
        "${CASE_JOB}")"
    # qsub prints "<id>.<server>"; keep the numeric id.
    id="$(echo "${out}" | grep -oE '^[0-9]+' || true)"
    if [[ -z "${id}" ]]; then
        echo "ERROR: could not parse job id from qsub output: ${out}" >&2
        exit 1
    fi
    job_ids+=("${id}")
    group_labels+=("${name}")
    echo "  -> job ${id}"
}

for i in "${!GROUP_NAMES[@]}"; do
    submit_group "${GROUP_NAMES[$i]}" "${GROUP_SELECTS[$i]}" "${GROUP_WALLTIMES[$i]}" "${GROUP_CASES[$i]}"
done
if [[ ${#remainder[@]} -gt 0 ]]; then
    submit_group "remainder" "${REMAINDER_SELECT}" "${REMAINDER_WALLTIME}" "${remainder[*]}"
fi

# --- Build the dependency string --------------------------------------------
depend="afterany:${job_ids[0]}"
for id in "${job_ids[@]:1}"; do
    depend="${depend}:${id}"
done
echo "aggregation dependency: ${depend}"

# --- Submit the aggregation job ---------------------------------------------
# The dependency and shared variables are passed on the command line.
# Forward the gallery dir to the aggregation job (it reads WX_REGRESSION_GALLERY
# to know where to publish). Only passed when set, so the Python default applies
# otherwise (an empty value would otherwise shadow the default).
GALLERY_VAR=""
if [[ -n "${WX_REGRESSION_GALLERY:-}" ]]; then
    GALLERY_VAR=",WX_REGRESSION_GALLERY='${WX_REGRESSION_GALLERY}'"
fi
echo "submitting aggregation job (depends on ${#job_ids[@]} jobs) ..."
agg_out="$(qsub -W "depend=${depend}" \
    -v "REGRESSION_OUTPUT_ROOT=${OUTPUT_ROOT},WX_ENV_SCRIPT='${ENV_SCRIPT}',WX_REPO_ROOT=${REPO_ROOT}${GALLERY_VAR}" \
    "${AGGREGATE_JOB}")"
agg_id="$(echo "${agg_out}" | grep -oE '^[0-9]+' || true)"
if [[ -z "${agg_id}" ]]; then
    echo "ERROR: could not parse aggregation job id from qsub output: ${agg_out}" >&2
    exit 1
fi
echo "  -> aggregation job ${agg_id}"

echo
echo "=== submitted ==="
for i in "${!job_ids[@]}"; do
    echo "  ${group_labels[$i]}: ${job_ids[$i]}"
done
echo "  ${AGGREGATE_JOB}: ${agg_id}  (runs after all of the above)"
echo
echo "Shared output:  ${OUTPUT_ROOT}"
echo "Gallery (when done): ${WX_REGRESSION_GALLERY:-~/public_html/wx_factory/regression}/${RUN_ID}"
echo "Track with:  qstat ${job_ids[*]} ${agg_id}"
