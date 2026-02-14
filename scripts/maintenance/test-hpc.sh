#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2026-02-14 22:00:00 (ywatanabe)"
# File: ./scripts/maintenance/test-hpc.sh
#
# Run pytest on HPC via Slurm.
# Usage:
#   test-hpc.sh              # Sync + blocking srun (default)
#   test-hpc.sh --async      # Sync + sbatch, returns job ID
#   test-hpc.sh --poll JID   # Check job status
#   test-hpc.sh --result JID # Fetch output from completed job
#
# Environment:
#   HPC_HOST=spartan  HPC_CPUS=8  HPC_PARTITION=sapphire
#   HPC_TIME=00:10:00  HPC_MEM=16G  REMOTE_BASE=~/proj

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_PATH="$THIS_DIR/.$(basename "$0").log"

GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
PROJECT="$(basename "$GIT_ROOT")"

# Color scheme (matches test.sh)
GRAY='\033[0;90m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_info() { echo -e "${GRAY}INFO: $1${NC}"; }
echo_success() { echo -e "${GREEN}SUCC: $1${NC}"; }
echo_warning() { echo -e "${YELLOW}WARN: $1${NC}"; }
echo_error() { echo -e "${RED}ERRO: $1${NC}"; }
echo_header() { echo -e "\n${GRAY}=== $1 ===${NC}\n"; }

# HPC configuration
HPC_HOST="${HPC_HOST:-spartan}"
HPC_CPUS="${HPC_CPUS:-8}"
HPC_PARTITION="${HPC_PARTITION:-sapphire}"
HPC_TIME="${HPC_TIME:-00:10:00}"
HPC_MEM="${HPC_MEM:-16G}"
REMOTE_BASE="${REMOTE_BASE:-~/proj}"
REMOTE_OUT="${REMOTE_BASE}/${PROJECT}/.pytest-hpc-output"

# -----------------------------------------------
# Functions
# -----------------------------------------------

check_ssh() {
    if ! ssh -o ConnectTimeout=5 "${HPC_HOST}" true 2>/dev/null; then
        echo_error "Cannot connect to ${HPC_HOST}"
        exit 255
    fi
}

do_sync() {
    echo_header "Syncing to ${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/"
    rsync -az --delete \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.eggs' \
        --exclude='*.egg-info' \
        --exclude='dist' \
        --exclude='build' \
        --exclude='docs/sphinx/_build' \
        --exclude='.tox' \
        --exclude='.mypy_cache' \
        --exclude='.pytest_cache' \
        --exclude='*_out' \
        --exclude='GITIGNORED' \
        --exclude='.pytest-hpc-output' \
        "$GIT_ROOT/" "${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/" 2>&1 | tee -a "$LOG_PATH"

    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        echo_error "rsync failed"
        exit 1
    fi
    echo_success "Sync complete"
}

do_srun() {
    echo_header "Running pytest on ${HPC_HOST} (srun: ${HPC_CPUS} CPUs, ${HPC_PARTITION})"

    # shellcheck disable=SC2029
    ssh "${HPC_HOST}" "bash -lc 'srun \
        --partition=${HPC_PARTITION} \
        --cpus-per-task=${HPC_CPUS} \
        --time=${HPC_TIME} \
        --mem=${HPC_MEM} \
        --job-name=pytest-${PROJECT} \
        bash -lc \"cd ${REMOTE_BASE}/${PROJECT} && pip install -e .[dev] \
-q && python -m pytest tests/ -n ${HPC_CPUS} --dist loadfile -x --tb=short\"'" 2>&1 | tee -a "$LOG_PATH"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo_success "All tests passed on ${HPC_HOST}"
    else
        echo_error "Tests failed on ${HPC_HOST} (exit code: ${EXIT_CODE})"
    fi
    exit "$EXIT_CODE"
}

do_async() {
    echo_header "Submitting async pytest job on ${HPC_HOST}"

    # shellcheck disable=SC2029
    JOB_ID=$(ssh "${HPC_HOST}" "bash -lc '
        mkdir -p ${REMOTE_OUT}
        sbatch --parsable \
            --partition=${HPC_PARTITION} \
            --cpus-per-task=${HPC_CPUS} \
            --time=${HPC_TIME} \
            --mem=${HPC_MEM} \
            --job-name=pytest-${PROJECT} \
            --output=${REMOTE_OUT}/%j.out \
            --error=${REMOTE_OUT}/%j.err \
            --wrap=\"bash -lc \\\"cd ${REMOTE_BASE}/${PROJECT} && pip install -e .[dev] -q --no-deps && python -m pytest tests/ -n ${HPC_CPUS} --dist loadfile -x --tb=short\\\"\"
    '")

    echo_success "Job submitted: ${JOB_ID}"
    echo "${JOB_ID}" >"$THIS_DIR/.last-hpc-job"
    echo "$JOB_ID"
}

do_poll() {
    local JOB_ID="$1"
    echo_info "Polling job ${JOB_ID} on ${HPC_HOST}..."

    local STATE
    # shellcheck disable=SC2029
    STATE=$(ssh "${HPC_HOST}" "bash -lc 'sacct -j ${JOB_ID} --format=State --noheader -P | head -1'" 2>/dev/null | tr -d '[:space:]')

    case "$STATE" in
    COMPLETED)
        echo_success "Job ${JOB_ID}: COMPLETED"
        local TMPOUT="/tmp/pytest-hpc-${JOB_ID}.out"
        scp -q "${HPC_HOST}:${REMOTE_OUT}/${JOB_ID}.out" "$TMPOUT" 2>/dev/null
        if [ -f "$TMPOUT" ]; then
            echo ""
            tail -20 "$TMPOUT"
        fi
        exit 0
        ;;
    FAILED | OUT_OF_ME* | TIMEOUT | CANCELLED*)
        echo_error "Job ${JOB_ID}: ${STATE}"
        local TMPOUT="/tmp/pytest-hpc-${JOB_ID}.out"
        scp -q "${HPC_HOST}:${REMOTE_OUT}/${JOB_ID}.out" "$TMPOUT" 2>/dev/null
        if [ -f "$TMPOUT" ]; then
            echo ""
            tail -30 "$TMPOUT"
        fi
        exit 1
        ;;
    PENDING | RUNNING)
        echo_info "Job ${JOB_ID}: ${STATE}"
        exit 2
        ;;
    *)
        echo_warning "Job ${JOB_ID}: unknown state '${STATE}'"
        exit 2
        ;;
    esac
}

do_watch() {
    local JOB_ID="$1"
    local INTERVAL="${2:-15}"
    echo_info "Watching job ${JOB_ID} (poll every ${INTERVAL}s)..."

    while true; do
        local STATE
        local RAW_STATE
        # shellcheck disable=SC2029
        RAW_STATE=$(ssh "${HPC_HOST}" "bash -lc 'sacct -j ${JOB_ID} --format=State --noheader -P | head -1'" 2>/dev/null)
        STATE=$(echo "$RAW_STATE" | grep -oE '(COMPLETED|FAILED|RUNNING|PENDING|TIMEOUT|CANCELLED|OUT_OF_ME)' | head -1)

        case "$STATE" in
        COMPLETED)
            echo_success "Job ${JOB_ID}: COMPLETED"
            local TMPOUT="/tmp/pytest-hpc-${JOB_ID}.out"
            scp -q "${HPC_HOST}:${REMOTE_OUT}/${JOB_ID}.out" "$TMPOUT" 2>/dev/null
            if [ -f "$TMPOUT" ]; then
                local SUMMARY
                SUMMARY=$(tail -5 "$TMPOUT")
                echo ""
                echo "$SUMMARY"
            fi
            exit 0
            ;;
        FAILED | OUT_OF_ME* | TIMEOUT | CANCELLED*)
            echo_error "Job ${JOB_ID}: ${STATE}"
            local TMPOUT="/tmp/pytest-hpc-${JOB_ID}.out"
            scp -q "${HPC_HOST}:${REMOTE_OUT}/${JOB_ID}.out" "$TMPOUT" 2>/dev/null
            if [ -f "$TMPOUT" ]; then
                echo ""
                tail -30 "$TMPOUT"
            fi
            exit 1
            ;;
        PENDING | RUNNING)
            echo -ne "\r${GRAY}INFO: Job ${JOB_ID}: ${STATE} (${INTERVAL}s)${NC}  "
            ;;
        esac
        sleep "$INTERVAL"
    done
}

do_result() {
    local JOB_ID="$1"
    local TMPOUT="/tmp/pytest-hpc-${JOB_ID}.out"

    # shellcheck disable=SC2029
    scp -q "${HPC_HOST}:${REMOTE_OUT}/${JOB_ID}.out" "$TMPOUT" 2>/dev/null
    scp -q "${HPC_HOST}:${REMOTE_OUT}/${JOB_ID}.err" "/tmp/pytest-hpc-${JOB_ID}.err" 2>/dev/null

    if [ -f "$TMPOUT" ]; then
        cat "$TMPOUT"
    else
        echo_error "No output found for job ${JOB_ID}"
        exit 1
    fi
}

# -----------------------------------------------
# Main
# -----------------------------------------------

echo >"$LOG_PATH"
MODE="${1:-sync}"

case "$MODE" in
--async | -a)
    check_ssh
    do_sync
    do_async
    ;;
--poll | -p)
    JOB_ID="${2:-$(cat "$THIS_DIR/.last-hpc-job" 2>/dev/null || true)}"
    if [ -z "$JOB_ID" ]; then
        echo_error "No job ID. Usage: $0 --poll <JOB_ID>"
        exit 1
    fi
    check_ssh
    do_poll "$JOB_ID"
    ;;
--watch | -w)
    JOB_ID="${2:-$(cat "$THIS_DIR/.last-hpc-job" 2>/dev/null || true)}"
    INTERVAL="${3:-15}"
    if [ -z "$JOB_ID" ]; then
        echo_error "No job ID. Usage: $0 --watch <JOB_ID> [interval]"
        exit 1
    fi
    check_ssh
    do_watch "$JOB_ID" "$INTERVAL"
    ;;
--result | -r)
    JOB_ID="${2:-$(cat "$THIS_DIR/.last-hpc-job" 2>/dev/null || true)}"
    if [ -z "$JOB_ID" ]; then
        echo_error "No job ID. Usage: $0 --result <JOB_ID>"
        exit 1
    fi
    check_ssh
    do_result "$JOB_ID"
    ;;
sync | --sync | -s | "")
    check_ssh
    do_sync
    do_srun
    ;;
-h | --help)
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Modes:"
    echo "  (default)       Sync + blocking srun"
    echo "  --async, -a       Sync + sbatch (returns job ID)"
    echo "  --watch, -w JID   Poll until done (every 15s)"
    echo "  --poll, -p JID    Check job status once"
    echo "  --result, -r JID  Fetch full output"
    echo ""
    echo "Environment:"
    echo "  HPC_HOST=${HPC_HOST}  HPC_CPUS=${HPC_CPUS}"
    echo "  HPC_PARTITION=${HPC_PARTITION}  HPC_MEM=${HPC_MEM}"
    echo "  HPC_TIME=${HPC_TIME}  REMOTE_BASE=${REMOTE_BASE}"
    ;;
*)
    echo_error "Unknown mode: $MODE"
    echo "Use --help for usage"
    exit 1
    ;;
esac

# EOF
