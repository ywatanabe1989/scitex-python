#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: 2026-01-22
# File: src/scitex/scholar/examples/00_run_all.sh
# ----------------------------------------
# Run all scholar examples in sequence
# ----------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Run all scholar module examples in sequence.

Options:
    -h, --help      Show this help message
    --dry-run       Show commands without executing
    --skip-auth     Skip authentication example (01_auth.py)
    --quick         Run only quick examples (00, 01, 02)

Examples:
    $(basename "$0")              # Run all examples
    $(basename "$0") --quick      # Run quick examples only
    $(basename "$0") --dry-run    # Show what would run

EOF
    exit 0
}

DRY_RUN=false
SKIP_AUTH=false
QUICK=false

while [[ $# -gt 0 ]]; do
    case $1 in
    -h | --help) usage ;;
    --dry-run)
        DRY_RUN=true
        shift
        ;;
    --skip-auth)
        SKIP_AUTH=true
        shift
        ;;
    --quick)
        QUICK=true
        shift
        ;;
    *)
        echo "Unknown option: $1"
        usage
        ;;
    esac
done

run_example() {
    local script="$1"
    shift
    local args=("$@")

    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        echo "[SKIP] $script not found"
        return 0
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $script ${args[*]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] python $SCRIPT_DIR/$script ${args[*]}"
    else
        python "$SCRIPT_DIR/$script" "${args[@]}"
    fi
}

echo "SciTeX Scholar Examples Runner"
echo "==============================="

# Core examples
run_example "00_config.py"

if [[ "$SKIP_AUTH" != "true" ]]; then
    run_example "01_auth.py"
fi

run_example "02_browser.py"

if [[ "$QUICK" == "true" ]]; then
    echo ""
    echo "Quick mode: Stopping after basic examples"
    exit 0
fi

# Engine examples
run_example "03_01-engine.py"
run_example "03_02-engine-for-bibtex.py" --no-cache

# URL finding examples
run_example "04_01-url.py" --no-cache
run_example "04_02-url-for-bibtex.py" --no-cache-url-finder --n-samples 3

# Download examples
run_example "05_download_pdf.py"
run_example "06_find_and_download.py"

# Storage integration
run_example "07_storage_integration.py"

echo ""
echo "==============================="
echo "All examples completed!"
echo "==============================="

# EOF
