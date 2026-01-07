#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: 2026-01-07
# Author: ywatanabe
# File: measure-install-time.sh
#
# Measure installation time for scitex modules using pip or uv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
USE_UV=false
MODULE=""
EXP_BASE_DIR="/tmp/exp-scitex"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS] MODULE

Measure installation time for a scitex module.

Arguments:
    MODULE          Module extra to install (e.g., io, ml, all)

Options:
    -u, --uv        Use uv instead of pip (default: pip)
    -h, --help      Show this help message
    -v, --verbose   Verbose output

Examples:
    $(basename "$0") io              # Measure pip install time for [io]
    $(basename "$0") --uv io         # Measure uv install time for [io]
    $(basename "$0") --uv all        # Measure uv install time for [all]
EOF
    exit "${1:-0}"
}

log_info() {
    echo "[INFO] $*"
}

log_error() {
    echo "[ERROR] $*" >&2
}

prepare_exp_dir() {
    local suffix=$1
    EXP_DIR="${EXP_BASE_DIR}-${suffix}"
    rm -rf "$EXP_DIR"
    mkdir -p "$EXP_DIR"
    cd "$EXP_DIR"
    log_info "Created experiment directory: $EXP_DIR"
}

cleanup_exp_dir() {
    local suffix=$1
    cd "$HOME"
    EXP_DIR="${EXP_BASE_DIR}-${suffix}"
    rm -rf "$EXP_DIR"
    log_info "Cleaned up: $EXP_DIR"
}

measure_with_pip() {
    local module=$1
    log_info "Setting up pip virtual environment..."
    python -m venv .venv
    # shellcheck source=/dev/null
    source .venv/bin/activate
    pip install -U pip -q
    pip cache purge -q 2>/dev/null || true

    log_info "Installing scitex[$module] with pip..."
    echo "---"
    time pip install -e "${PROJECT_ROOT}[$module]"
    echo "---"

    deactivate 2>/dev/null || true
}

measure_with_uv() {
    local module=$1

    if ! command -v uv &>/dev/null; then
        log_error "uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi

    log_info "Setting up uv virtual environment..."
    uv venv
    # shellcheck source=/dev/null
    source .venv/bin/activate
    uv cache clean 2>/dev/null || true

    log_info "Installing scitex[$module] with uv..."
    echo "---"
    time uv pip install -e "${PROJECT_ROOT}[$module]"
    echo "---"

    deactivate 2>/dev/null || true
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
        -u | --uv)
            USE_UV=true
            shift
            ;;
        -h | --help)
            usage 0
            ;;
        -v | --verbose)
            set -x
            shift
            ;;
        -*)
            log_error "Unknown option: $1"
            usage 1
            ;;
        *)
            MODULE=$1
            shift
            ;;
        esac
    done

    # Validate module
    if [[ -z "$MODULE" ]]; then
        log_error "MODULE is required"
        usage 1
    fi

    # Determine installer
    local installer
    if [[ "$USE_UV" == "true" ]]; then
        installer="uv"
    else
        installer="pip"
    fi

    local suffix="${installer}-${MODULE}"

    log_info "Measuring installation time for scitex[$MODULE] using $installer"

    prepare_exp_dir "$suffix"

    if [[ "$USE_UV" == "true" ]]; then
        measure_with_uv "$MODULE"
    else
        measure_with_pip "$MODULE"
    fi

    cleanup_exp_dir "$suffix"

    log_info "Done!"
}

main "$@"
