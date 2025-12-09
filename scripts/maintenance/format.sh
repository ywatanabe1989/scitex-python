#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09"
# File: ./scripts/maintenance/format.sh

set -e

THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
ROOT_DIR="$(realpath $THIS_DIR/../..)"

# Color scheme
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

cd "$ROOT_DIR"

########################################
# Usage
########################################

usage() {
    echo "Usage: $0 [options] [path...]"
    echo ""
    echo "Format Python code using ruff."
    echo ""
    echo "Options:"
    echo "  -c, --check      Check formatting without modifying files"
    echo "  -d, --diff       Show diff of what would change"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Paths (optional):"
    echo "  Specific files or directories to format."
    echo "  Default: src/ tests/"
    echo ""
    echo "Examples:"
    echo "  $0                    # Format all code"
    echo "  $0 --check            # Check without modifying"
    echo "  $0 --diff             # Show what would change"
    echo "  $0 src/scitex/config  # Format specific directory"
}

########################################
# Main
########################################

main() {
    local check=false
    local diff=false
    local paths=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--check)
                check=true
                shift
                ;;
            -d|--diff)
                diff=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                echo_error "Unknown option: $1"
                usage
                exit 1
                ;;
            *)
                paths+=("$1")
                shift
                ;;
        esac
    done

    # Default paths
    if [ ${#paths[@]} -eq 0 ]; then
        paths=("src/" "tests/")
    fi

    echo_header "Format (ruff)"
    echo_info "Paths: ${paths[*]}"
    echo ""

    # Check if ruff is installed
    if ! command -v ruff &> /dev/null; then
        echo_warning "ruff not found, installing..."
        pip install ruff
        echo ""
    fi

    # Build command
    local cmd="ruff format"

    if [ "$check" = true ]; then
        cmd="$cmd --check"
        echo_info "Mode: Check"
    elif [ "$diff" = true ]; then
        cmd="$cmd --diff"
        echo_info "Mode: Diff"
    else
        echo_info "Mode: Format"
    fi

    cmd="$cmd ${paths[*]}"
    echo_info "Command: $cmd"
    echo ""

    # Run formatter
    if eval "$cmd"; then
        echo ""
        if [ "$check" = true ]; then
            echo_success "Format check passed"
        elif [ "$diff" = true ]; then
            echo_success "No formatting changes needed"
        else
            echo_success "Formatting complete"
        fi
    else
        local exit_code=$?
        echo ""
        if [ "$check" = true ]; then
            echo_warning "Files need formatting (run without --check to format)"
        fi
        exit $exit_code
    fi
}

main "$@"

# EOF
