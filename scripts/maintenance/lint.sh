#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09"
# File: ./scripts/maintenance/lint.sh

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
    echo "Lint Python code using ruff."
    echo ""
    echo "Options:"
    echo "  -f, --fix        Auto-fix fixable issues"
    echo "  -w, --watch      Watch for changes and re-lint"
    echo "  -s, --stats      Show statistics only"
    echo "  -h, --help       Show this help"
    echo ""
    echo "Paths (optional):"
    echo "  Specific files or directories to lint."
    echo "  Default: src/ tests/"
    echo ""
    echo "Examples:"
    echo "  $0                    # Lint all code"
    echo "  $0 --fix              # Lint and auto-fix"
    echo "  $0 src/scitex/config  # Lint specific directory"
    echo "  $0 --watch            # Watch mode"
}

########################################
# Main
########################################

main() {
    local fix=false
    local watch=false
    local stats=false
    local paths=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--fix)
                fix=true
                shift
                ;;
            -w|--watch)
                watch=true
                shift
                ;;
            -s|--stats)
                stats=true
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

    echo_header "Lint (ruff)"
    echo_info "Paths: ${paths[*]}"
    echo ""

    # Check if ruff is installed
    if ! command -v ruff &> /dev/null; then
        echo_warning "ruff not found, installing..."
        pip install ruff
        echo ""
    fi

    # Build command
    local cmd="ruff check"

    if [ "$fix" = true ]; then
        cmd="$cmd --fix"
        echo_info "Mode: Fix"
    elif [ "$watch" = true ]; then
        cmd="$cmd --watch"
        echo_info "Mode: Watch"
    elif [ "$stats" = true ]; then
        cmd="$cmd --statistics"
        echo_info "Mode: Statistics"
    else
        echo_info "Mode: Check"
    fi

    cmd="$cmd ${paths[*]}"
    echo_info "Command: $cmd"
    echo ""

    # Run linter
    if eval "$cmd"; then
        echo ""
        echo_success "Lint passed"
    else
        local exit_code=$?
        echo ""
        if [ "$fix" = true ]; then
            echo_warning "Some issues could not be auto-fixed"
        else
            echo_warning "Lint issues found (run with --fix to auto-fix)"
        fi
        exit $exit_code
    fi
}

main "$@"

# EOF
