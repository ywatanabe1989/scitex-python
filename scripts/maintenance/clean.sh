#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09"
# File: ./scripts/maintenance/clean.sh

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
# Clean Functions
########################################

clean_build() {
    echo_info "Cleaning build artifacts..."
    rm -rf build/ dist/ .eggs/ 2>/dev/null || true
    find . -name '*.egg-info' -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name '*.egg' -type f -delete 2>/dev/null || true
    echo_success "Build artifacts cleaned"
}

clean_pyc() {
    echo_info "Cleaning Python cache..."
    find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
    find . -name '*.pyc' -type f -delete 2>/dev/null || true
    find . -name '*.pyo' -type f -delete 2>/dev/null || true
    echo_success "Python cache cleaned"
}

clean_test() {
    echo_info "Cleaning test artifacts..."
    rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ 2>/dev/null || true
    rm -f coverage*.json coverage*.xml 2>/dev/null || true
    echo_success "Test artifacts cleaned"
}

clean_old() {
    echo_info "Cleaning obsolete directories..."
    find . -type d -name '.old*' -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name 'legacy' -exec rm -rf {} + 2>/dev/null || true
    echo_success "Obsolete directories cleaned"
}

clean_ruff() {
    echo_info "Cleaning ruff cache..."
    rm -rf .ruff_cache/ 2>/dev/null || true
    echo_success "Ruff cache cleaned"
}

########################################
# Main
########################################

usage() {
    echo "Usage: $0 [target...]"
    echo ""
    echo "Targets:"
    echo "  all     - Clean everything (default)"
    echo "  build   - Clean build artifacts (dist/, build/, *.egg*)"
    echo "  pyc     - Clean Python cache (__pycache__, *.pyc)"
    echo "  test    - Clean test artifacts (.pytest_cache, coverage)"
    echo "  old     - Clean obsolete directories (.old*, legacy)"
    echo "  ruff    - Clean ruff cache"
    echo ""
    echo "Examples:"
    echo "  $0              # Clean all"
    echo "  $0 build pyc    # Clean build and pyc only"
}

main() {
    local targets=("$@")

    # Default to all if no targets specified
    if [ ${#targets[@]} -eq 0 ]; then
        targets=("all")
    fi

    echo_header "Clean"

    for target in "${targets[@]}"; do
        case "$target" in
            all)
                clean_build
                clean_pyc
                clean_test
                clean_old
                clean_ruff
                ;;
            build)
                clean_build
                ;;
            pyc)
                clean_pyc
                ;;
            test)
                clean_test
                ;;
            old)
                clean_old
                ;;
            ruff)
                clean_ruff
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                echo_error "Unknown target: $target"
                usage
                exit 1
                ;;
        esac
    done

    echo ""
    echo_success "Clean completed"
}

main "$@"

# EOF
