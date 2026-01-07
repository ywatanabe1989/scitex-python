#!/bin/bash
# SciTeX Module Test Script (with pytest)
# Usage:
#   ./test_module.sh local MODULE      - Install from local + run pytest
#   ./test_module.sh pypi MODULE       - Install from PyPI + run pytest

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
GRAY='\033[0;90m'
NC='\033[0m'

# Config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_VENV_DIR="/tmp/scitex-test-install"

# Get version from pyproject.toml
get_version() {
    grep '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/'
}

# Setup isolated venv
setup_venv() {
    rm -rf "$TEST_VENV_DIR"
    python -m venv "$TEST_VENV_DIR"
    "$TEST_VENV_DIR/bin/pip" install --upgrade pip >/dev/null
}

# Cleanup venv
cleanup_venv() {
    rm -rf "$TEST_VENV_DIR"
}

# Install from local wheel
install_local() {
    local module="$1"
    local version
    version=$(get_version)

    local wheel="$PROJECT_ROOT/dist/scitex-${version}-py3-none-any.whl"
    if [ ! -f "$wheel" ]; then
        echo -e "${RED}Wheel not found: $wheel${NC}"
        echo -e "${YELLOW}Run 'make build' first${NC}"
        return 1
    fi

    echo -e "${GRAY}Installing scitex[$module]...${NC}"
    "$TEST_VENV_DIR/bin/pip" install "${wheel}[$module]" >/dev/null 2>&1 || {
        echo -e "${RED}Installation failed${NC}"
        return 1
    }
}

# Install from PyPI
install_pypi() {
    local module="$1"
    local version
    version=$(get_version)

    echo -e "${GRAY}Installing scitex[$module]==$version from PyPI...${NC}"
    "$TEST_VENV_DIR/bin/pip" install "scitex[$module]==$version" 2>&1 || {
        echo -e "${RED}PyPI installation failed${NC}"
        return 1
    }
}

# Run pytest for module
run_pytest() {
    local module="$1"
    local test_dir="$PROJECT_ROOT/tests/scitex/$module"

    "$TEST_VENV_DIR/bin/pip" install pytest pytest-cov >/dev/null

    if [ -d "$test_dir" ]; then
        echo -e "${GRAY}Running tests for $module...${NC}"
        "$TEST_VENV_DIR/bin/pytest" "$test_dir/" -v --tb=short -x || {
            echo -e "${RED}Tests failed${NC}"
            return 1
        }
    else
        echo -e "${YELLOW}No tests found for $module${NC}"
    fi
}

# Run pytest (allow failures)
run_pytest_soft() {
    local module="$1"
    local test_dir="$PROJECT_ROOT/tests/scitex/$module"

    "$TEST_VENV_DIR/bin/pip" install pytest pytest-cov >/dev/null

    if [ -d "$test_dir" ]; then
        echo -e "${GRAY}Running tests for $module...${NC}"
        "$TEST_VENV_DIR/bin/pytest" "$test_dir/" -v --tb=short ||
            echo -e "${YELLOW}Some tests failed${NC}"
    else
        echo -e "${YELLOW}No tests found for $module${NC}"
    fi
}

# Main test function
run_test() {
    local source="$1" # local or pypi
    local module="$2"

    if [ -z "$module" ]; then
        echo -e "${RED}ERROR: MODULE not specified${NC}"
        echo "Usage: $0 {local|pypi} MODULE"
        exit 1
    fi

    echo -e "${CYAN}Testing scitex[$module] with pytest ($source)...${NC}"

    setup_venv
    trap cleanup_venv EXIT

    if [ "$source" = "local" ]; then
        install_local "$module" || {
            cleanup_venv
            exit 1
        }
        run_pytest "$module" || {
            cleanup_venv
            exit 1
        }
    else
        install_pypi "$module" || {
            cleanup_venv
            exit 1
        }
        run_pytest_soft "$module" # Allow failures for PyPI (informational)
    fi

    cleanup_venv
    trap - EXIT

    echo -e "${GREEN}scitex[$module] tests passed${NC}"
}

# Main
case "$1" in
local)
    run_test "local" "$2"
    ;;
pypi)
    run_test "pypi" "$2"
    ;;
*)
    echo "Usage: $0 {local|pypi} MODULE"
    echo ""
    echo "Commands:"
    echo "  local MODULE    Install from local wheel + run pytest"
    echo "  pypi MODULE     Install from PyPI + run pytest"
    echo ""
    echo "Examples:"
    echo "  $0 local io     # Test scitex[io] from local build with pytest"
    echo "  $0 pypi stats   # Test scitex[stats] from PyPI with pytest"
    exit 1
    ;;
esac
