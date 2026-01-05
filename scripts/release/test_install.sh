#!/bin/bash
# SciTeX Installation Test Script
# Usage:
#   ./test_install.sh local [MODULE]     - Test local wheel installation
#   ./test_install.sh pypi [MODULE]      - Test PyPI installation
#   ./test_install.sh local-all          - Test all key modules from local build
#   ./test_install.sh pypi-all           - Test all key modules from PyPI

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
KEY_MODULES="io plt stats nn ai dsp cli db writer"

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

# Test imports work
test_imports() {
    local module="$1"

    "$TEST_VENV_DIR/bin/python" -c "import scitex; print(f'Version: {scitex.__version__}')" || {
        echo -e "${RED}Import failed${NC}"
        return 1
    }

    if [ -n "$module" ] && [ "$module" != "all" ]; then
        "$TEST_VENV_DIR/bin/python" -c "from scitex import $module" 2>/dev/null ||
            echo -e "${YELLOW}Note: 'from scitex import $module' not available (meta-extra)${NC}"
    fi
}

# Install from local wheel
install_local() {
    local module="${1:-all}"
    local version
    version=$(get_version)

    local wheel="$PROJECT_ROOT/dist/scitex-${version}-py3-none-any.whl"
    if [ ! -f "$wheel" ]; then
        echo -e "${RED}Wheel not found: $wheel${NC}"
        echo -e "${YELLOW}Run 'make build' first${NC}"
        return 1
    fi

    echo -e "${GRAY}Installing scitex[$module] from local build...${NC}"
    "$TEST_VENV_DIR/bin/pip" install "${wheel}[$module]" >/dev/null 2>&1 || {
        echo -e "${RED}Installation failed${NC}"
        return 1
    }
}

# Install from PyPI
install_pypi() {
    local module="${1:-all}"
    local version
    version=$(get_version)

    echo -e "${GRAY}Installing scitex[$module]==$version from PyPI...${NC}"
    "$TEST_VENV_DIR/bin/pip" install "scitex[$module]==$version" >/dev/null 2>&1 || {
        echo -e "${RED}PyPI installation failed${NC}"
        return 1
    }
}

# Main test function
run_test() {
    local source="$1" # local or pypi
    local module="$2" # module name or empty for all

    [ -z "$module" ] && module="all"

    echo -e "${CYAN}Testing scitex[$module] installation ($source)...${NC}"

    setup_venv
    trap cleanup_venv EXIT

    if [ "$source" = "local" ]; then
        install_local "$module" || {
            cleanup_venv
            exit 1
        }
    else
        install_pypi "$module" || {
            cleanup_venv
            exit 1
        }
    fi

    echo -e "${GRAY}Testing imports...${NC}"
    test_imports "$module" || {
        cleanup_venv
        exit 1
    }

    cleanup_venv
    trap - EXIT

    echo -e "${GREEN}scitex[$module] installation test passed${NC}"
}

# Test all key modules
run_test_all() {
    local source="$1" # local or pypi

    echo -e "${CYAN}Testing all key module installations ($source)...${NC}"

    for mod in $KEY_MODULES; do
        echo -e "${GRAY}Testing scitex[$mod]...${NC}"
        run_test "$source" "$mod" || exit 1
    done

    echo -e "${GREEN}All module installation tests passed${NC}"
}

# Main
case "$1" in
local)
    run_test "local" "$2"
    ;;
pypi)
    run_test "pypi" "$2"
    ;;
local-all)
    run_test_all "local"
    ;;
pypi-all)
    run_test_all "pypi"
    ;;
*)
    echo "Usage: $0 {local|pypi|local-all|pypi-all} [MODULE]"
    echo ""
    echo "Commands:"
    echo "  local [MODULE]    Test local wheel installation"
    echo "  pypi [MODULE]     Test PyPI installation"
    echo "  local-all         Test all key modules from local build"
    echo "  pypi-all          Test all key modules from PyPI"
    echo ""
    echo "Examples:"
    echo "  $0 local          # Test scitex[all] from local build"
    echo "  $0 local io       # Test scitex[io] from local build"
    echo "  $0 pypi stats     # Test scitex[stats] from PyPI"
    exit 1
    ;;
esac
