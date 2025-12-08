#!/bin/bash
# ============================================
# SciTeX Release Script
# Builds and publishes both scitex and scitex-python packages
# ============================================
# Usage: ./scripts/release.sh [sync|build|upload-test|upload|release]
# ============================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REDIRECT_DIR="$PROJECT_ROOT/redirect"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Get current version from main package
get_version() {
    grep -oP '__version__ = "\K[^"]+' "$PROJECT_ROOT/src/scitex/__version__.py"
}

# Sync redirect package version
cmd_sync() {
    local version=$(get_version)
    log_info "Syncing redirect package version to $version"
    sed -i "s/^version = \".*\"/version = \"$version\"/" "$REDIRECT_DIR/pyproject.toml"
    sed -i "s/\"scitex>=.*\"/\"scitex>=$version\"/" "$REDIRECT_DIR/pyproject.toml"
    log_success "Redirect package version synced to $version"
}

# Build both packages
cmd_build() {
    local version=$(get_version)
    log_info "Building packages (version: $version)"

    # Clean redirect builds
    rm -rf "$REDIRECT_DIR/dist" "$REDIRECT_DIR/build" "$REDIRECT_DIR"/*.egg-info

    # Sync version first
    cmd_sync

    # Build main package (assumes Makefile already cleaned and built main)
    log_info "Building main package (scitex)..."
    cd "$PROJECT_ROOT"
    python -m build
    log_success "Main package built"

    # Build redirect package
    log_info "Building redirect package (scitex-python)..."
    cd "$REDIRECT_DIR"
    python -m build
    log_success "Redirect package built"

    echo ""
    log_success "Both packages built!"
    echo "  Main:     $PROJECT_ROOT/dist/"
    echo "  Redirect: $REDIRECT_DIR/dist/"
}

# Upload to TestPyPI
cmd_upload_test() {
    local version=$(get_version)
    log_info "Uploading version $version to TestPyPI..."

    cd "$PROJECT_ROOT"
    python -m twine upload --repository testpypi dist/*

    cd "$REDIRECT_DIR"
    python -m twine upload --repository testpypi dist/*

    echo ""
    log_success "Uploaded to TestPyPI!"
    echo ""
    echo "Test with:"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ scitex==$version"
    echo "  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ scitex-python==$version"
}

# Upload to PyPI
cmd_upload() {
    local version=$(get_version)
    log_info "Uploading version $version to PyPI..."

    cd "$PROJECT_ROOT"
    python -m twine upload dist/*

    cd "$REDIRECT_DIR"
    python -m twine upload dist/*

    echo ""
    log_success "Released to PyPI!"
    echo ""
    echo "Users can now install with:"
    echo "  pip install scitex"
    echo "  pip install scitex-python"
}

# Full release (build + upload)
cmd_release() {
    local version=$(get_version)
    log_warn "About to release version $version to PyPI (PRODUCTION)"
    read -p "Continue? [y/N] " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_warn "Cancelled"
        exit 0
    fi

    cmd_build
    cmd_upload
}

# Show usage
usage() {
    echo "SciTeX Release Script"
    echo "====================="
    echo "Manages both 'scitex' and 'scitex-python' packages"
    echo ""
    echo "Usage: $0 COMMAND"
    echo ""
    echo "Commands:"
    echo "  sync         - Sync redirect package version with main"
    echo "  build        - Build both packages"
    echo "  upload-test  - Upload both to TestPyPI"
    echo "  upload       - Upload both to PyPI"
    echo "  release      - Full release (build + upload to PyPI)"
    echo ""
    echo "Current version: $(get_version)"
}

# Main
case "${1:-}" in
    sync)        cmd_sync ;;
    build)       cmd_build ;;
    upload-test) cmd_upload_test ;;
    upload)      cmd_upload ;;
    release)     cmd_release ;;
    -h|--help|"") usage ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
