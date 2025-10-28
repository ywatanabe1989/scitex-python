#!/bin/bash
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/containers/build_containers.sh

################################################################################
# Build SciTeX Singularity Containers
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../build/containers"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if singularity is installed
if ! command -v singularity &> /dev/null; then
    print_error "Singularity is not installed"
    print_info "Install with: sudo apt-get install singularity-container"
    print_info "Or see: https://sylabs.io/guides/latest/user-guide/quick_start.html"
    exit 1
fi

# Check if running as root (required for building)
if [ "$EUID" -ne 0 ]; then
    print_error "This script must be run as root (use sudo)"
    print_info "Usage: sudo ./build_containers.sh [container_name]"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Available containers
CONTAINERS=("core" "writer" "scholar" "ml")

# Function to build a container
build_container() {
    local name=$1
    local def_file="${SCRIPT_DIR}/scitex-${name}.def"
    local output_file="${OUTPUT_DIR}/scitex-${name}.sif"

    if [ ! -f "$def_file" ]; then
        print_error "Definition file not found: $def_file"
        return 1
    fi

    print_info "Building scitex-${name} container..."
    print_info "  Definition: $def_file"
    print_info "  Output: $output_file"

    # Remove old container if exists
    if [ -f "$output_file" ]; then
        print_warn "Removing old container: $output_file"
        rm -f "$output_file"
    fi

    # Build container
    if singularity build "$output_file" "$def_file"; then
        print_info "✓ Successfully built scitex-${name}.sif"

        # Show container size
        local size=$(du -h "$output_file" | cut -f1)
        print_info "  Size: $size"

        # Test container
        print_info "Testing container..."
        if singularity test "$output_file" 2>/dev/null; then
            print_info "✓ Container tests passed"
        else
            print_warn "Container tests failed or not defined"
        fi

        return 0
    else
        print_error "Failed to build scitex-${name}.sif"
        return 1
    fi
}

# Parse arguments
if [ $# -eq 0 ]; then
    # Build all containers
    print_info "Building all containers..."
    print_info "This will take ~30-60 minutes depending on your internet speed"
    print_info ""

    success_count=0
    fail_count=0

    for container in "${CONTAINERS[@]}"; do
        print_info "=========================================="
        if build_container "$container"; then
            ((success_count++))
        else
            ((fail_count++))
        fi
        print_info ""
    done

    print_info "=========================================="
    print_info "Build Summary:"
    print_info "  Successful: $success_count"
    if [ $fail_count -gt 0 ]; then
        print_error "  Failed: $fail_count"
    fi

    if [ $fail_count -eq 0 ]; then
        print_info "✓ All containers built successfully!"
        print_info ""
        print_info "Containers location: $OUTPUT_DIR"
        print_info ""
        print_info "Next steps:"
        print_info "  1. Test containers: singularity exec $OUTPUT_DIR/scitex-core.sif python --version"
        print_info "  2. Upload to GitHub Releases"
        print_info "  3. Update scitex Python code to download containers"
    else
        exit 1
    fi

else
    # Build specific container
    container_name=$1

    # Validate container name
    if [[ ! " ${CONTAINERS[@]} " =~ " ${container_name} " ]]; then
        print_error "Invalid container name: $container_name"
        print_info "Available containers: ${CONTAINERS[*]}"
        exit 1
    fi

    build_container "$container_name"
fi

# EOF
