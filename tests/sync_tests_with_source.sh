#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-03 17:10:43 (ywatanabe)"
# File: ./tests/sync_tests_with_source.sh

# =============================================================================
# Test Synchronization Script
# =============================================================================
#
# PURPOSE:
#   Synchronizes test file structure with source code structure, ensuring
#   every source file has a corresponding test file with embedded source
#   code for reference.
#
# BEHAVIOR:
#   1. Mirrors src/scitex/ directory structure to tests/scitex/
#   2. For each source file (e.g., src/scitex/foo/bar.py):
#      - Creates/updates tests/scitex/foo/test_bar.py
#      - Preserves existing test code (before source block)
#      - Updates commented source code block at file end
#   3. Identifies "stale" tests (tests without matching source files)
#   4. With -m flag: moves stale tests to .old-{timestamp}/ directories
#
# STRUCTURE OF GENERATED TEST FILES:
#   - User's test code (preserved across syncs)
#   - pytest __main__ guard
#   - Commented source code block (auto-updated)
#
# USAGE:
#   ./sync_tests_with_source.sh          # Dry run - report stale files
#   ./sync_tests_with_source.sh -m       # Move stale files to .old/
#   ./sync_tests_with_source.sh -j 16    # Use 16 parallel jobs
#
# =============================================================================

ORIG_DIR="$(pwd)"
THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
echo > "$LOG_PATH"

GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

# Color scheme
GRAY='\033[0;90m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo_info() { echo -e "${GRAY}INFO: $1${NC}"; }
echo_success() { echo -e "${GREEN}SUCC: $1${NC}"; }
echo_warning() { echo -e "${YELLOW}WARN: $1${NC}"; }
echo_error() { echo -e "${RED}ERRO: $1${NC}"; }
echo_header() { echo_info "=== $1 ==="; }

THIS_DIR="./tests"
ROOT_DIR="$(realpath $THIS_DIR/..)"
cmd="cd $ROOT_DIR" && echo "$cmd" && eval "$cmd"

########################################
# Usage & Argument Parser
########################################
# Default Values
DO_MOVE=false
SRC_DIR="$(realpath "${THIS_DIR}/../src/scitex")"
TESTS_DIR="$(realpath "${THIS_DIR}/../tests/scitex")"
# Use half of available CPU cores by default (minimum 1)
CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
PARALLEL_JOBS=$(( CPU_COUNT / 2 > 0 ? CPU_COUNT / 2 : 1 ))

usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Synchronizes test files with source files, maintaining test code while updating source references."
    echo
    echo "Options:"
    echo "  -m, --move         Move stale test files to .old directory instead of just reporting (default: $DO_MOVE)"
    echo "  -s, --source DIR   Specify custom source directory (default: $SRC_DIR)"
    echo "  -t, --tests DIR    Specify custom tests directory (default: $TESTS_DIR)"
    echo "  -j, --jobs N       Number of parallel jobs (default: $PARALLEL_JOBS)"
    echo "  -h, --help         Display this help message"
    echo
    echo "Example:"
    echo "  $0 --move"
    echo "  $0 --source /path/to/src --tests /path/to/tests"
    echo "  $0 -j \$((CPU_COUNT))  # Use all CPU cores"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--move)
            DO_MOVE=true
            shift
            ;;
        -s|--source)
            SRC_DIR="$2"
            shift 2
            ;;
        -t|--tests)
            TESTS_DIR="$2"
            shift 2
            ;;
        -j|--jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Set default directories if not specified
if [ -z "$SRC_DIR" ]; then
    cmd="cd $ROOT_DIR" && echo "$cmd" && eval "$cmd"
fi

########################################
# Test Structure
########################################
prepare_tests_structure_as_source() {
    [ ! -d "$SRC_DIR" ] && echo "Source directory not found: $SRC_DIR" && exit 1
    construct_blacklist_patterns
    find "$SRC_DIR" -type d "${FIND_EXCLUDES[@]}" | while read -r dir; do
        tests_dir="${dir/$SRC_DIR/$TESTS_DIR}"
        mkdir -p "$tests_dir"
    done
}

########################################
# Source as Comment
########################################
get_source_code_block() {
    local src_file=$1
    echo ""
    echo "# --------------------------------------------------------------------------------"
    echo "# Start of Source Code from: $src_file"
    echo "# --------------------------------------------------------------------------------"
    sed 's/^/# /' "$src_file"
    echo ""
    echo "# --------------------------------------------------------------------------------"
    echo "# End of Source Code from: $src_file"
    echo "# --------------------------------------------------------------------------------"
}

extract_test_code() {
    local test_file=$1
    local temp_file=$(mktemp)

    # Check if file has source code block
    if grep -q "# Start of Source Code from:" "$test_file"; then
        # Extract content before the source comment block and before any pytest guard
        sed -n '/# Start of Source Code from:/q;/if __name__ == "__main__":/q;p' "$test_file" > "$temp_file"
    else
        # File doesn't have source block, copy everything before pytest guard if any
        sed -n '/if __name__ == "__main__":/q;p' "$test_file" > "$temp_file"
    fi

    # Return content if any (trimming trailing blank lines)
    if [ -s "$temp_file" ]; then
        # Remove trailing blank lines
        sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$temp_file"
        cat "$temp_file"
    fi
    rm "$temp_file"
}

get_pytest_guard() {
    echo ''
    echo 'if __name__ == "__main__":'
    echo '    import os'
    echo ''
    echo '    import pytest'
    echo ''
    echo '    pytest.main([os.path.abspath(__file__)])'
}

update_test_file() {
    local test_file=$1
    local src_file=$2

    if [ ! -f "$test_file" ]; then
        # If file doesn't exist, create it with minimal structure
        echo "$test_file not found. Creating..."
        mkdir -p "$(dirname "$test_file")"

        # Create with default structure: test placeholder -> pytest guard -> source code
        cat > "$test_file" << EOL
# Add your tests here

$(get_pytest_guard)
EOL
        # Add source code block
        get_source_code_block "$src_file" >> "$test_file"
    else
        # File exists, preserve test code
        local temp_file=$(mktemp)
        local test_code=$(extract_test_code "$test_file")

        # Create new file: test code -> pytest guard -> source code
        if [ -n "$test_code" ]; then
            echo "$test_code" > "$temp_file"
            # Add a blank line if test_code doesn't end with one
            [[ "$(tail -c 1 "$temp_file")" != "" ]] && echo "" >> "$temp_file"
        else
            # Add default comment if no test code
            echo "# Add your tests here" > "$temp_file"
            echo "" >> "$temp_file"
        fi

        # Add standard pytest guard
        get_pytest_guard >> "$temp_file"

        # Add source code block
        get_source_code_block "$src_file" >> "$temp_file"

        # Replace original file
        mv "$temp_file" "$test_file"
    fi
}

########################################
# Finder
########################################
construct_blacklist_patterns() {
    local EXCLUDE_PATHS=(
        "*/.*"
        "*/.*/*"
        "*/deprecated*"
        "*/archive*"
        "*/backup*"
        "*/tmp*"
        "*/temp*"
        "*/RUNNING/*"
        "*/FINISHED/*"
        "*/FINISHED_SUCCESS/*"
        "*/2025Y*"
        "*/2024Y*"
        "*/__pycache__/*"
    )

    FIND_EXCLUDES=()
    PRUNE_ARGS=()
    for path in "${EXCLUDE_PATHS[@]}"; do
        FIND_EXCLUDES+=( -not -path "$path" )
        PRUNE_ARGS+=( -path "$path" -o )
    done
    unset 'PRUNE_ARGS[${#PRUNE_ARGS[@]}-1]'
}

find_files() {
    local search_path=$1
    local type=$2
    local name_pattern=$3

    construct_blacklist_patterns
    find "$search_path" \
        \( "${PRUNE_ARGS[@]}" \) -prune -o -type "$type" -name "$name_pattern" -print
}

########################################
# Clean-upper
########################################
move_stale_test_files_to_old() {
    local timestamp="$(date +%Y%m%d_%H%M%S)"
    local stale_count=0
    local moved_count=0
    local stale_files=()

    # Collect stale files first
    while IFS= read -r test_path; do
        # Skip files in ./tests/custom
        [[ "$test_path" =~ ^${TESTS_DIR}/custom ]] && continue

        # Determine corresponding source file
        test_rel_path="${test_path#$TESTS_DIR/}"
        test_rel_dir="$(dirname $test_rel_path)"
        test_filename="$(basename $test_rel_path)"

        src_filename="${test_filename#test_}"
        src_rel_dir="$test_rel_dir"
        src_rel_path="$src_rel_dir/$src_filename"
        src_path="$SRC_DIR/$src_rel_path"

        if [ ! -f "$src_path" ] && [ -f "$test_path" ]; then
            stale_files+=("$test_path")
            ((stale_count++))
        fi
    done < <(find "$TESTS_DIR" -name "test_*.py" -not -path "*.old*")

    # Report stale files
    if [ $stale_count -gt 0 ]; then
        echo ""
        echo_header "Stale Test Files ($stale_count found)"
        echo ""
        for stale_path in "${stale_files[@]}"; do
            local rel_path="${stale_path#$TESTS_DIR/}"
            if [ "$DO_MOVE" = "true" ]; then
                stale_filename="$(basename $stale_path)"
                stale_path_dir="$(dirname $stale_path)"
                old_dir_with_timestamp="$stale_path_dir/.old-$timestamp"
                tgt_path="$old_dir_with_timestamp/$stale_filename"

                mkdir -p "$old_dir_with_timestamp"
                mv "$stale_path" "$tgt_path"
                echo_success "  [MOVED] $rel_path"
                ((moved_count++))
            else
                echo_warning "  [STALE] $rel_path"
            fi
        done
        echo ""
        if [ "$DO_MOVE" = "false" ]; then
            echo_info "To move stale files, run: $0 -m"
        else
            echo_success "Moved $moved_count stale test files"
        fi
        echo ""
    fi
}

remove_hidden_test_files_and_dirs() {
    find "$TESTS_DIR" -type f -name ".*" -delete 2>/dev/null
    find "$TESTS_DIR" -type d -name ".*" -not -path "$TESTS_DIR/.old" -not -path "$TESTS_DIR/.old/*" -exec rm -rf {} \; 2>/dev/null
}

cleanup_unnecessary_test_files() {
    find "$TESTS_DIR" -type d -name "*RUNNING*" -exec rm -rf {} \; 2>/dev/null
    find "$TESTS_DIR" -type d -name "*FINISHED*" -exec rm -rf {} \; 2>/dev/null
    find "$TESTS_DIR" -type d -name "*FINISHED_SUCCESS*" -exec rm -rf {} \; 2>/dev/null
    find "$TESTS_DIR" -type d -name "*2024Y*" -exec rm -rf {} \; 2>/dev/null
    find "$TESTS_DIR" -type d -name "*2025Y*" -exec rm -rf {} \; 2>/dev/null
    find "$TESTS_DIR" -type d -name "*.py_out" -exec rm -rf {} \; 2>/dev/null
    find "$TESTS_DIR" -type d -name "*__pycache__*" -exec rm -rf {} \; 2>/dev/null
    find "$TESTS_DIR" -type d -name "*.pyc" -exec rm -rf {} \; 2>/dev/null
}

########################################
# Permission
########################################
chmod_python_source_scripts_as_executable() {
    construct_blacklist_patterns
    find "$SRC_DIR" -type f -name "*.py" "${FIND_EXCLUDES[@]}" -exec chmod +x {} \;
}

########################################
# Parallel Processing Helper
########################################
# Process a single source file (called in parallel)
process_single_file() {
    local src_file="$1"
    local SRC_DIR="$2"
    local TESTS_DIR="$3"

    # Skip __init__.py files (they cause import conflicts in tests)
    [[ "$(basename "$src_file")" == "__init__.py" ]] && return

    # derive relative path and parts
    rel="${src_file#$SRC_DIR/}"
    rel_dir=$(dirname "$rel")
    src_base=$(basename "$rel")

    # ensure test subdir exists
    tests_dir="$TESTS_DIR/$rel_dir"
    mkdir -p "$tests_dir"

    # build correct test file path
    test_file="$tests_dir/test_$src_base"

    # Process the file (inline the update logic to avoid export complexity)
    if [ ! -f "$test_file" ]; then
        # If file doesn't exist, create it with minimal structure
        mkdir -p "$(dirname "$test_file")"

        cat > "$test_file" << EOL
# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
EOL
        # Add source code block
        {
            echo ""
            echo "# --------------------------------------------------------------------------------"
            echo "# Start of Source Code from: $src_file"
            echo "# --------------------------------------------------------------------------------"
            sed 's/^/# /' "$src_file"
            echo ""
            echo "# --------------------------------------------------------------------------------"
            echo "# End of Source Code from: $src_file"
            echo "# --------------------------------------------------------------------------------"
        } >> "$test_file"
    else
        # File exists, preserve test code
        local temp_file=$(mktemp)

        # Extract test code (content before source block or pytest guard)
        local test_code=""
        if grep -q "# Start of Source Code from:" "$test_file"; then
            test_code=$(sed -n '/# Start of Source Code from:/q;/if __name__ == "__main__":/q;p' "$test_file")
        else
            test_code=$(sed -n '/if __name__ == "__main__":/q;p' "$test_file")
        fi

        # Write test code or default
        if [ -n "$test_code" ]; then
            echo "$test_code" > "$temp_file"
            [[ "$(tail -c 1 "$temp_file")" != "" ]] && echo "" >> "$temp_file"
        else
            echo "# Add your tests here" > "$temp_file"
            echo "" >> "$temp_file"
        fi

        # Add pytest guard
        cat >> "$temp_file" << 'EOL'

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
EOL

        # Add source code block
        {
            echo ""
            echo "# --------------------------------------------------------------------------------"
            echo "# Start of Source Code from: $src_file"
            echo "# --------------------------------------------------------------------------------"
            sed 's/^/# /' "$src_file"
            echo ""
            echo "# --------------------------------------------------------------------------------"
            echo "# End of Source Code from: $src_file"
            echo "# --------------------------------------------------------------------------------"
        } >> "$temp_file"

        mv "$temp_file" "$test_file"
    fi
}
export -f process_single_file

########################################
# Main
########################################
main() {
    local do_move=${1:-false}
    local start_time=$(date +%s)

    echo ""
    echo_header "Test Synchronization"
    echo ""
    echo_info "Source:    $SRC_DIR"
    echo_info "Tests:     $TESTS_DIR"
    echo_info "Jobs:      $PARALLEL_JOBS"
    echo ""

    echo_info "Preparing test structure..."
    remove_hidden_test_files_and_dirs
    prepare_tests_structure_as_source
    chmod_python_source_scripts_as_executable
    cleanup_unnecessary_test_files

    echo_info "Synchronizing test files (parallel)..."
    local file_count=$(find_files "$SRC_DIR" f "*.py" | wc -l)
    find_files "$SRC_DIR" f "*.py" | \
        xargs -P "$PARALLEL_JOBS" -I {} bash -c 'process_single_file "$@"' _ {} "$SRC_DIR" "$TESTS_DIR"
    echo_success "Processed $file_count source files"

    remove_hidden_test_files_and_dirs
    move_stale_test_files_to_old

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    echo_header "Summary"
    echo_success "Completed in ${elapsed}s"
    echo ""

    tree "$TESTS_DIR" 2>&1 >> "$LOG_PATH"
}

main "$@"
cd $ORIG_DIR

# EOF