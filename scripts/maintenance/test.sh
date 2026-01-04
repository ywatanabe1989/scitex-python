#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09"
# File: ./scripts/maintenance/test.sh

set -e
set -o pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(realpath "$THIS_DIR"/../..)"

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
    echo "Usage: $0 [options] [module]"
    echo ""
    echo "Run pytest for the entire test suite or a specific module."
    echo ""
    echo "Options:"
    echo "  -c, --cov          Enable coverage reporting"
    echo "  -v, --verbose      Verbose output (default)"
    echo "  -q, --quiet        Quiet output"
    echo "  -f, --fast         Run only fast tests (exclude slow markers)"
    echo "  -x, --exitfirst    Stop on first failure"
    echo "  -s, --sequential   Disable parallel testing (run sequentially)"
    echo "  -n, --parallel N   Number of parallel workers (default: auto)"
    echo "  -k PATTERN         Only run tests matching PATTERN"
    echo "  -h, --help         Show this help"
    echo ""
    echo "Cache options (speeds up re-runs):"
    echo "  --lf, --last-failed   Re-run only tests that failed last time"
    echo "  --ff, --failed-first  Run failed tests first, then rest"
    echo "  --nf, --new-first     Run new tests first (modified files)"
    echo "  --sw, --stepwise      Stop on failure, resume from last fail"
    echo "  --cache-clear         Clear cache before running"
    echo ""
    echo "Incremental testing (only affected tests):"
    echo "  --testmon, --incremental  Run only tests affected by code changes"
    echo "  --testmon-nocollect       Skip collection, run from testmon db"
    echo ""
    echo "Root directory protection:"
    echo "  --strict-root             Fail if tests create files in project root"
    echo ""
    echo "Smart test selection:"
    echo "  --changed                 Run tests for git-changed source files"
    echo "  -m, --marker MARKER       Run tests with specific marker (unit, slow, etc)"
    echo ""
    echo "Module (optional):"
    echo "  Specify a module name to test only that module."
    echo "  Examples: stats, logging, config, plt, io, torch"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run all tests"
    echo "  $0 config              # Run tests for config module"
    echo "  $0 stats -c            # Run stats tests with coverage"
    echo "  $0 -k 'test_resolve'   # Run tests matching 'test_resolve'"
    echo "  $0 plt.ax              # Run tests for plt/ax submodule"
    echo ""
    echo "Coverage reports are saved to:"
    echo "  - HTML: htmlcov/<module>/index.html"
    echo "  - JSON: coverage-<module>.json"
    echo "  - XML:  coverage-<module>.xml"
}

########################################
# Main
########################################

main() {
    local module=""
    local coverage=false
    local verbose="-v"
    local fast=""
    local exitfirst=""
    local pattern=""
    local parallel="auto" # Default to parallel with auto workers
    local cache_opts=""   # Cache-related options
    local testmon=""      # Incremental testing with testmon
    local extra_args=()

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
        -c | --cov)
            coverage=true
            shift
            ;;
        -v | --verbose)
            verbose="-v"
            shift
            ;;
        -q | --quiet)
            verbose="-q"
            shift
            ;;
        -f | --fast)
            fast='-m "not slow"'
            shift
            ;;
        -x | --exitfirst)
            exitfirst="-x"
            shift
            ;;
        -s | --sequential)
            parallel=""
            shift
            ;;
        -n | --parallel)
            parallel="$2"
            shift 2
            ;;
        --lf | --last-failed)
            cache_opts="--lf"
            shift
            ;;
        --ff | --failed-first)
            cache_opts="--ff"
            shift
            ;;
        --nf | --new-first)
            cache_opts="--nf"
            shift
            ;;
        --sw | --stepwise)
            cache_opts="--sw"
            shift
            ;;
        --cache-clear)
            cache_opts="--cache-clear"
            shift
            ;;
        --testmon | --incremental)
            testmon="--testmon --testmon-datafile=tests/.testmondata"
            parallel="" # testmon doesn't work well with xdist
            shift
            ;;
        --testmon-nocollect)
            testmon="--testmon-nocollect --testmon-datafile=tests/.testmondata"
            parallel=""
            shift
            ;;
        --strict-root)
            extra_args+=("--strict-root")
            shift
            ;;
        --changed)
            # Run tests for git-changed files
            while IFS= read -r f; do
                test_path="tests/scitex/${f#src/scitex/}"
                test_dir="${test_path%/*}"
                [ -d "$test_dir" ] && extra_args+=("$test_dir")
            done < <(git diff --name-only HEAD -- 'src/scitex/*.py' 2>/dev/null)
            shift
            ;;
        -m | --marker)
            extra_args+=("-m" "$2")
            shift 2
            ;;
        -k)
            pattern="-k $2"
            shift 2
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        -*)
            # Pass through other pytest arguments
            extra_args+=("$1")
            shift
            ;;
        *)
            # Module name (first positional argument)
            if [ -z "$module" ]; then
                module="$1"
            else
                extra_args+=("$1")
            fi
            shift
            ;;
        esac
    done

    # Convert module dots to slashes for path (e.g., plt.ax -> plt/ax)
    local module_path="${module//.//}"

    # Determine test path and source path
    local test_path="tests/"
    local src_path="src/scitex"
    local report_name="all"

    if [ -n "$module" ]; then
        test_path="tests/scitex/${module_path}/"
        src_path="src/scitex/${module_path}"
        report_name="$module"

        # Validate paths exist
        if [ ! -d "$test_path" ]; then
            echo_error "Test directory not found: $test_path"
            echo ""
            echo_info "Available modules:"
            find tests/scitex/ -maxdepth 1 -type d ! -name '__*' -printf '%f\n' 2>/dev/null | head -20
            exit 1
        fi
    fi

    echo_header "Running Tests"
    echo_info "Test path: $test_path"
    [ -n "$module" ] && echo_info "Module:    $module"
    [ -n "$parallel" ] && echo_info "Parallel:  $parallel workers"
    [ -z "$parallel" ] && echo_info "Parallel:  disabled (sequential)"
    echo ""

    # Build pytest command
    local cmd="pytest $test_path $verbose --continue-on-collection-errors"
    [ -n "$parallel" ] && cmd="$cmd -n $parallel"
    [ -n "$fast" ] && cmd="$cmd $fast"
    [ -n "$exitfirst" ] && cmd="$cmd $exitfirst"
    [ -n "$pattern" ] && cmd="$cmd $pattern"
    [ -n "$cache_opts" ] && cmd="$cmd $cache_opts"
    [ -n "$testmon" ] && cmd="$cmd $testmon"
    [ ${#extra_args[@]} -gt 0 ] && cmd="$cmd ${extra_args[*]}"

    # Create results directory with timestamp
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local results_dir="tests/results"
    mkdir -p "$results_dir"

    # Always save test results as JSON
    cmd="$cmd --json-report --json-report-file=${results_dir}/test-${report_name}-${timestamp}.json"

    # Add coverage if requested
    if [ "$coverage" = true ]; then
        mkdir -p tests/htmlcov
        cmd="$cmd --cov=$src_path"
        cmd="$cmd --cov-report=term-missing"
        cmd="$cmd --cov-report=html:tests/htmlcov/${report_name}"
        cmd="$cmd --cov-report=json:${results_dir}/coverage-${report_name}-${timestamp}.json"
        cmd="$cmd --cov-report=xml:${results_dir}/coverage-${report_name}-${timestamp}.xml"

        # Also save latest (for badges) - in tests/results/
        cmd="$cmd --cov-report=json:${results_dir}/coverage-${report_name}.json"
    fi

    echo_info "Command: $cmd"
    echo ""

    # Save command output to log file
    local log_file="${results_dir}/test-${report_name}-${timestamp}.log"

    # Run tests
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo ""
        echo_success "Tests passed"

        echo ""
        echo_info "Test results saved:"
        echo_info "  JSON: ${results_dir}/test-${report_name}-${timestamp}.json"

        if [ "$coverage" = true ]; then
            echo ""
            echo_info "Coverage reports:"
            echo_info "  HTML:   tests/htmlcov/${report_name}/index.html"
            echo_info "  JSON:   ${results_dir}/coverage-${report_name}-${timestamp}.json"
            echo_info "  Latest: ${results_dir}/coverage-${report_name}.json (for badges)"
        fi

        echo ""
        echo_info "Full log: $log_file"
    else
        echo ""
        echo_error "Tests failed"
        echo ""
        echo_info "Full log: $log_file"
        exit 1
    fi
}

main "$@"

# EOF
