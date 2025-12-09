#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 20:28:11 (ywatanabe)"
# File: ./examples/plt/run_demo_plots.sh

ORIG_DIR="$(pwd)"
THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
echo > "$LOG_PATH"

GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"

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
# ---------------------------------------

echo $THIS_DIR

main() {
    rm -rf $THIS_DIR/demo_matplotlib_basic_out
    rm -rf $THIS_DIR/demo_scitex_wrappers_out
    rm -rf $THIS_DIR/demo_seaborn_wrappers_out

    python $THIS_DIR/demo_matplotlib_basic.py
    python $THIS_DIR/demo_scitex_wrappers.py
    python $THIS_DIR/demo_seaborn_wrappers.py

    ls $THIS_DIR/demo_matplotlib_basic_out
    ls $THIS_DIR/demo_scitex_wrappers_out
    ls $THIS_DIR/demo_seaborn_wrappers_out

}

main "$@" | tee -a $LOG_PATH 2>&1

# EOF