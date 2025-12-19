#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-19 12:53:52 (ywatanabe)"
# File: ./examples/fig/run_demos.sh

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

run_demos() {
    for f in "$THIS_DIR"/??_*.py; do
        python $f
    done
}

collect_png_files() {
    TGT_DIR="$THIS_DIR/pngs"
    mkdir -p "$TGT_DIR"

    for ff in $(find $THIS_DIR -type f -name "*.png"); do
        ls -al $ff
        fnorm="${ff//\//-}"
        cp -v $ff $TGT_DIR/$fnorm
    done
}

main() {
    run_demos
    collect_png_files
}

main | tee -a $LOG_PATH

# EOF