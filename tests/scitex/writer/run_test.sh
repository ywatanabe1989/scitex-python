#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 09:25:28 (ywatanabe)"
# File: ./tests/scitex/writer/run_test.sh

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

echo_info "Running $0..." | tee $LOG_PATH

cd $GIT_ROOT
pytest "$THIS_DIR" | tee $LOG_PATH

echo_info "See full log: $LOG_PATH" | tee $LOG_PATH

# EOF