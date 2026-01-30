#!/bin/bash
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-13 12:32:56 (ywatanabe)"
# File: ./src/scitex/scholar/utils/create_demo_movie.sh

ORIG_DIR="$(pwd)"
THIS_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"
LOG_PATH="$THIS_DIR/.$(basename $0).log"
echo > "$LOG_PATH"

BLACK='\033[0;30m'
LIGHT_GRAY='\033[0;37m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo_info() { echo -e "${LIGHT_GRAY}$1${NC}"; }
echo_success() { echo -e "${GREEN}$1${NC}"; }
echo_warning() { echo -e "${YELLOW}$1${NC}"; }
echo_error() { echo -e "${RED}$1${NC}"; }
# ---------------------------------------

# --------------------
# SciTeX Scholar Demo
# --------------------

# # Cleanup the library
# bandicam (screen recorder)
# watch -n 1 tree -l $HOME~/.scitex/scholar/library/seizure_prediction
# export SCITEX_LOGGING_LEVEL=warning

# Cleanup the library
rm -rf $HOME/.scitex/scholar/library/*

# Run the single paper pipeline
python -m scitex.scholar single \
    --doi "10.1038/s41582-018-0055-2" \
    --project seizure_prediction \
    --browser-mode interactive

# Run the bibtex pipeline
python -m scitex.scholar bibtex \
    --bibtex $HOME/proj/scitex_repo/src/scitex/scholar/data/bib_files/seizure_prediction.bib \
    --project seizure_prediction \
    --num-workers 8 \
    --browser-mode stealth

# EOF