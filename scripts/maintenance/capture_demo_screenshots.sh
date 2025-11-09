#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# Timestamp: 2025-11-08
# Author: Claude Code
# File: ./scripts/maintenance/capture_demo_screenshots.sh
#
# Capture demo screenshots in demo-<timestamp>/<02d>-<normalized-url>.jpg format
#
# Usage:
#   ./scripts/maintenance/capture_demo_screenshots.sh <url1> <url2> ...
#
# Examples:
#   ./scripts/maintenance/capture_demo_screenshots.sh http://localhost:8000 http://localhost:8000/about

set -euo pipefail

# Check if URLs provided
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <url1> <url2> ..."
    echo "Example: $0 http://localhost:8000 http://localhost:8000/about"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="demo-${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "Capturing screenshots to: $OUTPUT_DIR"
echo ""

# Function to normalize URL for filename
normalize_url() {
    local url="$1"
    # Remove protocol
    local normalized="${url#http://}"
    normalized="${normalized#https://}"
    # Replace special chars with dash
    normalized="${normalized//\//-}"
    normalized="${normalized//:/-}"
    normalized="${normalized//\?/-}"
    normalized="${normalized//&/-}"
    normalized="${normalized//=/-}"
    # Remove trailing dashes
    normalized="${normalized%%-}"
    echo "$normalized"
}

# Loop through URLs
COUNTER=1
for url in "$@"; do
    # Create numbered filename
    NUM=$(printf "%02d" "$COUNTER")
    NORMALIZED=$(normalize_url "$url")
    OUTPUT_FILE="${OUTPUT_DIR}/${NUM}-${NORMALIZED}.jpg"

    echo "[$COUNTER/$#] $url"
    echo "    -> ${NUM}-${NORMALIZED}.jpg"

    # Capture screenshot (convert PNG to JPG)
    TEMP_PNG=$(mktemp --suffix=.png)
    if scitex web take-screenshot "$url" --output "$(dirname "$TEMP_PNG")" --quality 85 > /dev/null 2>&1; then
        # Find the generated PNG and convert to JPG
        GENERATED_PNG=$(ls -t "$(dirname "$TEMP_PNG")"/screenshot_*.png 2>/dev/null | head -1)
        if [[ -n "$GENERATED_PNG" ]] && [[ -f "$GENERATED_PNG" ]]; then
            convert "$GENERATED_PNG" -quality 85 "$OUTPUT_FILE"
            rm "$GENERATED_PNG"
            echo "    ✓ Saved"
        else
            echo "    ✗ Failed to find generated screenshot"
        fi
    else
        echo "    ✗ Failed to capture"
    fi
    rm -f "$TEMP_PNG"

    echo ""
    ((COUNTER++))
done

echo "Done! Screenshots saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
