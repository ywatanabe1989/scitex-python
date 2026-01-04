#!/bin/bash
# -*- coding: utf-8 -*-
# Time-stamp: "2026-01-04 (ywatanabe)"
# File: ./scripts/maintenance/generate_extras.sh

# ============================================
# Generate pyproject.toml extras from requirements
# ============================================

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ============================================
# Main
# ============================================

cd "$PROJECT_ROOT"

show_help() {
    echo -e ""
    echo -e "${CYAN}Generate pyproject.toml extras from config/requirements/*.txt${NC}"
    echo -e ""
    echo -e "${CYAN}Usage:${NC}"
    echo -e "  $0              Show what would be generated (dry-run)"
    echo -e "  $0 --update     Update pyproject.toml in place"
    echo -e "  $0 --check      Check if pyproject.toml is in sync"
    echo -e ""
}

case "${1:-}" in
--help | -h)
    show_help
    ;;
--update | -u)
    echo -e "${CYAN}Updating pyproject.toml extras...${NC}"
    python scripts/generate_extras.py --update
    echo -e "${GREEN}Done. pyproject.toml updated.${NC}"
    ;;
--check | -c)
    echo -e "${CYAN}Checking if extras are in sync...${NC}"
    if python scripts/generate_extras.py --check; then
        echo -e "${GREEN}pyproject.toml is in sync with requirements.${NC}"
    else
        echo -e "${RED}pyproject.toml is out of sync!${NC}"
        echo -e "${YELLOW}Hint: Run 'make generate-extras' to update${NC}"
        exit 1
    fi
    ;;
*)
    echo -e "${CYAN}Dry-run: Showing generated extras...${NC}"
    python scripts/generate_extras.py
    echo -e ""
    echo -e "${YELLOW}Hint: Use --update to apply changes to pyproject.toml${NC}"
    ;;
esac

# EOF
