#!/bin/bash
# Timestamp: "2026-01-05 (ywatanabe)"
# File: ./scripts/maintenance/inhibit_project_root_pollution.sh
# Pre-commit hook to prevent project root directory pollution

set -e

# Ensure we're in a git repository
cd "$(git rev-parse --show-toplevel)" || exit 1

# Hardcoded whitelist - tracked files
WHITELIST_TRACKED=(
    "CHANGELOG.md"
    "CLAUDE.md"
    "config"
    "containers"
    "data"
    "docs"
    "examples"
    "GITIGNORED"
    "LICENSE"
    "Makefile"
    "pyproject.toml"
    "README.md"
    "scitex-python"
    "scripts"
    "src"
    "tests"
    ".git"
    ".github"
    ".gitignore"
    ".pre-commit-config.yaml"
)

# Hardcoded whitelist - gitignored files (allowed but not tracked)
WHITELIST_GITIGNORED=(
    ".claude"
    ".dev"
    ".env"
    ".venv"
    ".playwright-mcp"
)

# Combine all whitelisted items
WHITELIST=("${WHITELIST_TRACKED[@]}" "${WHITELIST_GITIGNORED[@]}")

# Get staged files in root directory only (no subdirectory paths)
STAGED_ROOT_FILES=$(git diff --cached --name-only | grep -v '/' || true)

# Check each staged file
VIOLATIONS=""
for file in $STAGED_ROOT_FILES; do
    # Skip if in whitelist
    is_allowed=false
    for allowed in "${WHITELIST[@]}"; do
        if [[ "$file" == "$allowed" ]]; then
            is_allowed=true
            break
        fi
    done

    if [[ "$is_allowed" == false ]]; then
        VIOLATIONS="$VIOLATIONS\n  - $file"
    fi
done

if [[ -n "$VIOLATIONS" ]]; then
    echo "=========================================="
    echo "ERROR: Project root directory pollution!"
    echo "=========================================="
    echo ""
    echo "The following files are NOT allowed in project root:"
    echo -e "$VIOLATIONS"
    echo ""
    echo "Allowed files in root:"
    echo "  Tracked: ${WHITELIST_TRACKED[*]}"
    echo "  Gitignored: ${WHITELIST_GITIGNORED[*]}"
    echo ""
    echo "Please relocate to appropriate directories:"
    echo "  - Logs/output   -> ./logs/ or ./output/"
    echo "  - Temporary     -> ./.tmp/"
    echo "  - Working notes -> ./.claude/"
    echo "  - Data files    -> ./data/"
    echo "  - Config files  -> ./config/"
    echo ""
    exit 1
fi

exit 0
