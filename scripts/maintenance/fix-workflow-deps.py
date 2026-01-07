#!/usr/bin/env python3
"""Fix workflow files to use module-specific dependencies instead of [dev]."""

import re
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")

# Modules with optional extras in pyproject.toml
MODULES_WITH_EXTRAS = {
    "ai",
    "audio",
    "benchmark",
    "browser",
    "capture",
    "cli",
    "db",
    "decorators",
    "dsp",
    "fig",
    "fts",
    "gen",
    "git",
    "io",
    "linalg",
    "msword",
    "nn",
    "path",
    "plt",
    "repro",
    "resource",
    "scholar",
    "stats",
    "str",
    "torch",
    "types",
    "utils",
    "web",
    "writer",
}

# Skip these workflows
SKIP_FILES = {"test-install.yml", "test-install-modules.yml"}


def fix_workflow(filepath: Path) -> bool:
    """Fix a single workflow file. Returns True if modified."""
    if filepath.name in SKIP_FILES:
        return False

    # Extract module name from filename
    match = re.match(r"test-(.+)\.yml", filepath.name)
    if not match:
        return False

    module = match.group(1)
    content = filepath.read_text()
    original = content

    # Determine correct install command
    if module in MODULES_WITH_EXTRAS:
        new_install = f'pip install -e ".[{module}]"\n          pip install pytest pytest-cov pytest-timeout'
    else:
        new_install = (
            "pip install -e .\n          pip install pytest pytest-cov pytest-timeout"
        )

    # Replace various patterns
    patterns = [
        r'pip install -e "\.\[dev,ml\]"',
        r'pip install -e "\.\[dev\]"',
        r'pip install -e "\.\[ml,dev\]"',
    ]

    for pattern in patterns:
        content = re.sub(pattern, new_install, content)

    if content != original:
        filepath.write_text(content)
        extra_info = f".[{module}]" if module in MODULES_WITH_EXTRAS else "core only"
        print(f"  Fixed: {filepath.name} -> {extra_info}")
        return True
    return False


def main():
    """Process all test workflow files."""
    print("Fixing workflow dependencies...\n")

    modified = 0
    for workflow in sorted(WORKFLOW_DIR.glob("test-*.yml")):
        if fix_workflow(workflow):
            modified += 1

    print(f"\nModified {modified} workflow files")
    print("Review changes with: git diff .github/workflows/")


if __name__ == "__main__":
    main()
