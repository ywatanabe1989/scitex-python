#!/usr/bin/env python3
"""Update all test-*.yml workflows to use the reusable _test-module.yml workflow."""

import re
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")

# Skip these workflows
SKIP_FILES = {
    "test-install.yml",
    "test-install-modules.yml",
    "_test-module.yml",
}

TEMPLATE = """name: {name} Module Tests

on:
  push:
    branches: [main, develop]
    paths:
      - 'src/scitex/{module}/**'
      - 'tests/scitex/{module}/**'
      - '.github/workflows/test-{module}.yml'
      - 'scripts/test-module.sh'
  pull_request:
    branches: [main, develop]
    paths:
      - 'src/scitex/{module}/**'
      - 'tests/scitex/{module}/**'

jobs:
  test:
    uses: ./.github/workflows/_test-module.yml
    with:
      module: {module}
      mode: editable
"""


def update_workflow(filepath: Path) -> bool:
    """Update a single workflow file. Returns True if modified."""
    if filepath.name in SKIP_FILES:
        return False

    match = re.match(r"test-(.+)\.yml", filepath.name)
    if not match:
        return False

    module = match.group(1)
    name = module.upper() if len(module) <= 3 else module.capitalize()

    new_content = TEMPLATE.format(module=module, name=name)

    old_content = filepath.read_text()
    if old_content.strip() == new_content.strip():
        return False

    filepath.write_text(new_content)
    print(f"  Updated: {filepath.name}")
    return True


def main():
    """Process all test workflow files."""
    print("Updating module workflows to use reusable workflow...\n")

    modified = 0
    for workflow in sorted(WORKFLOW_DIR.glob("test-*.yml")):
        if update_workflow(workflow):
            modified += 1

    print(f"\nUpdated {modified} workflow files")
    print("Review changes with: git diff .github/workflows/")


if __name__ == "__main__":
    main()
