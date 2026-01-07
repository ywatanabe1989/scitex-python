#!/usr/bin/env python3
"""Auto-fix missing dependencies in pyproject.toml based on import analysis.

Usage:
    ./scripts/maintenance/fix-module-deps.py --dry-run   # Show what would change
    ./scripts/maintenance/fix-module-deps.py --apply     # Apply changes
"""

import re
import sys

# Run detect-module-deps to get missing deps
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def load_detector():
    """Load the detect-module-deps module."""
    spec = spec_from_file_location("detect_deps", SCRIPT_DIR / "detect-module-deps.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_all_missing(detector) -> dict[str, list[str]]:
    """Get missing deps for all modules."""
    current_extras = detector.load_pyproject_extras(PROJECT_ROOT)
    modules = sorted(
        d.name
        for d in (PROJECT_ROOT / "src" / "scitex").iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )

    all_missing = {}
    for module in modules:
        result = detector.analyze_module(module, PROJECT_ROOT, current_extras)
        if result["missing"]:
            all_missing[module] = result["missing"]
    return all_missing


def update_pyproject(missing: dict[str, list[str]], dry_run: bool = True) -> None:
    """Update pyproject.toml with missing dependencies."""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    content = pyproject_path.read_text()

    changes = []

    for module, deps in sorted(missing.items()):
        # Find the module's extras section
        # Pattern: module = [\n    "dep1",\n    ...\n]
        pattern = rf"^({module}\s*=\s*\[)(.*?)(^\])"
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

        if match:
            before = match.group(1)
            existing = match.group(2)
            after = match.group(3)

            # Add new deps before the closing bracket
            new_deps = "\n".join(f'    "{dep}",' for dep in sorted(deps))
            new_section = f"{before}{existing}{new_deps}\n{after}"

            content = content[: match.start()] + new_section + content[match.end() :]
            changes.append(f"[{module}]: +{', '.join(deps)}")
        else:
            # Module doesn't have an extras section yet
            print(f"Warning: No extras section found for [{module}], skipping")

    if changes:
        print("Changes to apply:" if dry_run else "Applied changes:")
        for change in changes:
            print(f"  {change}")

        if not dry_run:
            pyproject_path.write_text(content)
            print(f"\nUpdated {pyproject_path}")
    else:
        print("No changes needed")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("--dry-run", "--apply"):
        print(__doc__)
        sys.exit(1)

    dry_run = sys.argv[1] == "--dry-run"

    print("Analyzing module dependencies...")
    detector = load_detector()
    missing = get_all_missing(detector)

    if not missing:
        print("All module dependencies are properly configured!")
        return

    print(
        f"\nFound {sum(len(v) for v in missing.values())} missing dependencies in {len(missing)} modules\n"
    )

    update_pyproject(missing, dry_run=dry_run)

    if dry_run:
        print("\nRun with --apply to make changes")


if __name__ == "__main__":
    main()
