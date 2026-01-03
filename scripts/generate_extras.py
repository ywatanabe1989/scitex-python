#!/usr/bin/env python3
# Time-stamp: "2026-01-04 (ywatanabe)"
# File: ./scripts/generate_extras.py

"""
Generate pyproject.toml optional-dependencies from requirements files.

Reads requirements from config/requirements/*.txt and generates
the [project.optional-dependencies] section for pyproject.toml.

Usage:
    python scripts/generate_extras.py              # Print to stdout
    python scripts/generate_extras.py --update     # Update pyproject.toml in-place
    python scripts/generate_extras.py --check      # Check if pyproject.toml is up-to-date
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DIR = PROJECT_ROOT / "config" / "requirements"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"


def parse_requirements_file(filepath: Path) -> list[str]:
    """Parse a requirements.txt file and return list of packages."""
    packages = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Handle version specifiers
            packages.append(line)
    return packages


def load_all_requirements() -> dict[str, list[str]]:
    """Load all requirements files from config/requirements/."""
    extras = {}

    if not REQUIREMENTS_DIR.exists():
        print(f"Error: {REQUIREMENTS_DIR} does not exist", file=sys.stderr)
        sys.exit(1)

    for req_file in sorted(REQUIREMENTS_DIR.glob("*.txt")):
        extra_name = req_file.stem  # filename without extension
        packages = parse_requirements_file(req_file)
        if packages:
            extras[extra_name] = packages

    return extras


def generate_toml_section(extras: dict[str, list[str]]) -> str:
    """Generate the [project.optional-dependencies] TOML section."""
    lines = [
        "[project.optional-dependencies]",
        "# Auto-generated from config/requirements/*.txt",
        "# Run: python scripts/generate_extras.py --update",
        "",
    ]

    for extra_name, packages in sorted(extras.items()):
        lines.append(f"# {extra_name} module")
        lines.append(f"{extra_name} = [")
        for pkg in packages:
            lines.append(f'    "{pkg}",')
        lines.append("]")
        lines.append("")

    # Generate 'all' extra that combines all module extras (excluding dev)
    all_packages = set()
    for name, pkgs in extras.items():
        if name != "dev":
            all_packages.update(pkgs)

    lines.append("# All modules (excluding dev)")
    lines.append("all = [")
    for pkg in sorted(all_packages):
        lines.append(f'    "{pkg}",')
    lines.append("]")
    lines.append("")

    return "\n".join(lines)


def update_pyproject_toml(new_section: str) -> bool:
    """Update pyproject.toml with new optional-dependencies section."""
    content = PYPROJECT_PATH.read_text(encoding="utf-8")

    # Pattern to match the [project.optional-dependencies] section
    # This matches from the header to the next [section] or end of file
    pattern = r"\[project\.optional-dependencies\].*?(?=\n\[(?!project\.optional)|$)"

    if re.search(pattern, content, re.DOTALL):
        # Replace existing section
        new_content = re.sub(pattern, new_section, content, flags=re.DOTALL)
    else:
        # Add new section before [tool.*] sections
        tool_match = re.search(r"\n\[tool\.", content)
        if tool_match:
            insert_pos = tool_match.start()
            new_content = (
                content[:insert_pos] + "\n" + new_section + "\n" + content[insert_pos:]
            )
        else:
            new_content = content + "\n" + new_section + "\n"

    PYPROJECT_PATH.write_text(new_content, encoding="utf-8")
    return True


def check_pyproject_toml(new_section: str) -> bool:
    """Check if pyproject.toml matches the generated section."""
    content = PYPROJECT_PATH.read_text(encoding="utf-8")
    return new_section in content


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate pyproject.toml extras")
    parser.add_argument(
        "--update", action="store_true", help="Update pyproject.toml in-place"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check if pyproject.toml is up-to-date"
    )
    args = parser.parse_args()

    extras = load_all_requirements()
    section = generate_toml_section(extras)

    if args.check:
        if check_pyproject_toml(section):
            print("pyproject.toml is up-to-date")
            sys.exit(0)
        else:
            print(
                "pyproject.toml needs updating. Run: python scripts/generate_extras.py --update"
            )
            sys.exit(1)
    elif args.update:
        update_pyproject_toml(section)
        print(f"Updated {PYPROJECT_PATH}")
        print(f"Generated extras: {', '.join(sorted(extras.keys()))}")
    else:
        print(section)


if __name__ == "__main__":
    main()

# EOF
