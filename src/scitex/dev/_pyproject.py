#!/usr/bin/env python3
# Timestamp: 2025-01-08
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/_pyproject.py

"""
Utility for programmatically managing pyproject.toml dependencies.

Usage:
    from scitex.dev import pyproject

    # Load and inspect
    pp = pyproject.load()
    extras = pyproject.get_extras(pp)

    # Audit dependencies
    pyproject.print_report()
    pyproject.validate_heavy_sync()

    # Find issues
    duplicates = pyproject.find_duplicates()
    missing = pyproject.find_missing_heavy_deps()
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    import tomlkit

    TOMLKIT_AVAILABLE = True
except ImportError:
    TOMLKIT_AVAILABLE = False
    tomlkit = None


def _get_default_path() -> Path:
    """Get default pyproject.toml path (project root)."""
    # Navigate from this file to project root
    current = Path(__file__).resolve()
    # src/scitex/dev/_pyproject.py -> project root
    for _ in range(4):
        current = current.parent
    return current / "pyproject.toml"


def load(path: Optional[Path] = None) -> dict:
    """
    Load pyproject.toml preserving comments and formatting.

    Parameters
    ----------
    path : Path, optional
        Path to pyproject.toml. Defaults to project root.

    Returns
    -------
    dict
        Parsed TOML document (tomlkit.TOMLDocument)
    """
    if not TOMLKIT_AVAILABLE:
        raise ImportError(
            "tomlkit is required for pyproject utilities. "
            "Install with: pip install tomlkit"
        )

    path = path or _get_default_path()
    with open(path) as f:
        return tomlkit.load(f)


def save(doc: dict, path: Optional[Path] = None) -> None:
    """
    Save pyproject.toml preserving comments and formatting.

    Parameters
    ----------
    doc : dict
        TOML document to save
    path : Path, optional
        Path to pyproject.toml. Defaults to project root.
    """
    if not TOMLKIT_AVAILABLE:
        raise ImportError("tomlkit is required")

    path = path or _get_default_path()
    with open(path, "w") as f:
        tomlkit.dump(doc, f)


def get_extras(doc: Optional[dict] = None) -> Dict[str, List[str]]:
    """
    Get all optional dependency extras.

    Returns
    -------
    dict
        Mapping of extra name to list of dependencies
    """
    doc = doc or load()
    return dict(doc.get("project", {}).get("optional-dependencies", {}))


def get_core_deps(doc: Optional[dict] = None) -> List[str]:
    """Get core dependencies."""
    doc = doc or load()
    return list(doc.get("project", {}).get("dependencies", []))


def get_heavy_deps(doc: Optional[dict] = None) -> Set[str]:
    """Get dependencies in the [heavy] extra."""
    extras = get_extras(doc)
    return set(extras.get("heavy", []))


def parse_commented_deps(path: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Parse commented-out dependencies from pyproject.toml.

    Looks for patterns like:
        # "torch",
        # "mne",

    Returns
    -------
    dict
        Mapping of extra name to list of commented dependencies
    """
    path = path or _get_default_path()

    commented_deps = defaultdict(list)
    current_extra = None

    with open(path) as f:
        for line in f:
            # Detect extra section
            match = re.match(r"^(\w+)\s*=\s*\[", line)
            if match:
                current_extra = match.group(1)
                continue

            # Detect end of section
            if line.strip() == "]":
                current_extra = None
                continue

            # Detect commented dependency
            if current_extra:
                match = re.match(r'^\s*#\s*"([^"]+)"', line)
                if match:
                    dep = match.group(1)
                    commented_deps[current_extra].append(dep)

    return dict(commented_deps)


def find_duplicates(doc: Optional[dict] = None) -> Dict[str, List[str]]:
    """
    Find dependencies that appear in multiple extras.

    Returns
    -------
    dict
        Mapping of dependency to list of extras containing it
    """
    extras = get_extras(doc)
    dep_locations = defaultdict(list)

    for extra_name, deps in extras.items():
        if extra_name in ("all", "dev", "heavy"):
            continue  # Skip meta-extras
        for dep in deps:
            # Normalize dep name (strip version specifiers)
            dep_name = re.split(r"[<>=\[]", dep)[0].strip().lower()
            dep_locations[dep_name].append(extra_name)

    # Return only duplicates
    return {dep: extras for dep, extras in dep_locations.items() if len(extras) > 1}


def find_missing_heavy_deps(path: Optional[Path] = None) -> List[str]:
    """
    Find commented deps that are NOT in [heavy] extra.

    Returns
    -------
    list
        Dependencies that should be added to [heavy]
    """
    doc = load(path)
    heavy_deps = get_heavy_deps(doc)
    commented = parse_commented_deps(path)

    # Normalize heavy deps
    heavy_normalized = {re.split(r"[<>=\[]", d)[0].strip().lower() for d in heavy_deps}

    missing = set()
    for extra, deps in commented.items():
        for dep in deps:
            dep_normalized = re.split(r"[<>=\[]", dep)[0].strip().lower()
            if dep_normalized not in heavy_normalized:
                missing.add(dep)

    return sorted(missing)


def validate_heavy_sync(path: Optional[Path] = None, verbose: bool = True) -> bool:
    """
    Validate that all commented deps are in [heavy] extra.

    Parameters
    ----------
    path : Path, optional
        Path to pyproject.toml
    verbose : bool
        Print validation results

    Returns
    -------
    bool
        True if all commented deps are in [heavy]
    """
    missing = find_missing_heavy_deps(path)

    if verbose:
        if missing:
            print(f"Missing from [heavy] extra ({len(missing)}):")
            for dep in missing:
                print(f"  - {dep}")
        else:
            print("All commented deps are in [heavy] extra")

    return len(missing) == 0


def get_extra_stats(doc: Optional[dict] = None) -> Dict[str, dict]:
    """
    Get statistics for each extra.

    Returns
    -------
    dict
        Mapping of extra name to stats dict
    """
    extras = get_extras(doc)
    stats = {}

    for name, deps in extras.items():
        stats[name] = {
            "count": len(deps),
            "deps": deps,
        }

    return stats


def print_report(path: Optional[Path] = None) -> None:
    """Print a comprehensive dependency report."""
    doc = load(path)
    extras = get_extras(doc)
    core = get_core_deps(doc)
    commented = parse_commented_deps(path)
    duplicates = find_duplicates(doc)
    missing = find_missing_heavy_deps(path)

    print("=" * 60)
    print("PYPROJECT.TOML DEPENDENCY REPORT")
    print("=" * 60)

    # Core deps
    print(f"\nCore dependencies: {len(core)}")
    for dep in core:
        print(f"  - {dep}")

    # Extras summary
    print(f"\nExtras ({len(extras)}):")
    for name, deps in sorted(extras.items()):
        commented_count = len(commented.get(name, []))
        suffix = f" (+{commented_count} commented)" if commented_count else ""
        print(f"  {name}: {len(deps)} deps{suffix}")

    # Duplicates
    if duplicates:
        print(f"\nDuplicates ({len(duplicates)}):")
        for dep, locations in sorted(duplicates.items()):
            print(f"  {dep}: {', '.join(locations)}")
    else:
        print("\nNo duplicates found")

    # Heavy sync status
    print("\nHeavy sync: ", end="")
    if missing:
        print(f"MISSING {len(missing)} deps")
        for dep in missing:
            print(f"  - {dep}")
    else:
        print("OK")

    print("=" * 60)


def add_to_extra(
    extra_name: str,
    deps: List[str],
    path: Optional[Path] = None,
    save_file: bool = False,
) -> dict:
    """
    Add dependencies to an extra.

    Parameters
    ----------
    extra_name : str
        Name of the extra
    deps : list
        Dependencies to add
    path : Path, optional
        Path to pyproject.toml
    save_file : bool
        If True, save changes to file

    Returns
    -------
    dict
        Updated document
    """
    doc = load(path)
    extras = doc["project"]["optional-dependencies"]

    if extra_name not in extras:
        extras[extra_name] = []

    existing = set(extras[extra_name])
    for dep in deps:
        if dep not in existing:
            extras[extra_name].append(dep)

    if save_file:
        save(doc, path)

    return doc


def remove_from_extra(
    extra_name: str,
    deps: List[str],
    path: Optional[Path] = None,
    save_file: bool = False,
) -> dict:
    """
    Remove dependencies from an extra.

    Parameters
    ----------
    extra_name : str
        Name of the extra
    deps : list
        Dependencies to remove
    path : Path, optional
        Path to pyproject.toml
    save_file : bool
        If True, save changes to file

    Returns
    -------
    dict
        Updated document
    """
    doc = load(path)
    extras = doc["project"]["optional-dependencies"]

    if extra_name in extras:
        deps_set = set(deps)
        extras[extra_name] = [d for d in extras[extra_name] if d not in deps_set]

    if save_file:
        save(doc, path)

    return doc


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "report":
            print_report()
        elif cmd == "validate":
            valid = validate_heavy_sync()
            sys.exit(0 if valid else 1)
        elif cmd == "duplicates":
            dups = find_duplicates()
            for dep, extras in sorted(dups.items()):
                print(f"{dep}: {', '.join(extras)}")
        else:
            print(f"Unknown command: {cmd}")
            print("Commands: report, validate, duplicates")
    else:
        print_report()


# EOF
