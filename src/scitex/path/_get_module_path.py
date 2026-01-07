#!/usr/bin/env python3
# Timestamp: "2026-01-08 02:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/path/_get_module_path.py

"""Module path utilities."""

import importlib.util
from pathlib import Path


def get_data_path_from_a_package(package_str: str, resource: str) -> Path:
    """Get the path to a data file within a package.

    Parameters
    ----------
    package_str : str
        The name of the package as a string.
    resource : str
        The name of the resource file within the package's data directory.

    Returns
    -------
    Path
        The full path to the resource file.

    Raises
    ------
    ImportError
        If the specified package cannot be found.
    FileNotFoundError
        If the resource file does not exist in the package's data directory.
    """
    spec = importlib.util.find_spec(package_str)
    if spec is None:
        raise ImportError(f"Package '{package_str}' not found")

    origin = Path(spec.origin)
    data_dir = origin.parents[0]
    while "src" not in str(data_dir) and data_dir != data_dir.parent:
        data_dir = data_dir.parent
    data_dir = data_dir.parent / "data"

    resource_path = data_dir / resource

    if not resource_path.exists():
        raise FileNotFoundError(
            f"Resource '{resource}' not found in package '{package_str}'"
        )

    return resource_path


# EOF
