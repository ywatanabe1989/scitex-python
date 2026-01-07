#!/usr/bin/env python3
# Timestamp: "2026-01-08 02:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/path/_increment_version.py

"""File versioning utilities."""

import re
from pathlib import Path
from typing import Union


def increment_version(
    dirname: Union[str, Path],
    fname: str,
    ext: str,
    version_prefix: str = "_v",
) -> Path:
    """Generate the next version of a filename based on existing versioned files.

    This function searches for files in the given directory that match the pattern:
    {fname}{version_prefix}{number}{ext} and returns the path for the next version.

    Parameters
    ----------
    dirname : str or Path
        The directory to search in and where the new file will be created.
    fname : str
        The base filename without version number or extension.
    ext : str
        The file extension, including the dot (e.g., '.txt').
    version_prefix : str, optional
        The prefix used before the version number. Default is '_v'.

    Returns
    -------
    Path
        The full path for the next version of the file.

    Example
    -------
    >>> increment_version('/path/to/dir', 'myfile', '.txt')
    Path('/path/to/dir/myfile_v004.txt')

    Notes
    -----
    - If no existing versioned files are found, it starts with version 001.
    - The version number is always formatted with at least 3 digits.
    """
    dirname = Path(dirname)

    version_pattern = re.compile(
        rf"({re.escape(fname)}{re.escape(version_prefix)})(\d+)({re.escape(ext)})$"
    )

    glob_pattern = f"{fname}{version_prefix}*{ext}"
    files = list(dirname.glob(glob_pattern))

    highest_version = 0
    base, suffix = None, None

    for file in files:
        match = version_pattern.search(file.name)
        if match:
            base, version_str, suffix = match.groups()
            version_num = int(version_str)
            if version_num > highest_version:
                highest_version = version_num

    if base is None or suffix is None:
        base = f"{fname}{version_prefix}"
        suffix = ext
        highest_version = 0

    next_version_number = highest_version + 1
    next_version_str = f"{base}{next_version_number:03d}{suffix}"

    return dirname / next_version_str


# EOF
