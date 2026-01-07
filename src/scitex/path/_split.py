#!/usr/bin/env python3
# Timestamp: "2026-01-08 02:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/path/_split.py

"""Path splitting utilities."""

from pathlib import Path
from typing import Tuple, Union


def split(fpath: Union[str, Path]) -> Tuple[Path, str, str]:
    """Split a file path into directory, filename, and extension.

    Parameters
    ----------
    fpath : str or Path
        File path to split.

    Returns
    -------
    tuple of (Path, str, str)
        (directory, filename without extension, extension)

    Example
    -------
    >>> dirname, fname, ext = split('../data/01/day1/tt8-2.mat')
    >>> print(dirname)  # Path('../data/01/day1')
    >>> print(fname)    # 'tt8-2'
    >>> print(ext)      # '.mat'
    """
    path = Path(fpath)
    return path.parent, path.stem, path.suffix


# EOF
