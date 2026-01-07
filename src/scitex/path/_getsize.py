#!/usr/bin/env python3
# Timestamp: "2026-01-08 02:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/path/_getsize.py

"""File size utilities."""

from pathlib import Path
from typing import Union

import numpy as np


def getsize(path: Union[str, Path]) -> Union[int, float]:
    """Get file size in bytes.

    Parameters
    ----------
    path : str or Path
        Path to file.

    Returns
    -------
    int or float
        File size in bytes, or np.nan if file doesn't exist.
    """
    path = Path(path)
    if path.exists():
        return path.stat().st_size
    return np.nan


# EOF
