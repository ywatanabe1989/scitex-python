#!/usr/bin/env python3
# Timestamp: "2026-01-08 02:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/path/_this_path.py

"""Get current file path utilities."""

import inspect
from pathlib import Path


def this_path(ipython_fake_path: str = "/tmp/fake.py") -> Path:
    """Get the path of the calling script.

    Parameters
    ----------
    ipython_fake_path : str
        Fake path to return when running in IPython.

    Returns
    -------
    Path
        Path to the calling script.
    """
    caller_file = inspect.stack()[1].filename
    if "ipython" in caller_file.lower():
        return Path(ipython_fake_path)
    return Path(caller_file)


get_this_path = this_path


# EOF
