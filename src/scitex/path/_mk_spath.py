#!/usr/bin/env python3
# Timestamp: "2026-01-08 02:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/path/_mk_spath.py

"""Save path creation utilities."""

import inspect
import os
from pathlib import Path
from typing import Union


def mk_spath(sfname: Union[str, Path], makedirs: bool = False) -> Path:
    """Create a save path based on the calling script's location.

    Parameters
    ----------
    sfname : str or Path
        The name of the file to be saved.
    makedirs : bool, optional
        If True, create the directory structure for the save path.

    Returns
    -------
    Path
        The full save path for the file.

    Example
    -------
    >>> spath = mk_spath('output.txt', makedirs=True)
    >>> print(spath)
    Path('/path/to/current/script_out/output.txt')
    """
    caller_file = inspect.stack()[1].filename
    if "ipython" in caller_file.lower():
        caller_file = f"/tmp/fake-{os.getenv('USER')}.py"

    fpath = Path(caller_file)
    sdir = fpath.parent / f"{fpath.stem}_out"
    spath = sdir / sfname

    if makedirs:
        spath.parent.mkdir(parents=True, exist_ok=True)

    return spath


# EOF
