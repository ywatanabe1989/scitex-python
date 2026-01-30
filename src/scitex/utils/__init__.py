#!/usr/bin/env python3
"""Scitex utils module."""

from ._compress_hdf5 import compress_hdf5
from ._email import ansi_escape as _ansi_escape
from ._email import send_gmail as _send_gmail
from ._grid import count_grids, yield_grids
from ._notify import (
    gen_footer as _gen_footer,
)
from ._notify import (
    get_git_branch as _get_git_branch,
)
from ._notify import (
    get_hostname as _get_hostname,
)
from ._notify import (
    get_username as _get_username,
)
from ._notify import (
    notify,
)
from ._search import search

__all__ = [
    # Public API
    "compress_hdf5",
    "count_grids",
    "yield_grids",
    "notify",
    "search",
]
