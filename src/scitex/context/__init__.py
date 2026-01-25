#!/usr/bin/env python3
"""Scitex context module."""

from ._detect_environment import (
    detect_environment,
    get_output_directory,
    is_ipython,
    is_notebook,
    is_script,
)
from ._get_notebook_path import (
    get_notebook_directory,
    get_notebook_info_simple,
    get_notebook_name,
    get_notebook_path,
)
from ._suppress_output import quiet, suppress_output

__all__ = [
    "detect_environment",
    "get_notebook_directory",
    "get_notebook_info_simple",
    "get_notebook_name",
    "get_notebook_path",
    "get_output_directory",
    "is_ipython",
    "is_notebook",
    "is_script",
    "quiet",
    "suppress_output",
]
