#!/usr/bin/env python3
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/__init__.py

"""
SciTeX Writer - LaTeX Compilation System

Python wrapper around scitex-writer shell scripts for LaTeX compilation.

Examples:
    >>> from scitex.writer import Writer, compile

    # Using Writer class
    >>> writer = Writer(project_dir=Path("."))
    >>> result = writer.compile_manuscript()

    # Using unified compile function
    >>> result = compile("manuscript", project_dir=Path("."))
    >>> results = compile("manuscript", "supplementary", project_dir=Path("."))
    >>> results = await compile("all", project_dir=Path("."), async_=True)
"""

from . import utils
from ._compile import compile
from .Writer import Writer

__all__ = [
    "Writer",
    "compile",
    "utils",
]


# Clean up namespace - hide internal submodules
def _cleanup():
    import sys

    _this = sys.modules[__name__]
    for _attr in list(vars(_this).keys()):
        if _attr in ("_dataclasses",):
            delattr(_this, _attr)


_cleanup()
del _cleanup

# EOF
