#!/usr/bin/env python3
"""
DEPRECATED: The 'rng' module has been merged into 'repro'.

This module exists only for backward compatibility and will be removed in a
future version.

Please update your imports from:
    from scitex.rng import RandomStateManager
to:
    from scitex.repro import RandomStateManager
"""

import warnings

warnings.warn(
    "The 'rng' module is deprecated and has been merged into 'repro'. "
    "Please update your imports to use 'scitex.repro' instead of "
    "'scitex.rng'. This compatibility layer will be removed in a "
    "future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from the new module for backward compatibility
from scitex.repro import RandomStateManager, get, reset  # noqa: F401

__all__ = [
    "RandomStateManager",
    "get",
    "reset",
]
