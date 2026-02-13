#!/usr/bin/env python3
"""Backward compatibility: scitex.verify is now scitex.clew."""

import warnings

warnings.warn(
    "scitex.verify is deprecated, use scitex.clew instead",
    DeprecationWarning,
    stacklevel=2,
)

from scitex.clew import *  # noqa: F401,F403
