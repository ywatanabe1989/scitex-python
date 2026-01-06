#!/usr/bin/env python3
# Timestamp: 2026-01-07
# Author: SciTeX Module Refactoring
# File: src/scitex/fig/__init__.py
#
# DEPRECATED: This module is deprecated. Use scitex.canvas instead.
# This file provides backward compatibility and will be removed in a future version.

import warnings

warnings.warn(
    "scitex.fig is deprecated. Use scitex.canvas instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from scitex.canvas
from scitex.canvas import *
from scitex.canvas import __all__

# EOF
