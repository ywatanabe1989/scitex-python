#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/compat/_fig.py

"""
DEPRECATED: scitex.fig → scitex.canvas

This module provides backward compatibility for code using scitex.fig.
Update your imports to use scitex.canvas instead.
"""

import warnings

warnings.warn(
    "scitex.fig is deprecated. Use scitex.canvas instead.",
    DeprecationWarning,
    stacklevel=3,  # Adjust for import chain: fig/__init__.py -> compat/_fig.py
)

# Re-export everything from scitex.canvas
from scitex.canvas import *

# EOF
