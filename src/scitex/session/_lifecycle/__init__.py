#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/session/_lifecycle/__init__.py
"""Session lifecycle management - refactored subpackage.

This package contains the split modules from the original _lifecycle.py:
- _start.py: Session start function
- _close.py: Session close and running2finished functions
- _config.py: Configuration setup
- _matplotlib.py: Matplotlib configuration
- _utils.py: Utility functions
"""

from ._close import close, running2finished
from ._start import start

__all__ = [
    "start",
    "close",
    "running2finished",
]

# EOF
