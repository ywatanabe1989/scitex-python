#!/usr/bin/env python3
# Timestamp: 2026-02-05
# File: scitex/cloud/__init__.py

"""SciTeX Cloud - Web service integration.

This module delegates to the scitex-cloud package (Django web application).
Install separately: pip install scitex-cloud

Architecture:
    scitex (hub) → stx.cloud → scitex_cloud (spoke package)

Example:
    >>> import scitex as stx
    >>> stx.cloud.get_version()
    '0.7.0a0'
    >>> stx.cloud.health_check()
    {'status': 'healthy', ...}
"""

from __future__ import annotations

import os as _os

# Set branding BEFORE importing scitex-cloud
_os.environ.setdefault("SCITEX_CLOUD_BRAND", "scitex.cloud")
_os.environ.setdefault("SCITEX_CLOUD_ALIAS", "cloud")

__all__ = [
    "get_version",
    "health_check",
    "AVAILABLE",
]

AVAILABLE = False
_import_error_msg = None

try:
    from scitex_cloud import get_version, health_check

    AVAILABLE = True
except ImportError as e:
    _import_error_msg = str(e)

    def get_version() -> str:
        """Get scitex-cloud version (requires scitex-cloud package)."""
        raise ImportError(
            "scitex-cloud package not installed. "
            "Install with: pip install scitex-cloud\n"
            f"Original error: {_import_error_msg}"
        )

    def health_check() -> dict:
        """Check scitex-cloud health (requires scitex-cloud package)."""
        raise ImportError(
            "scitex-cloud package not installed. "
            "Install with: pip install scitex-cloud\n"
            f"Original error: {_import_error_msg}"
        )


# EOF
