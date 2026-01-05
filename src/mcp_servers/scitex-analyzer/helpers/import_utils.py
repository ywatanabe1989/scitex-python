#!/usr/bin/env python3
# Timestamp: "2025-12-28 (ywatanabe)"
# File: ./mcp_servers/scitex-analyzer/helpers/import_utils.py

"""Import classification utilities."""

from typing import Optional

from ..constants import STDLIB_MODULES


def classify_import(module_name: Optional[str]) -> str:
    """Classify import type based on module name.

    Parameters
    ----------
    module_name : str or None
        The module name to classify

    Returns
    -------
    str
        Import type: 'stdlib', 'scitex', 'local', 'third_party', or 'unknown'
    """
    if not module_name:
        return "unknown"

    if module_name in STDLIB_MODULES or module_name.split(".")[0] in STDLIB_MODULES:
        return "stdlib"
    elif module_name == "scitex" or module_name.startswith("scitex."):
        return "scitex"
    elif module_name.startswith("."):
        return "local"
    else:
        return "third_party"


# EOF
