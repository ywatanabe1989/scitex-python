#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_conversion/__init__.py

"""FTS Conversion Utilities."""

from .bundle2dict import bundle_to_dict
from .dict2bundle import dict_to_bundle

__all__ = [
    "bundle_to_dict",
    "dict_to_bundle",
]

# EOF
