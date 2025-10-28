#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/tree/__init__.py

"""
Tree dataclasses for Writer module (internal use).

Provides dataclasses for directory tree structures at the project level.
Not intended for public API - access trees through Writer class.
"""

# Import for internal use only
from ._ConfigTree import ConfigTree
from ._SharedTree import SharedTree
from ._ScriptsTree import ScriptsTree
from ._ManuscriptTree import ManuscriptTree
from ._SupplementaryTree import SupplementaryTree
from ._RevisionTree import RevisionTree

# Do not expose in __all__ - internal use only
__all__ = []

# EOF
