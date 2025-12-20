#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_dataclasses/__init__.py

"""FSB Dataclasses - Core shared data models for bundles."""

# Core models (shared between fig and stats)
from ._Axes import Axes
from ._BBox import BBox
from ._ColumnDef import ColumnDef
from ._DataFormat import DataFormat
from ._DataInfo import DATA_INFO_VERSION, DataInfo
from ._DataSource import DataSource
from ._Node import Node
from ._NodeRefs import NodeRefs
from ._SizeMM import SizeMM

__all__ = [
    # Core models
    "BBox",
    "SizeMM",
    "Axes",
    "NodeRefs",
    "Node",
    # Data Info
    "DATA_INFO_VERSION",
    "DataSource",
    "DataFormat",
    "ColumnDef",
    "DataInfo",
]

# EOF
