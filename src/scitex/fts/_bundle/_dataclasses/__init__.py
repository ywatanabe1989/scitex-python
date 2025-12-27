#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/__init__.py

"""FTS Dataclasses - Core shared data models for bundles."""

# Core models (shared between fig and stats)
from ._Axes import Axes
from ._BBox import BBox
from ._ColumnDef import ColumnDef
from ._DataFormat import DataFormat
from ._DataInfo import DATA_INFO_VERSION, DataInfo
from ._DataSource import DataSource
from ._Node import Node, ShapeParams, TextContent
from ._NodeRefs import NodeRefs
from ._SizeMM import SizeMM

__all__ = [
    # Core models
    "BBox",
    "SizeMM",
    "Axes",
    "NodeRefs",
    "Node",
    "TextContent",
    "ShapeParams",
    # Data Info
    "DATA_INFO_VERSION",
    "DataSource",
    "DataFormat",
    "ColumnDef",
    "DataInfo",
]

# EOF
