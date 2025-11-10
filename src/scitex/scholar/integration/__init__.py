#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference Manager Integrations for SciTeX Scholar.

Unified integration framework for multiple reference management systems:
- Zotero
- Mendeley
- RefWorks
- Paperpile
- EndNote
- CiteDrive
- Papers

All integrations follow the same pattern:
- Import: Bibliography with collections/tags, annotations, metadata
- Export: Papers, manuscripts, citation files
- Link: Live synchronization, citation insertion
"""

from .base import (
    BaseImporter,
    BaseExporter,
    BaseLinker,
    BaseMapper,
)

# Import all reference manager modules
from .zotero import (
    ZoteroImporter,
    ZoteroExporter,
    ZoteroLinker,
    ZoteroMapper,
)

from .mendeley import (
    MendeleyImporter,
    MendeleyExporter,
    MendeleyLinker,
    MendeleyMapper,
)

__all__ = [
    # Base classes
    "BaseImporter",
    "BaseExporter",
    "BaseLinker",
    "BaseMapper",
    # Zotero
    "ZoteroImporter",
    "ZoteroExporter",
    "ZoteroLinker",
    "ZoteroMapper",
    # Mendeley
    "MendeleyImporter",
    "MendeleyExporter",
    "MendeleyLinker",
    "MendeleyMapper",
]
