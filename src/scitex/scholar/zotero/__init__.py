#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zotero integration for SciTeX Scholar.

DEPRECATED: This location is maintained for backward compatibility.
Please use: from scitex.scholar.integration.zotero import ...

This module provides bidirectional integration with Zotero reference manager:
- Import: Bibliography with collections/tags, PDF annotations, paper metadata
- Export: Manuscripts as preprint entries, project metadata, citation files (.bib, .ris)
- Link: Live citation insertion, auto-update on library changes, tagged items
"""

import warnings

# Show deprecation warning
warnings.warn(
    "Importing from scitex.scholar.zotero is deprecated. "
    "Please use: from scitex.scholar.integration.zotero import ...",
    DeprecationWarning,
    stacklevel=2,
)

# Import from new location for backward compatibility
from scitex.scholar.integration.zotero import (
    ZoteroImporter,
    ZoteroExporter,
    ZoteroLinker,
    ZoteroMapper,
)

__all__ = [
    "ZoteroImporter",
    "ZoteroExporter",
    "ZoteroLinker",
    "ZoteroMapper",
]
