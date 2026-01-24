#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/__init__.py

"""
Scholar mixin classes for modular functionality.

Each mixin provides a specific set of methods for the Scholar class.
"""

from ._enrichers import EnricherMixin
from ._library_handlers import LibraryHandlerMixin
from ._loaders import LoaderMixin
from ._pdf_download import PDFDownloadMixin
from ._pipeline import PipelineMixin
from ._project_handlers import ProjectHandlerMixin
from ._savers import SaverMixin
from ._search import SearchMixin
from ._services import ServiceMixin
from ._url_finding import URLFindingMixin

__all__ = [
    "EnricherMixin",
    "URLFindingMixin",
    "PDFDownloadMixin",
    "LoaderMixin",
    "SearchMixin",
    "SaverMixin",
    "ProjectHandlerMixin",
    "LibraryHandlerMixin",
    "PipelineMixin",
    "ServiceMixin",
]


# EOF
