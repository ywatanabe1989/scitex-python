#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/scholar/core/_mixins/_services.py

"""
Services mixin for Scholar class.

Provides internal service properties with lazy loading.
"""

from __future__ import annotations

from pathlib import Path

from scitex.scholar.auth import ScholarAuthManager
from scitex.scholar.browser import ScholarBrowserManager
from scitex.scholar.config import ScholarConfig
from scitex.scholar.metadata_engines.ScholarEngine import ScholarEngine
from scitex.scholar.storage import LibraryManager, ScholarLibrary


class ServiceMixin:
    """Mixin providing internal service properties."""

    def _init_config(self, config) -> ScholarConfig:
        """Initialize configuration from various input types."""
        if config is None:
            return ScholarConfig.load()
        elif isinstance(config, (str, Path)):
            return ScholarConfig.from_yaml(config)
        elif isinstance(config, ScholarConfig):
            return config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

    @property
    def _scholar_engine(self) -> ScholarEngine:
        """Get Scholar engine for search and enrichment (PRIVATE)."""
        if not hasattr(self, "_ServiceMixin__scholar_engine"):
            self.__scholar_engine = None
        if self.__scholar_engine is None:
            self.__scholar_engine = ScholarEngine(config=self.config)
        return self.__scholar_engine

    @property
    def _auth_manager(self) -> ScholarAuthManager:
        """Get authentication manager service (PRIVATE)."""
        if not hasattr(self, "_ServiceMixin__auth_manager"):
            self.__auth_manager = None
        if self.__auth_manager is None:
            self.__auth_manager = ScholarAuthManager()
        return self.__auth_manager

    @property
    def _browser_manager(self) -> ScholarBrowserManager:
        """Get browser manager service (PRIVATE)."""
        if not hasattr(self, "_ServiceMixin__browser_manager"):
            self.__browser_manager = None
        if self.__browser_manager is None:
            self.__browser_manager = ScholarBrowserManager(
                auth_manager=self._auth_manager,
                chrome_profile_name="system",
                browser_mode=self.browser_mode,
            )
        return self.__browser_manager

    @property
    def _library_manager(self) -> LibraryManager:
        """Get library manager service - low-level operations (PRIVATE)."""
        if not hasattr(self, "_ServiceMixin__library_manager"):
            self.__library_manager = None
        if self.__library_manager is None:
            self.__library_manager = LibraryManager(
                project=self.project, config=self.config
            )
        return self.__library_manager

    @property
    def _library(self) -> ScholarLibrary:
        """Get Scholar library service - high-level operations (PRIVATE)."""
        if not hasattr(self, "_ServiceMixin__library"):
            self.__library = None
        if self.__library is None:
            self.__library = ScholarLibrary(project=self.project, config=self.config)
        return self.__library


# EOF
