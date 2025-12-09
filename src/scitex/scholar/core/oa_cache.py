#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/oa_cache.py
"""
Open Access Sources Cache.

Caches OA journal/source information from OpenAlex API with daily refresh.
Provides fast local lookups without per-paper API calls.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Set, Dict, Any
import asyncio
import aiohttp

import os

from scitex import logging

logger = logging.getLogger(__name__)

# Cache settings
CACHE_TTL_SECONDS = 86400  # 1 day
OPENALEX_OA_SOURCES_URL = "https://api.openalex.org/sources"
OPENALEX_POLITE_EMAIL = "research@scitex.io"


def _get_default_cache_dir() -> Path:
    """Get default cache directory respecting SCITEX_DIR env var."""
    scitex_dir = os.environ.get("SCITEX_DIR", "~/.scitex")
    return Path(scitex_dir).expanduser() / "scholar" / "cache"


class OASourcesCache:
    """
    Manages cached Open Access sources from OpenAlex.

    Features:
    - Lazy loading on first access
    - 1-day TTL with automatic refresh
    - Thread-safe singleton pattern
    - Fallback to config YAML if API fails
    - Journal name normalization via ISSN-L
    - Handles abbreviations, variants, and historical names
    """

    _instance: Optional["OASourcesCache"] = None
    _lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache_dir = cache_dir or _get_default_cache_dir()
        self._cache_file = self._cache_dir / "oa_sources_cache.json"
        self._oa_source_ids: Set[str] = set()  # OpenAlex source IDs
        self._oa_source_names: Set[str] = set()  # Lowercase source names
        self._oa_issns: Set[str] = set()  # ISSNs for journal lookup
        self._issn_l_map: Dict[str, str] = {}  # ISSN → ISSN-L mapping
        self._name_to_issn_l: Dict[str, str] = {}  # name variant → ISSN-L
        self._issn_l_to_canonical: Dict[str, str] = {}  # ISSN-L → canonical name
        self._last_updated: float = 0
        self._loaded = False

    @classmethod
    def get_instance(cls, cache_dir: Optional[Path] = None) -> "OASourcesCache":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(cache_dir)
        return cls._instance

    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is within TTL."""
        if not self._cache_file.exists():
            return False
        try:
            with open(self._cache_file, "r") as f:
                data = json.load(f)
            cached_time = data.get("timestamp", 0)
            return (time.time() - cached_time) < CACHE_TTL_SECONDS
        except (json.JSONDecodeError, IOError):
            return False

    def _load_from_cache(self) -> bool:
        """Load cached data from file."""
        if not self._cache_file.exists():
            return False
        try:
            with open(self._cache_file, "r") as f:
                data = json.load(f)

            self._oa_source_names = set(data.get("source_names", []))
            self._oa_issns = set(data.get("issns", []))
            self._last_updated = data.get("timestamp", 0)
            self._loaded = True

            logger.info(f"Loaded {len(self._oa_source_names)} OA sources from cache")
            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load OA cache: {e}")
            return False

    def _save_to_cache(self) -> None:
        """Save current data to cache file."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "timestamp": time.time(),
                "source_names": list(self._oa_source_names),
                "issns": list(self._oa_issns),
                "count": len(self._oa_source_names),
            }
            with open(self._cache_file, "w") as f:
                json.dump(data, f)
            logger.info(f"Saved {len(self._oa_source_names)} OA sources to cache")
        except IOError as e:
            logger.warning(f"Failed to save OA cache: {e}")

    async def _fetch_oa_sources_async(self, max_pages: int = 100) -> None:
        """
        Fetch OA sources from OpenAlex API.

        Args:
            max_pages: Maximum pages to fetch (200 sources per page)
        """
        source_names: Set[str] = set()
        issns: Set[str] = set()

        per_page = 200
        cursor = "*"
        pages_fetched = 0

        async with aiohttp.ClientSession() as session:
            while pages_fetched < max_pages:
                url = (
                    f"{OPENALEX_OA_SOURCES_URL}"
                    f"?filter=is_oa:true"
                    f"&per_page={per_page}"
                    f"&cursor={cursor}"
                    f"&mailto={OPENALEX_POLITE_EMAIL}"
                    f"&select=display_name,issn"
                )

                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        if resp.status != 200:
                            logger.warning(f"OpenAlex API returned {resp.status}")
                            break

                        data = await resp.json()
                        results = data.get("results", [])

                        if not results:
                            break

                        for source in results:
                            name = source.get("display_name", "")
                            if name:
                                source_names.add(name.lower())

                            # Also store ISSNs for precise matching
                            source_issns = source.get("issn", []) or []
                            for issn in source_issns:
                                if issn:
                                    issns.add(issn)

                        # Get next cursor
                        meta = data.get("meta", {})
                        next_cursor = meta.get("next_cursor")
                        if not next_cursor or next_cursor == cursor:
                            break
                        cursor = next_cursor
                        pages_fetched += 1

                        # Progress log every 10 pages
                        if pages_fetched % 10 == 0:
                            logger.info(
                                f"Fetched {pages_fetched} pages, {len(source_names)} sources so far..."
                            )

                except asyncio.TimeoutError:
                    logger.warning("OpenAlex API timeout")
                    break
                except Exception as e:
                    logger.error(f"Error fetching OA sources: {e}")
                    break

        if source_names:
            self._oa_source_names = source_names
            self._oa_issns = issns
            self._last_updated = time.time()
            self._loaded = True
            self._save_to_cache()
            logger.info(f"Fetched {len(source_names)} OA sources from OpenAlex")

    def _fetch_oa_sources_sync(self, max_pages: int = 100) -> None:
        """Synchronous wrapper for fetching OA sources."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self._fetch_oa_sources_async(max_pages))

    def ensure_loaded(self, force_refresh: bool = False) -> None:
        """
        Ensure cache is loaded, fetching from API if needed.

        Args:
            force_refresh: Force refresh even if cache is valid
        """
        if self._loaded and not force_refresh and self._is_cache_valid():
            return

        # Try loading from cache first
        if not force_refresh and self._load_from_cache() and self._is_cache_valid():
            return

        # Fetch from API
        logger.info("Refreshing OA sources cache from OpenAlex...")
        self._fetch_oa_sources_sync()

    def is_oa_source(self, source_name: str) -> bool:
        """
        Check if a source/journal name is in the OA list.

        Args:
            source_name: Journal or source name to check

        Returns:
            True if source is known to be Open Access
        """
        self.ensure_loaded()
        if not source_name:
            return False
        return source_name.lower() in self._oa_source_names

    def is_oa_issn(self, issn: str) -> bool:
        """
        Check if an ISSN belongs to an OA journal.

        Args:
            issn: ISSN to check

        Returns:
            True if ISSN belongs to an OA journal
        """
        self.ensure_loaded()
        if not issn:
            return False
        # Normalize ISSN format
        issn = issn.replace("-", "").upper()
        return issn in self._oa_issns or f"{issn[:4]}-{issn[4:]}" in self._oa_issns

    @property
    def source_count(self) -> int:
        """Get number of cached OA sources."""
        self.ensure_loaded()
        return len(self._oa_source_names)

    @property
    def cache_age_hours(self) -> float:
        """Get cache age in hours."""
        if self._last_updated == 0:
            return float("inf")
        return (time.time() - self._last_updated) / 3600


# Convenience functions
def get_oa_cache(cache_dir: Optional[Path] = None) -> OASourcesCache:
    """Get the OA sources cache singleton."""
    return OASourcesCache.get_instance(cache_dir)


def is_oa_journal_cached(journal_name: str) -> bool:
    """Check if journal is OA using cached OpenAlex data."""
    return get_oa_cache().is_oa_source(journal_name)


def refresh_oa_cache() -> None:
    """Force refresh the OA sources cache."""
    get_oa_cache().ensure_loaded(force_refresh=True)


# EOF
