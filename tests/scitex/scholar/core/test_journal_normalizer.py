# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/journal_normalizer.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/journal_normalizer.py
# """
# Journal Name Normalizer.
# 
# Handles journal name variations, abbreviations, and historical names
# using ISSN-L as the unique identifier (single source of truth).
# 
# Data sources:
# - OpenAlex API (display_name, alternate_titles, abbreviated_title, issn_l)
# - Crossref API (container-title, short-container-title)
# - Local cache with 1-day TTL
# 
# Usage:
#     from scitex.scholar.core import JournalNormalizer
# 
#     normalizer = JournalNormalizer.get_instance()
# 
#     # Normalize any journal name variant
#     canonical = normalizer.normalize("J. Neurosci.")  # → "Journal of Neuroscience"
# 
#     # Get ISSN-L for a journal
#     issn_l = normalizer.get_issn_l("PLOS ONE")  # → "1932-6203"
# 
#     # Check if two names refer to same journal
#     normalizer.is_same_journal("J Neurosci", "Journal of Neuroscience")  # → True
# """
# 
# from __future__ import annotations
# 
# import json
# import os
# import re
# import time
# from pathlib import Path
# from typing import Optional, Set, Dict, List, Tuple, Any
# import asyncio
# import aiohttp
# 
# from scitex import logging
# 
# logger = logging.getLogger(__name__)
# 
# # Cache settings
# CACHE_TTL_SECONDS = 86400  # 1 day
# OPENALEX_SOURCES_URL = "https://api.openalex.org/sources"
# OPENALEX_POLITE_EMAIL = "research@scitex.io"
# 
# 
# def _get_default_cache_dir() -> Path:
#     """Get default cache directory respecting SCITEX_DIR env var."""
#     scitex_dir = os.environ.get("SCITEX_DIR", "~/.scitex")
#     return Path(scitex_dir).expanduser() / "scholar" / "cache"
# 
# 
# def _normalize_name(name: str) -> str:
#     """
#     Basic string normalization for matching.
# 
#     - Lowercase
#     - Remove extra whitespace
#     - Normalize punctuation
#     """
#     if not name:
#         return ""
#     # Lowercase
#     name = name.lower()
#     # Normalize whitespace
#     name = " ".join(name.split())
#     # Remove common punctuation variations
#     name = name.replace(".", "").replace(",", "").replace(":", "")
#     # Normalize ampersand
#     name = name.replace(" & ", " and ")
#     return name.strip()
# 
# 
# def _normalize_issn(issn: str) -> str:
#     """Normalize ISSN format to XXXX-XXXX."""
#     if not issn:
#         return ""
#     issn = issn.upper().replace("-", "").replace(" ", "")
#     if len(issn) == 8:
#         return f"{issn[:4]}-{issn[4:]}"
#     return issn
# 
# 
# class JournalNormalizer:
#     """
#     Journal name normalizer using ISSN-L as unique identifier.
# 
#     Handles:
#     - Full names ↔ abbreviations
#     - Name variants (spelling, punctuation, capitalization)
#     - Historical/former names
#     - Publisher variations
# 
#     Data is cached locally with daily refresh from OpenAlex.
#     """
# 
#     _instance: Optional["JournalNormalizer"] = None
# 
#     def __init__(self, cache_dir: Optional[Path] = None):
#         self._cache_dir = cache_dir or _get_default_cache_dir()
#         self._cache_file = self._cache_dir / "journal_normalizer_cache.json"
# 
#         # Core mappings (ISSN-L is the key)
#         self._issn_l_data: Dict[str, Dict[str, Any]] = {}  # ISSN-L → full metadata
# 
#         # Lookup indexes (for fast search)
#         self._name_to_issn_l: Dict[str, str] = {}  # normalized name → ISSN-L
#         self._issn_to_issn_l: Dict[str, str] = {}  # any ISSN → ISSN-L
#         self._abbrev_to_issn_l: Dict[str, str] = {}  # abbreviated name → ISSN-L
# 
#         # Stats
#         self._last_updated: float = 0
#         self._loaded = False
#         self._journal_count = 0
# 
#     @classmethod
#     def get_instance(cls, cache_dir: Optional[Path] = None) -> "JournalNormalizer":
#         """Get singleton instance."""
#         if cls._instance is None:
#             cls._instance = cls(cache_dir)
#         return cls._instance
# 
#     def _is_cache_valid(self) -> bool:
#         """Check if cache exists and is within TTL."""
#         if not self._cache_file.exists():
#             return False
#         try:
#             with open(self._cache_file, "r") as f:
#                 data = json.load(f)
#             cached_time = data.get("timestamp", 0)
#             return (time.time() - cached_time) < CACHE_TTL_SECONDS
#         except (json.JSONDecodeError, IOError):
#             return False
# 
#     def _load_from_cache(self) -> bool:
#         """Load cached data from file."""
#         if not self._cache_file.exists():
#             return False
#         try:
#             with open(self._cache_file, "r") as f:
#                 data = json.load(f)
# 
#             self._issn_l_data = data.get("issn_l_data", {})
#             self._name_to_issn_l = data.get("name_to_issn_l", {})
#             self._issn_to_issn_l = data.get("issn_to_issn_l", {})
#             self._abbrev_to_issn_l = data.get("abbrev_to_issn_l", {})
#             self._last_updated = data.get("timestamp", 0)
#             self._journal_count = len(self._issn_l_data)
#             self._loaded = True
# 
#             logger.info(f"Loaded {self._journal_count} journals from normalizer cache")
#             return True
#         except (json.JSONDecodeError, IOError) as e:
#             logger.warning(f"Failed to load journal normalizer cache: {e}")
#             return False
# 
#     def _save_to_cache(self) -> None:
#         """Save current data to cache file."""
#         try:
#             self._cache_dir.mkdir(parents=True, exist_ok=True)
#             data = {
#                 "timestamp": time.time(),
#                 "journal_count": len(self._issn_l_data),
#                 "issn_l_data": self._issn_l_data,
#                 "name_to_issn_l": self._name_to_issn_l,
#                 "issn_to_issn_l": self._issn_to_issn_l,
#                 "abbrev_to_issn_l": self._abbrev_to_issn_l,
#             }
#             with open(self._cache_file, "w") as f:
#                 json.dump(data, f)
#             logger.info(f"Saved {len(self._issn_l_data)} journals to normalizer cache")
#         except IOError as e:
#             logger.warning(f"Failed to save journal normalizer cache: {e}")
# 
#     def _add_journal(self, source_data: Dict[str, Any]) -> None:
#         """
#         Add a journal to the normalizer from OpenAlex source data.
# 
#         Args:
#             source_data: OpenAlex source object with display_name, issn_l, etc.
#         """
#         issn_l = source_data.get("issn_l")
#         if not issn_l:
#             return
# 
#         issn_l = _normalize_issn(issn_l)
#         display_name = source_data.get("display_name", "")
#         abbreviated_title = source_data.get("abbreviated_title", "")
#         alternate_titles = source_data.get("alternate_titles", []) or []
#         issns = source_data.get("issn", []) or []
#         is_oa = source_data.get("is_oa", False)
# 
#         # Store full metadata
#         self._issn_l_data[issn_l] = {
#             "canonical_name": display_name,
#             "abbreviated_title": abbreviated_title,
#             "alternate_titles": alternate_titles,
#             "issns": [_normalize_issn(i) for i in issns if i],
#             "is_oa": is_oa,
#             "publisher": source_data.get("host_organization_name", ""),
#         }
# 
#         # Build lookup indexes
#         # 1. Canonical name
#         if display_name:
#             norm_name = _normalize_name(display_name)
#             self._name_to_issn_l[norm_name] = issn_l
# 
#         # 2. Alternate titles (variants)
#         for alt in alternate_titles:
#             if alt:
#                 norm_alt = _normalize_name(alt)
#                 if norm_alt and norm_alt not in self._name_to_issn_l:
#                     self._name_to_issn_l[norm_alt] = issn_l
# 
#         # 3. Abbreviated title
#         if abbreviated_title:
#             norm_abbrev = _normalize_name(abbreviated_title)
#             self._abbrev_to_issn_l[norm_abbrev] = issn_l
#             # Also add without periods (common variation)
#             self._abbrev_to_issn_l[norm_abbrev.replace(".", "")] = issn_l
# 
#         # 4. All ISSNs → ISSN-L
#         for issn in issns:
#             if issn:
#                 norm_issn = _normalize_issn(issn)
#                 self._issn_to_issn_l[norm_issn] = issn_l
#         self._issn_to_issn_l[issn_l] = issn_l  # Self-reference
# 
#     async def _fetch_journals_async(
#         self, max_pages: int = 500, filter_oa_only: bool = False
#     ) -> None:
#         """
#         Fetch journal data from OpenAlex API.
# 
#         Args:
#             max_pages: Maximum pages to fetch (200 per page)
#             filter_oa_only: If True, only fetch OA journals
#         """
#         per_page = 200
#         cursor = "*"
#         pages_fetched = 0
# 
#         # Select fields to minimize response size
#         select_fields = "display_name,issn_l,issn,abbreviated_title,alternate_titles,is_oa,host_organization_name"
# 
#         filter_param = "is_oa:true" if filter_oa_only else "type:journal"
# 
#         async with aiohttp.ClientSession() as session:
#             while pages_fetched < max_pages:
#                 url = (
#                     f"{OPENALEX_SOURCES_URL}"
#                     f"?filter={filter_param}"
#                     f"&per_page={per_page}"
#                     f"&cursor={cursor}"
#                     f"&mailto={OPENALEX_POLITE_EMAIL}"
#                     f"&select={select_fields}"
#                 )
# 
#                 try:
#                     async with session.get(
#                         url, timeout=aiohttp.ClientTimeout(total=30)
#                     ) as resp:
#                         if resp.status != 200:
#                             logger.warning(f"OpenAlex API returned {resp.status}")
#                             break
# 
#                         data = await resp.json()
#                         results = data.get("results", [])
# 
#                         if not results:
#                             break
# 
#                         for source in results:
#                             self._add_journal(source)
# 
#                         # Get next cursor
#                         meta = data.get("meta", {})
#                         next_cursor = meta.get("next_cursor")
#                         if not next_cursor or next_cursor == cursor:
#                             break
#                         cursor = next_cursor
#                         pages_fetched += 1
# 
#                         # Progress log
#                         if pages_fetched % 20 == 0:
#                             logger.info(
#                                 f"Fetched {pages_fetched} pages, {len(self._issn_l_data)} journals..."
#                             )
# 
#                 except asyncio.TimeoutError:
#                     logger.warning("OpenAlex API timeout")
#                     break
#                 except Exception as e:
#                     logger.error(f"Error fetching journals: {e}")
#                     break
# 
#         self._journal_count = len(self._issn_l_data)
#         self._last_updated = time.time()
#         self._loaded = True
# 
#         if self._journal_count > 0:
#             self._save_to_cache()
#             logger.info(f"Fetched {self._journal_count} journals from OpenAlex")
# 
#     def _fetch_journals_sync(
#         self, max_pages: int = 500, filter_oa_only: bool = False
#     ) -> None:
#         """Synchronous wrapper for fetching journals."""
#         try:
#             loop = asyncio.get_event_loop()
#         except RuntimeError:
#             loop = asyncio.new_event_loop()
#             asyncio.set_event_loop(loop)
# 
#         loop.run_until_complete(self._fetch_journals_async(max_pages, filter_oa_only))
# 
#     def ensure_loaded(self, force_refresh: bool = False, max_pages: int = 500) -> None:
#         """
#         Ensure cache is loaded, fetching from API if needed.
# 
#         Args:
#             force_refresh: Force refresh even if cache is valid
#             max_pages: Max pages to fetch if refreshing
#         """
#         if self._loaded and not force_refresh and self._is_cache_valid():
#             return
# 
#         # Try loading from cache first
#         if not force_refresh and self._load_from_cache() and self._is_cache_valid():
#             return
# 
#         # Fetch from API
#         logger.info("Refreshing journal normalizer cache from OpenAlex...")
#         self._fetch_journals_sync(max_pages)
# 
#     # ==================== Public API ====================
# 
#     def get_issn_l(self, journal_name: str) -> Optional[str]:
#         """
#         Get ISSN-L for a journal name.
# 
#         Args:
#             journal_name: Any journal name variant, abbreviation, or ISSN
# 
#         Returns:
#             ISSN-L if found, None otherwise
#         """
#         self.ensure_loaded()
# 
#         if not journal_name:
#             return None
# 
#         # Check if it's an ISSN
#         if re.match(r"^\d{4}-?\d{3}[\dXx]$", journal_name.replace(" ", "")):
#             norm_issn = _normalize_issn(journal_name)
#             if norm_issn in self._issn_to_issn_l:
#                 return self._issn_to_issn_l[norm_issn]
# 
#         # Try normalized name lookup
#         norm_name = _normalize_name(journal_name)
# 
#         # Check full names
#         if norm_name in self._name_to_issn_l:
#             return self._name_to_issn_l[norm_name]
# 
#         # Check abbreviations
#         if norm_name in self._abbrev_to_issn_l:
#             return self._abbrev_to_issn_l[norm_name]
# 
#         return None
# 
#     def normalize(self, journal_name: str) -> Optional[str]:
#         """
#         Normalize journal name to canonical form.
# 
#         Args:
#             journal_name: Any journal name variant
# 
#         Returns:
#             Canonical journal name, or original if not found
#         """
#         issn_l = self.get_issn_l(journal_name)
#         if issn_l and issn_l in self._issn_l_data:
#             return self._issn_l_data[issn_l].get("canonical_name", journal_name)
#         return journal_name
# 
#     def get_abbreviation(self, journal_name: str) -> Optional[str]:
#         """
#         Get abbreviated title for a journal.
# 
#         Args:
#             journal_name: Any journal name variant
# 
#         Returns:
#             Abbreviated title if available
#         """
#         issn_l = self.get_issn_l(journal_name)
#         if issn_l and issn_l in self._issn_l_data:
#             return self._issn_l_data[issn_l].get("abbreviated_title")
#         return None
# 
#     def get_journal_info(self, journal_name: str) -> Optional[Dict[str, Any]]:
#         """
#         Get full journal metadata.
# 
#         Args:
#             journal_name: Any journal name variant
# 
#         Returns:
#             Dict with canonical_name, abbreviated_title, alternate_titles, issns, is_oa, publisher
#         """
#         issn_l = self.get_issn_l(journal_name)
#         if issn_l and issn_l in self._issn_l_data:
#             return {"issn_l": issn_l, **self._issn_l_data[issn_l]}
#         return None
# 
#     def is_same_journal(self, name1: str, name2: str) -> bool:
#         """
#         Check if two names refer to the same journal.
# 
#         Args:
#             name1: First journal name
#             name2: Second journal name
# 
#         Returns:
#             True if both names resolve to the same ISSN-L
#         """
#         issn_l_1 = self.get_issn_l(name1)
#         issn_l_2 = self.get_issn_l(name2)
# 
#         if issn_l_1 and issn_l_2:
#             return issn_l_1 == issn_l_2
# 
#         # Fallback: simple normalization comparison
#         return _normalize_name(name1) == _normalize_name(name2)
# 
#     def is_open_access(self, journal_name: str) -> bool:
#         """
#         Check if journal is Open Access.
# 
#         Args:
#             journal_name: Any journal name variant
# 
#         Returns:
#             True if journal is OA
#         """
#         issn_l = self.get_issn_l(journal_name)
#         if issn_l and issn_l in self._issn_l_data:
#             return self._issn_l_data[issn_l].get("is_oa", False)
#         return False
# 
#     def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
#         """
#         Search for journals by name (prefix/substring match).
# 
#         Args:
#             query: Search query
#             limit: Maximum results
# 
#         Returns:
#             List of matching journal info dicts
#         """
#         self.ensure_loaded()
# 
#         if not query:
#             return []
# 
#         norm_query = _normalize_name(query)
#         results = []
# 
#         for norm_name, issn_l in self._name_to_issn_l.items():
#             if norm_query in norm_name:
#                 if issn_l in self._issn_l_data:
#                     results.append({"issn_l": issn_l, **self._issn_l_data[issn_l]})
#                     if len(results) >= limit:
#                         break
# 
#         return results
# 
#     @property
#     def journal_count(self) -> int:
#         """Get number of cached journals."""
#         self.ensure_loaded()
#         return self._journal_count
# 
#     @property
#     def cache_age_hours(self) -> float:
#         """Get cache age in hours."""
#         if self._last_updated == 0:
#             return float("inf")
#         return (time.time() - self._last_updated) / 3600
# 
# 
# # ==================== Convenience Functions ====================
# 
# 
# def get_journal_normalizer(cache_dir: Optional[Path] = None) -> JournalNormalizer:
#     """Get the journal normalizer singleton."""
#     return JournalNormalizer.get_instance(cache_dir)
# 
# 
# def normalize_journal_name(name: str) -> str:
#     """Normalize journal name to canonical form."""
#     return get_journal_normalizer().normalize(name)
# 
# 
# def get_journal_issn_l(name: str) -> Optional[str]:
#     """Get ISSN-L for a journal name."""
#     return get_journal_normalizer().get_issn_l(name)
# 
# 
# def is_same_journal(name1: str, name2: str) -> bool:
#     """Check if two names refer to the same journal."""
#     return get_journal_normalizer().is_same_journal(name1, name2)
# 
# 
# def refresh_journal_cache() -> None:
#     """Force refresh the journal normalizer cache."""
#     get_journal_normalizer().ensure_loaded(force_refresh=True)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/journal_normalizer.py
# --------------------------------------------------------------------------------
