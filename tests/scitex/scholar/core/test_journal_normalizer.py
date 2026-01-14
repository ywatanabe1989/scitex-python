#!/usr/bin/env python3
"""
Tests for scitex.scholar.core.journal_normalizer module.

Tests cover:
- Basic string normalization functions
- ISSN normalization
- JournalNormalizer class initialization
- Journal lookup by name and ISSN
- Cache operations
"""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.core.journal_normalizer import (
    JournalNormalizer,
    _normalize_issn,
    _normalize_name,
    get_journal_normalizer,
    is_same_journal,
    normalize_journal_name,
)


class TestNormalizeNameFunction:
    """Tests for _normalize_name helper function."""

    def test_lowercase_conversion(self):
        """Should convert to lowercase."""
        assert _normalize_name("Nature") == "nature"
        assert _normalize_name("SCIENCE") == "science"
        assert _normalize_name("JouRNaL") == "journal"

    def test_whitespace_normalization(self):
        """Should normalize whitespace."""
        assert _normalize_name("Nature  Methods") == "nature methods"
        assert _normalize_name("  Nature Methods  ") == "nature methods"
        assert _normalize_name("Nature\tMethods") == "nature methods"

    def test_punctuation_removal(self):
        """Should remove common punctuation."""
        assert _normalize_name("J. Neurosci.") == "j neurosci"
        assert _normalize_name("Nature: Methods") == "nature methods"
        assert _normalize_name("Cell, Reports") == "cell reports"

    def test_ampersand_normalization(self):
        """Should normalize & to 'and'."""
        assert _normalize_name("Science & Technology") == "science and technology"
        assert _normalize_name("Food & Nutrition") == "food and nutrition"

    def test_empty_and_none(self):
        """Should handle empty and None inputs."""
        assert _normalize_name("") == ""
        assert _normalize_name(None) == ""


class TestNormalizeIssnFunction:
    """Tests for _normalize_issn helper function."""

    def test_already_formatted(self):
        """Should preserve correctly formatted ISSNs."""
        assert _normalize_issn("0028-0836") == "0028-0836"
        assert _normalize_issn("1234-567X") == "1234-567X"

    def test_without_hyphen(self):
        """Should add hyphen to unhyphenated ISSNs."""
        assert _normalize_issn("00280836") == "0028-0836"
        assert _normalize_issn("1234567X") == "1234-567X"

    def test_lowercase_x(self):
        """Should uppercase the check digit X."""
        assert _normalize_issn("1234-567x") == "1234-567X"
        assert _normalize_issn("1234567x") == "1234-567X"

    def test_with_spaces(self):
        """Should remove spaces."""
        assert _normalize_issn("0028 0836") == "0028-0836"
        assert _normalize_issn("1234 567X") == "1234-567X"

    def test_empty_and_none(self):
        """Should handle empty and invalid inputs."""
        assert _normalize_issn("") == ""
        assert _normalize_issn(None) == ""

    def test_invalid_length(self):
        """Should return as-is if not valid ISSN length."""
        assert _normalize_issn("123") == "123"
        assert _normalize_issn("123456789") == "123456789"


class TestJournalNormalizerInit:
    """Tests for JournalNormalizer initialization."""

    def test_singleton_pattern(self):
        """get_instance should return same instance."""
        # Reset singleton for test
        JournalNormalizer._instance = None

        instance1 = JournalNormalizer.get_instance()
        instance2 = JournalNormalizer.get_instance()
        assert instance1 is instance2

        # Clean up
        JournalNormalizer._instance = None

    def test_custom_cache_dir(self, tmp_path):
        """Should accept custom cache directory."""
        normalizer = JournalNormalizer(cache_dir=tmp_path)
        assert normalizer._cache_dir == tmp_path
        assert normalizer._cache_file == tmp_path / "journal_normalizer_cache.json"

    def test_initial_state(self, tmp_path):
        """Newly created normalizer should have empty state."""
        normalizer = JournalNormalizer(cache_dir=tmp_path)
        assert normalizer._loaded is False
        assert normalizer._journal_count == 0
        assert len(normalizer._issn_l_data) == 0


class TestJournalNormalizerAddJournal:
    """Tests for JournalNormalizer._add_journal method."""

    @pytest.fixture
    def normalizer(self, tmp_path):
        """Create a fresh normalizer instance."""
        return JournalNormalizer(cache_dir=tmp_path)

    def test_add_journal_basic(self, normalizer):
        """Should add journal with basic data."""
        source_data = {
            "issn_l": "0028-0836",
            "display_name": "Nature",
            "abbreviated_title": "Nature",
            "alternate_titles": ["nature"],
            "issn": ["0028-0836", "1476-4687"],
            "is_oa": False,
            "host_organization_name": "Springer Nature",
        }
        normalizer._add_journal(source_data)

        assert "0028-0836" in normalizer._issn_l_data
        assert normalizer._issn_l_data["0028-0836"]["canonical_name"] == "Nature"
        assert normalizer._issn_l_data["0028-0836"]["publisher"] == "Springer Nature"

    def test_add_journal_creates_name_index(self, normalizer):
        """Should create name-to-ISSN-L index."""
        source_data = {
            "issn_l": "0028-0836",
            "display_name": "Nature",
        }
        normalizer._add_journal(source_data)

        assert _normalize_name("Nature") in normalizer._name_to_issn_l
        assert normalizer._name_to_issn_l["nature"] == "0028-0836"

    def test_add_journal_creates_issn_index(self, normalizer):
        """Should create ISSN-to-ISSN-L index."""
        source_data = {
            "issn_l": "0028-0836",
            "display_name": "Nature",
            "issn": ["0028-0836", "1476-4687"],
        }
        normalizer._add_journal(source_data)

        assert "0028-0836" in normalizer._issn_to_issn_l
        assert "1476-4687" in normalizer._issn_to_issn_l
        assert normalizer._issn_to_issn_l["1476-4687"] == "0028-0836"

    def test_add_journal_creates_abbrev_index(self, normalizer):
        """Should create abbreviation-to-ISSN-L index."""
        source_data = {
            "issn_l": "0270-6474",
            "display_name": "Journal of Neuroscience",
            "abbreviated_title": "J. Neurosci.",
        }
        normalizer._add_journal(source_data)

        # Normalized abbreviation should be indexed
        norm_abbrev = _normalize_name("J. Neurosci.")
        assert norm_abbrev in normalizer._abbrev_to_issn_l

    def test_skip_journal_without_issn_l(self, normalizer):
        """Should skip journals without ISSN-L."""
        source_data = {
            "display_name": "Unknown Journal",
        }
        normalizer._add_journal(source_data)
        assert len(normalizer._issn_l_data) == 0


class TestJournalNormalizerLookup:
    """Tests for JournalNormalizer lookup methods."""

    @pytest.fixture
    def loaded_normalizer(self, tmp_path):
        """Create normalizer with pre-loaded test data."""
        normalizer = JournalNormalizer(cache_dir=tmp_path)

        # Add test journals
        journals = [
            {
                "issn_l": "0028-0836",
                "display_name": "Nature",
                "abbreviated_title": "Nature",
                "issn": ["0028-0836", "1476-4687"],
                "is_oa": False,
            },
            {
                "issn_l": "0270-6474",
                "display_name": "Journal of Neuroscience",
                "abbreviated_title": "J. Neurosci.",
                "issn": ["0270-6474", "1529-2401"],
                "is_oa": False,
            },
            {
                "issn_l": "1932-6203",
                "display_name": "PLOS ONE",
                "abbreviated_title": "PLoS One",
                "alternate_titles": ["PLoS ONE", "PLOS One"],
                "issn": ["1932-6203"],
                "is_oa": True,
            },
        ]

        for j in journals:
            normalizer._add_journal(j)

        normalizer._loaded = True
        normalizer._last_updated = time.time()  # Set update time
        normalizer._save_to_cache()  # Save to cache so _is_cache_valid() returns True
        return normalizer

    def test_get_issn_l_by_name(self, loaded_normalizer):
        """Should find ISSN-L by canonical name."""
        # Skip ensure_loaded by setting _loaded = True
        assert loaded_normalizer.get_issn_l("Nature") == "0028-0836"

    def test_get_issn_l_by_issn(self, loaded_normalizer):
        """Should find ISSN-L by any ISSN."""
        assert loaded_normalizer.get_issn_l("1476-4687") == "0028-0836"  # Print ISSN
        assert loaded_normalizer.get_issn_l("0028-0836") == "0028-0836"  # ISSN-L itself

    def test_get_issn_l_by_abbreviation(self, loaded_normalizer):
        """Should find ISSN-L by abbreviated title."""
        assert loaded_normalizer.get_issn_l("J. Neurosci.") == "0270-6474"

    def test_get_issn_l_case_insensitive(self, loaded_normalizer):
        """Should be case insensitive."""
        assert loaded_normalizer.get_issn_l("NATURE") == "0028-0836"
        assert loaded_normalizer.get_issn_l("plos one") == "1932-6203"

    def test_normalize_returns_canonical(self, loaded_normalizer):
        """normalize() should return canonical name."""
        assert loaded_normalizer.normalize("J. Neurosci.") == "Journal of Neuroscience"
        assert loaded_normalizer.normalize("plos one") == "PLOS ONE"

    def test_normalize_unknown_returns_original(self, loaded_normalizer):
        """normalize() should return original for unknown journals."""
        assert (
            loaded_normalizer.normalize("Unknown Journal XYZ") == "Unknown Journal XYZ"
        )

    def test_is_open_access(self, loaded_normalizer):
        """is_open_access() should correctly identify OA journals."""
        assert loaded_normalizer.is_open_access("PLOS ONE") is True
        assert loaded_normalizer.is_open_access("Nature") is False

    def test_is_same_journal_true(self, loaded_normalizer):
        """is_same_journal() should return True for same journal."""
        with patch.object(
            JournalNormalizer, "get_instance", return_value=loaded_normalizer
        ):
            # Different ISSNs for same journal
            assert is_same_journal("Nature", "Nature") is True

    def test_get_abbreviation(self, loaded_normalizer):
        """get_abbreviation() should return abbreviated title."""
        assert (
            loaded_normalizer.get_abbreviation("Journal of Neuroscience")
            == "J. Neurosci."
        )
        assert loaded_normalizer.get_abbreviation("PLOS ONE") == "PLoS One"


class TestJournalNormalizerCache:
    """Tests for JournalNormalizer cache operations."""

    @pytest.fixture
    def normalizer(self, tmp_path):
        """Create a normalizer with test data."""
        normalizer = JournalNormalizer(cache_dir=tmp_path)
        normalizer._add_journal(
            {
                "issn_l": "0028-0836",
                "display_name": "Nature",
                "issn": ["0028-0836"],
            }
        )
        normalizer._loaded = True
        normalizer._last_updated = time.time()
        return normalizer

    def test_save_to_cache(self, normalizer, tmp_path):
        """Should save cache to file."""
        normalizer._save_to_cache()

        cache_file = tmp_path / "journal_normalizer_cache.json"
        assert cache_file.exists()

        with open(cache_file) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "issn_l_data" in data
        assert "0028-0836" in data["issn_l_data"]

    def test_load_from_cache(self, tmp_path):
        """Should load cache from file."""
        # Create a cache file
        cache_data = {
            "timestamp": time.time(),
            "journal_count": 1,
            "issn_l_data": {"0028-0836": {"canonical_name": "Nature", "is_oa": False}},
            "name_to_issn_l": {"nature": "0028-0836"},
            "issn_to_issn_l": {"0028-0836": "0028-0836"},
            "abbrev_to_issn_l": {},
        }

        cache_file = tmp_path / "journal_normalizer_cache.json"
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        # Create fresh normalizer and load
        normalizer = JournalNormalizer(cache_dir=tmp_path)
        success = normalizer._load_from_cache()

        assert success is True
        assert normalizer._loaded is True
        assert "0028-0836" in normalizer._issn_l_data

    def test_cache_validity_check(self, tmp_path):
        """Should check cache TTL validity."""
        # Create an old cache
        cache_data = {
            "timestamp": time.time() - 100000,  # Old timestamp
            "issn_l_data": {},
            "name_to_issn_l": {},
            "issn_to_issn_l": {},
            "abbrev_to_issn_l": {},
        }

        cache_file = tmp_path / "journal_normalizer_cache.json"
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)

        normalizer = JournalNormalizer(cache_dir=tmp_path)
        assert normalizer._is_cache_valid() is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_journal_normalizer_singleton(self):
        """get_journal_normalizer should return singleton."""
        JournalNormalizer._instance = None  # Reset

        norm1 = get_journal_normalizer()
        norm2 = get_journal_normalizer()
        assert norm1 is norm2

        JournalNormalizer._instance = None  # Clean up

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/core/journal_normalizer.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
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
# import asyncio
# import json
# import os
# import re
# import time
# from pathlib import Path
# from typing import Any, Dict, List, Optional
# 
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
#     _instance: Optional[JournalNormalizer] = None
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
#     def get_instance(cls, cache_dir: Optional[Path] = None) -> JournalNormalizer:
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
#             with open(self._cache_file) as f:
#                 data = json.load(f)
#             cached_time = data.get("timestamp", 0)
#             return (time.time() - cached_time) < CACHE_TTL_SECONDS
#         except (OSError, json.JSONDecodeError):
#             return False
# 
#     def _load_from_cache(self) -> bool:
#         """Load cached data from file."""
#         if not self._cache_file.exists():
#             return False
#         try:
#             with open(self._cache_file) as f:
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
#         except (OSError, json.JSONDecodeError) as e:
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
#         except OSError as e:
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
#         """Synchronous wrapper for fetching journals (handles nested event loops)."""
#         import concurrent.futures
# 
#         try:
#             asyncio.get_running_loop()
#             # Already in async context - use thread to avoid nested loop error
#             with concurrent.futures.ThreadPoolExecutor() as executor:
#                 future = executor.submit(
#                     asyncio.run, self._fetch_journals_async(max_pages, filter_oa_only)
#                 )
#                 future.result(timeout=120)
#         except RuntimeError:
#             # No running loop - safe to run directly
#             asyncio.run(self._fetch_journals_async(max_pages, filter_oa_only))
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
#         Returns
#         -------
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
#         Returns
#         -------
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
#         Returns
#         -------
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
#         Returns
#         -------
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
#         Returns
#         -------
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
#         Returns
#         -------
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
#         Returns
#         -------
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
