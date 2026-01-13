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

    pytest.main([os.path.abspath(__file__), "-v"])
