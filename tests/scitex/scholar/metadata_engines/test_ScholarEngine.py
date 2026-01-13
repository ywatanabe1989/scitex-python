#!/usr/bin/env python3
# Timestamp: "2026-01-14"
# File: tests/scitex/scholar/metadata_engines/test_ScholarEngine.py
"""
Comprehensive tests for ScholarEngine metadata aggregator.

Tests cover:
- Initialization and configuration
- Cache operations (setup, load, save, key generation)
- Search functionality (async, batch)
- Engine management
- Metadata validation and combination
- Metadata merging with engine priority
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.scholar.metadata_engines import ScholarEngine


class TestScholarEngineInit:
    """Tests for ScholarEngine initialization."""

    def test_init_default_values(self, tmp_path):
        """Engine should initialize with default values."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: default
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            engine = ScholarEngine()

            assert engine.name == "ScholarEngine"
            assert engine.use_cache is True
            assert engine._engine_instances == {}

    def test_init_custom_engines(self, tmp_path):
        """Engine should accept custom engine list."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: (
                ["CrossRef", "PubMed"] if key == "engines" else default
            )
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            engine = ScholarEngine(engines=["CrossRef", "PubMed"])

            assert engine.engines == ["CrossRef", "PubMed"]

    def test_init_cache_disabled(self, tmp_path):
        """Engine should work with cache disabled."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: (
                False if key == "use_cache_engine" else default
            )
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            engine = ScholarEngine(use_cache=False)

            assert engine.use_cache is False


class TestScholarEngineCache:
    """Tests for cache operations."""

    @pytest.fixture
    def engine_with_cache(self, tmp_path):
        """Create engine with cache directory."""
        with patch(
            "scitex.scholar.metadata_engines.ScholarEngine.ScholarConfig"
        ) as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: default
            mock_path_manager = MagicMock()
            cache_dir = tmp_path / "cache"
            mock_path_manager.get_cache_engine_dir.return_value = cache_dir
            mock_config.return_value.path_manager = mock_path_manager

            engine = ScholarEngine()
            yield engine, cache_dir

    def test_cache_directory_created(self, engine_with_cache):
        """Cache directory should be created on init."""
        engine, cache_dir = engine_with_cache
        assert cache_dir.exists()

    def test_get_cache_key_deterministic(self, engine_with_cache):
        """Same parameters should produce same cache key."""
        engine, _ = engine_with_cache

        key1 = engine._get_cache_key(title="Test Paper")
        key2 = engine._get_cache_key(title="Test Paper")

        assert key1 == key2

    def test_get_cache_key_unique(self, engine_with_cache):
        """Different parameters should produce different keys."""
        engine, _ = engine_with_cache

        key1 = engine._get_cache_key(title="Test Paper")
        key2 = engine._get_cache_key(doi="10.1038/test")

        assert key1 != key2

    def test_get_cache_key_ignores_none(self, engine_with_cache):
        """None values should be ignored in cache key."""
        engine, _ = engine_with_cache

        key1 = engine._get_cache_key(title="Test", doi=None)
        key2 = engine._get_cache_key(title="Test")

        assert key1 == key2

    def test_save_and_load_cache(self, engine_with_cache):
        """Cache should persist between saves and loads."""
        engine, cache_dir = engine_with_cache

        engine._cache = {"test_key": {"id": {"doi": "10.1038/test"}}}
        engine._save_cache()

        # Reload cache
        engine._cache = {}
        engine._load_cache()

        assert "test_key" in engine._cache
        assert engine._cache["test_key"]["id"]["doi"] == "10.1038/test"

    def test_clear_cache(self, tmp_path):
        """Cache should be cleared when clear_cache=True."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: default
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            # Create engine with cache
            engine1 = ScholarEngine()
            engine1._cache = {"test": "data"}
            engine1._save_cache()

            # Create new engine with clear_cache
            engine2 = ScholarEngine(clear_cache=True)

            assert engine2._cache == {}


class TestScholarEngineSearch:
    """Tests for search functionality."""

    @pytest.fixture
    def mock_engine(self, tmp_path):
        """Create engine with mocked dependencies."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: (
                ["CrossRef"] if key == "engines" else default
            )
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            engine = ScholarEngine(engines=["CrossRef"], use_cache=False)
            return engine

    @pytest.mark.asyncio
    async def test_search_async_returns_cached(self, tmp_path):
        """Should return cached result when available."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: default
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            engine = ScholarEngine(use_cache=True)
            cache_key = engine._get_cache_key(doi="10.1038/test")
            engine._cache[cache_key] = {"id": {"doi": "10.1038/test"}}

            result = await engine.search_async(doi="10.1038/test")

            assert result["id"]["doi"] == "10.1038/test"

    @pytest.mark.asyncio
    async def test_search_async_calls_engines(self, mock_engine):
        """Should call engines when no cache hit."""
        mock_crossref = MagicMock()
        mock_crossref.search.return_value = {
            "id": {"doi": "10.1038/test"},
            "basic": {"title": "Test Paper", "year": 2023},
        }

        with patch.object(mock_engine, "_get_engine", return_value=mock_crossref):
            result = await mock_engine.search_async(doi="10.1038/test")

            assert result is not None

    @pytest.mark.asyncio
    async def test_search_async_handles_exception(self, mock_engine):
        """Should handle engine exceptions gracefully."""

        async def failing_search(*args, **kwargs):
            raise Exception("Network error")

        mock_engine._search_engine_with_timeout = failing_search

        # Should not raise, returns None/empty
        result = await mock_engine.search_async(doi="10.1038/test")
        # Empty result or None is acceptable for exception


class TestScholarEngineValidation:
    """Tests for metadata validation."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine instance."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: default
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            return ScholarEngine()

    def test_extract_identifiers_doi(self, engine):
        """Should extract DOI correctly."""
        metadata = {"id": {"doi": "10.1038/nature12373"}}

        ids = engine._extract_identifiers(metadata)

        assert ids["doi"] == "10.1038/nature12373"

    def test_extract_identifiers_cleans_url(self, engine):
        """Should clean DOI from URL format."""
        metadata = {"id": {"doi": "https://doi.org/10.1038/nature12373"}}

        ids = engine._extract_identifiers(metadata)

        assert "10.1038" in ids["doi"]
        assert "nature12373" in ids["doi"]

    def test_identifiers_match_same_doi(self, engine):
        """Should match when DOIs are same."""
        ids1 = {"doi": "10.1038/nature12373"}
        ids2 = {"doi": "10.1038/nature12373"}

        assert engine._identifiers_match(ids1, ids2) is True

    def test_identifiers_match_different(self, engine):
        """Should not match when no common identifiers."""
        ids1 = {"doi": "10.1038/nature12373"}
        ids2 = {"pmid": "12345678"}

        assert engine._identifiers_match(ids1, ids2) is False

    def test_validate_paper_consistency_same(self, engine):
        """Should validate consistent metadata."""
        metadata_list = [
            {
                "basic": {
                    "title": "Test Paper",
                    "year": 2023,
                    "authors": ["Smith, John"],
                }
            },
            {
                "basic": {
                    "title": "Test Paper",
                    "year": 2023,
                    "authors": ["Smith, John"],
                }
            },
        ]

        assert engine._validate_paper_consistency(metadata_list) is True

    def test_validate_paper_consistency_different_year(self, engine):
        """Should fail when years differ."""
        metadata_list = [
            {
                "basic": {
                    "title": "Test Paper",
                    "year": 2023,
                    "authors": ["Smith, John"],
                }
            },
            {
                "basic": {
                    "title": "Test Paper",
                    "year": 2022,
                    "authors": ["Smith, John"],
                }
            },
        ]

        assert engine._validate_paper_consistency(metadata_list) is False

    def test_validate_against_query_exact(self, engine):
        """Should validate exact title match."""
        metadata = {"basic": {"title": "Attention Is All You Need"}}

        assert (
            engine._validate_against_query(metadata, "Attention Is All You Need")
            is True
        )

    def test_validate_against_query_partial(self, engine):
        """Should validate partial title match."""
        metadata = {"basic": {"title": "Attention Is All You Need"}}

        assert engine._validate_against_query(metadata, "Attention Is All You") is True

    def test_validate_against_query_unrelated(self, engine):
        """Should reject unrelated titles."""
        metadata = {"basic": {"title": "Deep Learning for Image Classification"}}

        assert (
            engine._validate_against_query(metadata, "Attention Is All You Need")
            is False
        )


class TestScholarEngineMetadataMerging:
    """Tests for metadata combination and merging."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine instance."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: (
                ["CrossRef", "PubMed"] if key == "engines" else default
            )
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            return ScholarEngine()

    def test_combine_metadata_single_engine(self, engine):
        """Should return metadata from single engine."""
        engine_results = {
            "CrossRef": {"id": {"doi": "10.1038/test"}, "basic": {"title": "Test"}}
        }

        result = engine._combine_metadata(engine_results)

        assert result["id"]["doi"] == "10.1038/test"

    def test_combine_metadata_empty(self, engine):
        """Should return None for empty results."""
        result = engine._combine_metadata({})

        assert result is None

    def test_merge_metadata_structures_priority(self, engine):
        """Higher priority engine should win on conflict."""
        base = {
            "id": {"doi": "10.1038/test", "doi_engines": ["PubMed"]},
        }
        additional = {
            "id": {"doi": "10.1038/test", "doi_engines": ["CrossRef"]},
        }

        result = engine._merge_metadata_structures(base, additional)

        # CrossRef has higher priority than PubMed
        assert "CrossRef" in result["id"]["doi_engines"]

    def test_merge_metadata_structures_fills_missing(self, engine):
        """Should fill in missing fields."""
        base = {"id": {"doi": "10.1038/test"}}
        additional = {
            "id": {"doi": "10.1038/test"},
            "basic": {"title": "Test Paper", "title_engines": ["CrossRef"]},
        }

        result = engine._merge_metadata_structures(base, additional)

        assert result["basic"]["title"] == "Test Paper"

    def test_merge_metadata_structures_longer_value_wins(self, engine):
        """Longer value should win when priorities equal."""
        base = {
            "basic": {
                "title": "Short Title",
                "title_engines": ["arXiv"],
            }
        }
        additional = {
            "basic": {
                "title": "A Much Longer and More Complete Title",
                "title_engines": ["arXiv"],
            }
        }

        result = engine._merge_metadata_structures(base, additional)

        assert "Much Longer" in result["basic"]["title"]


class TestScholarEngineGetEngine:
    """Tests for engine instantiation."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine instance."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: default
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            return ScholarEngine()

    def test_get_engine_returns_cached(self, engine):
        """Should return cached instance on subsequent calls."""
        mock_instance = MagicMock()
        engine._engine_instances["CrossRef"] = mock_instance

        result = engine._get_engine("CrossRef")

        assert result is mock_instance

    def test_get_engine_unknown_returns_none(self, engine):
        """Should return None for unknown engine."""
        result = engine._get_engine("UnknownEngine")

        assert result is None


class TestScholarEngineBatchSearch:
    """Tests for batch search functionality."""

    @pytest.fixture
    def engine(self, tmp_path):
        """Create engine instance."""
        with patch("scitex.scholar.ScholarConfig") as mock_config:
            mock_config.return_value.resolve.side_effect = lambda key, default: default
            mock_path_manager = MagicMock()
            mock_path_manager.get_cache_engine_dir.return_value = tmp_path / "cache"
            mock_config.return_value.path_manager = mock_path_manager

            return ScholarEngine(use_cache=False)

    @pytest.mark.asyncio
    async def test_search_batch_empty_returns_empty(self, engine):
        """Should return empty list for no inputs."""
        result = await engine.search_batch_async()

        assert result == []

    @pytest.mark.asyncio
    async def test_search_batch_dois(self, engine):
        """Should search batch of DOIs."""
        engine.search_async = AsyncMock(return_value={"id": {"doi": "10.1038/test"}})

        result = await engine.search_batch_async(
            dois=["10.1038/test1", "10.1038/test2"]
        )

        assert len(result) == 2
        assert engine.search_async.call_count == 2

    @pytest.mark.asyncio
    async def test_search_batch_titles(self, engine):
        """Should search batch of titles."""
        engine.search_async = AsyncMock(return_value={"basic": {"title": "Test Paper"}})

        result = await engine.search_batch_async(titles=["Title 1", "Title 2"])

        assert len(result) == 2


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
