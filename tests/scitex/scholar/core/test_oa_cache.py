#!/usr/bin/env python3
"""Tests for OASourcesCache - Open Access sources caching functionality."""

import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scitex.scholar.core.oa_cache import (
    CACHE_TTL_SECONDS,
    OPENALEX_OA_SOURCES_URL,
    OPENALEX_POLITE_EMAIL,
    OASourcesCache,
    _get_default_cache_dir,
    get_oa_cache,
    is_oa_journal_cached,
    refresh_oa_cache,
)


class TestConstants:
    """Tests for module-level constants."""

    def test_cache_ttl_is_one_day(self):
        """Cache TTL should be 24 hours (86400 seconds)."""
        assert CACHE_TTL_SECONDS == 86400

    def test_openalex_url_is_valid(self):
        """OpenAlex URL should be properly formatted."""
        assert OPENALEX_OA_SOURCES_URL == "https://api.openalex.org/sources"

    def test_polite_email_is_set(self):
        """Polite email should be set for OpenAlex API."""
        assert OPENALEX_POLITE_EMAIL == "research@scitex.io"


class TestGetDefaultCacheDir:
    """Tests for _get_default_cache_dir function."""

    def test_default_cache_dir_without_env(self, monkeypatch):
        """Should return ~/.scitex/scholar/cache when SCITEX_DIR not set."""
        monkeypatch.delenv("SCITEX_DIR", raising=False)
        cache_dir = _get_default_cache_dir()
        expected = Path.home() / ".scitex" / "scholar" / "cache"
        assert cache_dir == expected

    def test_default_cache_dir_with_env(self, monkeypatch):
        """Should use SCITEX_DIR env var when set."""
        monkeypatch.setenv("SCITEX_DIR", "/custom/scitex")
        cache_dir = _get_default_cache_dir()
        expected = Path("/custom/scitex/scholar/cache")
        assert cache_dir == expected

    def test_default_cache_dir_with_tilde(self, monkeypatch):
        """Should expand tilde in SCITEX_DIR."""
        monkeypatch.setenv("SCITEX_DIR", "~/custom_scitex")
        cache_dir = _get_default_cache_dir()
        assert str(cache_dir).startswith(str(Path.home()))
        assert "custom_scitex" in str(cache_dir)


class TestOASourcesCacheInit:
    """Tests for OASourcesCache initialization."""

    def test_init_with_default_cache_dir(self, tmp_path, monkeypatch):
        """Should use default cache directory when none provided."""
        monkeypatch.setenv("SCITEX_DIR", str(tmp_path))
        # Reset singleton
        OASourcesCache._instance = None
        cache = OASourcesCache()
        assert cache._cache_dir == tmp_path / "scholar" / "cache"

    def test_init_with_custom_cache_dir(self, tmp_path):
        """Should use provided cache directory."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        assert cache._cache_dir == tmp_path

    def test_init_sets_cache_file_path(self, tmp_path):
        """Should set cache file path correctly."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        assert cache._cache_file == tmp_path / "oa_sources_cache.json"

    def test_init_empty_sets(self, tmp_path):
        """Should initialize empty sets and dicts."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        assert cache._oa_source_ids == set()
        assert cache._oa_source_names == set()
        assert cache._oa_issns == set()
        assert cache._issn_l_map == {}
        assert cache._name_to_issn_l == {}
        assert cache._issn_l_to_canonical == {}

    def test_init_timestamps(self, tmp_path):
        """Should initialize timestamps to default values."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        assert cache._last_updated == 0
        assert cache._loaded is False


class TestOASourcesCacheSingleton:
    """Tests for singleton pattern."""

    def test_get_instance_creates_singleton(self, tmp_path):
        """Should create singleton on first call."""
        OASourcesCache._instance = None
        instance1 = OASourcesCache.get_instance(cache_dir=tmp_path)
        instance2 = OASourcesCache.get_instance()
        assert instance1 is instance2

    def test_get_instance_reuses_existing(self, tmp_path):
        """Should reuse existing singleton instance."""
        OASourcesCache._instance = None
        instance1 = OASourcesCache.get_instance(cache_dir=tmp_path)
        # Second call with different path should still return same instance
        instance2 = OASourcesCache.get_instance(cache_dir=tmp_path / "different")
        assert instance1 is instance2


class TestIsCacheValid:
    """Tests for _is_cache_valid method."""

    def test_invalid_when_file_not_exists(self, tmp_path):
        """Should return False when cache file doesn't exist."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        assert cache._is_cache_valid() is False

    def test_invalid_when_expired(self, tmp_path):
        """Should return False when cache is expired."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write cache with old timestamp
        old_time = time.time() - CACHE_TTL_SECONDS - 1
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": old_time, "source_names": []}, f)

        assert cache._is_cache_valid() is False

    def test_valid_when_fresh(self, tmp_path):
        """Should return True when cache is within TTL."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write cache with recent timestamp
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "source_names": []}, f)

        assert cache._is_cache_valid() is True

    def test_invalid_on_json_decode_error(self, tmp_path):
        """Should return False on JSON decode error."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write invalid JSON
        with open(cache._cache_file, "w") as f:
            f.write("not valid json {{{")

        assert cache._is_cache_valid() is False

    def test_invalid_when_timestamp_missing(self, tmp_path):
        """Should handle missing timestamp in cache."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write cache without timestamp (defaults to 0)
        with open(cache._cache_file, "w") as f:
            json.dump({"source_names": []}, f)

        # timestamp=0 means cache is very old
        assert cache._is_cache_valid() is False


class TestLoadFromCache:
    """Tests for _load_from_cache method."""

    def test_returns_false_when_file_not_exists(self, tmp_path):
        """Should return False when cache file doesn't exist."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        assert cache._load_from_cache() is False

    def test_loads_source_names(self, tmp_path):
        """Should load source names from cache file."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        source_names = ["journal one", "journal two", "plos one"]
        with open(cache._cache_file, "w") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "source_names": source_names,
                    "issns": ["1234-5678"],
                },
                f,
            )

        result = cache._load_from_cache()
        assert result is True
        assert cache._oa_source_names == set(source_names)

    def test_loads_issns(self, tmp_path):
        """Should load ISSNs from cache file."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        issns = ["1234-5678", "8765-4321"]
        with open(cache._cache_file, "w") as f:
            json.dump(
                {
                    "timestamp": time.time(),
                    "source_names": [],
                    "issns": issns,
                },
                f,
            )

        cache._load_from_cache()
        assert cache._oa_issns == set(issns)

    def test_sets_loaded_flag(self, tmp_path):
        """Should set _loaded flag to True on successful load."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "source_names": []}, f)

        assert cache._loaded is False
        cache._load_from_cache()
        assert cache._loaded is True

    def test_returns_false_on_json_error(self, tmp_path):
        """Should return False and log warning on JSON error."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache._cache_file, "w") as f:
            f.write("invalid json")

        result = cache._load_from_cache()
        assert result is False

    def test_handles_missing_keys(self, tmp_path):
        """Should handle missing keys in cache data."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache._cache_file, "w") as f:
            json.dump({}, f)  # Empty cache data

        cache._load_from_cache()
        assert cache._oa_source_names == set()
        assert cache._oa_issns == set()
        assert cache._last_updated == 0


class TestSaveToCache:
    """Tests for _save_to_cache method."""

    def test_creates_cache_directory(self, tmp_path):
        """Should create cache directory if not exists."""
        OASourcesCache._instance = None
        cache_dir = tmp_path / "nested" / "cache" / "dir"
        cache = OASourcesCache(cache_dir=cache_dir)
        cache._oa_source_names = {"journal one"}

        cache._save_to_cache()
        assert cache_dir.exists()

    def test_saves_source_names(self, tmp_path):
        """Should save source names to cache file."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"journal one", "journal two"}
        cache._oa_issns = {"1234-5678"}

        cache._save_to_cache()

        with open(cache._cache_file) as f:
            data = json.load(f)

        assert set(data["source_names"]) == {"journal one", "journal two"}

    def test_saves_issns(self, tmp_path):
        """Should save ISSNs to cache file."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"journal one"}
        cache._oa_issns = {"1234-5678", "8765-4321"}

        cache._save_to_cache()

        with open(cache._cache_file) as f:
            data = json.load(f)

        assert set(data["issns"]) == {"1234-5678", "8765-4321"}

    def test_saves_timestamp(self, tmp_path):
        """Should save current timestamp."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"journal"}

        before = time.time()
        cache._save_to_cache()
        after = time.time()

        with open(cache._cache_file) as f:
            data = json.load(f)

        assert before <= data["timestamp"] <= after

    def test_saves_count(self, tmp_path):
        """Should save source count."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"journal one", "journal two", "journal three"}
        cache._oa_issns = set()

        cache._save_to_cache()

        with open(cache._cache_file) as f:
            data = json.load(f)

        assert data["count"] == 3


class TestFetchOASourcesAsync:
    """Tests for _fetch_oa_sources_async method."""

    @pytest.mark.asyncio
    async def test_fetches_sources_from_api(self, tmp_path):
        """Should fetch and store OA sources from API."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        mock_response = {
            "results": [
                {"display_name": "PLOS ONE", "issn": ["1932-6203"]},
                {"display_name": "Nature Communications", "issn": ["2041-1723"]},
            ],
            "meta": {"next_cursor": None},
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response_obj = MagicMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)

            # session.get() returns a context manager directly (not a coroutine)
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm

            # ClientSession() returns a context manager
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_cm

            await cache._fetch_oa_sources_async(max_pages=1)

        assert "plos one" in cache._oa_source_names
        assert "nature communications" in cache._oa_source_names
        assert "1932-6203" in cache._oa_issns

    @pytest.mark.asyncio
    async def test_handles_empty_results(self, tmp_path):
        """Should handle empty results gracefully."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"existing"}

        mock_response = {
            "results": [],
            "meta": {},
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response_obj = MagicMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)

            # session.get() returns a context manager directly (not a coroutine)
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm

            # ClientSession() returns a context manager
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_cm

            await cache._fetch_oa_sources_async(max_pages=1)

        # Should not update if no results
        assert cache._oa_source_names == {"existing"}

    @pytest.mark.asyncio
    async def test_handles_api_error(self, tmp_path):
        """Should handle API errors gracefully."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response_obj = MagicMock()
            mock_response_obj.status = 500

            # session.get() returns a context manager directly
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm

            # ClientSession() returns a context manager
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_cm

            await cache._fetch_oa_sources_async(max_pages=1)

        # Should not crash, just log warning
        assert cache._oa_source_names == set()

    @pytest.mark.asyncio
    async def test_handles_timeout(self, tmp_path):
        """Should handle timeout errors."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        with patch("aiohttp.ClientSession") as mock_session_class:
            # session.get() returns a context manager that raises TimeoutError
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm

            # ClientSession() returns a context manager
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_cm

            await cache._fetch_oa_sources_async(max_pages=1)

        # Should not crash
        assert cache._loaded is False

    @pytest.mark.asyncio
    async def test_handles_null_issns(self, tmp_path):
        """Should handle null ISSN values."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        mock_response = {
            "results": [
                {"display_name": "Journal One", "issn": None},
                {"display_name": "Journal Two", "issn": [None, "1234-5678"]},
            ],
            "meta": {"next_cursor": None},
        }

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response_obj = MagicMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(return_value=mock_response)

            # session.get() returns a context manager directly
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm

            # ClientSession() returns a context manager
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_cm

            await cache._fetch_oa_sources_async(max_pages=1)

        assert "journal one" in cache._oa_source_names
        assert "1234-5678" in cache._oa_issns

    @pytest.mark.asyncio
    async def test_pagination(self, tmp_path):
        """Should handle pagination correctly."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        responses = [
            {
                "results": [{"display_name": "Journal A", "issn": []}],
                "meta": {"next_cursor": "cursor2"},
            },
            {
                "results": [{"display_name": "Journal B", "issn": []}],
                "meta": {"next_cursor": None},
            },
        ]
        response_iter = iter(responses)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_response_obj = MagicMock()
            mock_response_obj.status = 200
            mock_response_obj.json = AsyncMock(side_effect=lambda: next(response_iter))

            # session.get() returns a context manager directly
            mock_get_cm = MagicMock()
            mock_get_cm.__aenter__ = AsyncMock(return_value=mock_response_obj)
            mock_get_cm.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.get.return_value = mock_get_cm

            # ClientSession() returns a context manager
            mock_session_cm = MagicMock()
            mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_cm.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session_cm

            await cache._fetch_oa_sources_async(max_pages=10)

        assert "journal a" in cache._oa_source_names
        assert "journal b" in cache._oa_source_names


class TestFetchOASourcesSync:
    """Tests for _fetch_oa_sources_sync method."""

    def test_calls_async_method(self, tmp_path):
        """Should call async method synchronously."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        with patch.object(
            cache, "_fetch_oa_sources_async", new_callable=AsyncMock
        ) as mock_async:
            cache._fetch_oa_sources_sync(max_pages=5)
            mock_async.assert_called_once_with(5)


class TestEnsureLoaded:
    """Tests for ensure_loaded method."""

    def test_does_nothing_if_already_loaded_and_valid(self, tmp_path):
        """Should skip loading if already loaded and cache valid."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._loaded = True
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write valid cache
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "source_names": []}, f)

        with patch.object(cache, "_load_from_cache") as mock_load:
            cache.ensure_loaded()
            mock_load.assert_not_called()

    def test_loads_from_cache_if_valid(self, tmp_path):
        """Should load from cache if cache is valid."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write valid cache
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "source_names": ["journal"]}, f)

        cache.ensure_loaded()
        assert "journal" in cache._oa_source_names

    def test_fetches_from_api_if_cache_invalid(self, tmp_path):
        """Should fetch from API if cache is invalid."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        with patch.object(cache, "_fetch_oa_sources_sync") as mock_fetch:
            cache.ensure_loaded()
            mock_fetch.assert_called_once()

    def test_force_refresh_fetches_from_api(self, tmp_path):
        """Should fetch from API when force_refresh is True."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._loaded = True
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write valid cache
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "source_names": []}, f)

        with patch.object(cache, "_fetch_oa_sources_sync") as mock_fetch:
            cache.ensure_loaded(force_refresh=True)
            mock_fetch.assert_called_once()


class TestIsOASource:
    """Tests for is_oa_source method."""

    def test_returns_true_for_oa_journal(self, tmp_path):
        """Should return True for OA journal."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"plos one", "nature communications"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_source("PLOS ONE") is True
        assert cache.is_oa_source("Nature Communications") is True

    def test_returns_false_for_non_oa_journal(self, tmp_path):
        """Should return False for non-OA journal."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"plos one"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_source("Nature") is False

    def test_case_insensitive(self, tmp_path):
        """Should be case insensitive."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"plos one"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_source("PLOS ONE") is True
        assert cache.is_oa_source("plos one") is True
        assert cache.is_oa_source("Plos One") is True

    def test_returns_false_for_empty_string(self, tmp_path):
        """Should return False for empty string."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"plos one"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_source("") is False

    def test_returns_false_for_none(self, tmp_path):
        """Should return False for None."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"plos one"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_source(None) is False

    def test_ensures_loaded(self, tmp_path):
        """Should ensure cache is loaded before checking."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        with patch.object(cache, "ensure_loaded") as mock_ensure:
            cache._oa_source_names = set()
            cache.is_oa_source("test")
            mock_ensure.assert_called_once()


class TestIsOAIssn:
    """Tests for is_oa_issn method."""

    def test_returns_true_for_oa_issn(self, tmp_path):
        """Should return True for OA ISSN."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_issns = {"1234-5678", "8765-4321"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_issn("1234-5678") is True

    def test_returns_false_for_non_oa_issn(self, tmp_path):
        """Should return False for non-OA ISSN."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_issns = {"1234-5678"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_issn("9999-9999") is False

    def test_normalizes_issn_without_hyphen(self, tmp_path):
        """Should handle ISSN without hyphen."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_issns = {"1234-5678"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_issn("12345678") is True

    def test_normalizes_issn_lowercase(self, tmp_path):
        """Should handle lowercase ISSN."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_issns = {"1234-567X"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_issn("1234-567x") is True

    def test_returns_false_for_empty_string(self, tmp_path):
        """Should return False for empty ISSN."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_issns = {"1234-5678"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_issn("") is False

    def test_returns_false_for_none(self, tmp_path):
        """Should return False for None ISSN."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_issns = {"1234-5678"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_issn(None) is False


class TestSourceCount:
    """Tests for source_count property."""

    def test_returns_count(self, tmp_path):
        """Should return number of cached sources."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"journal one", "journal two", "journal three"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.source_count == 3

    def test_ensures_loaded(self, tmp_path):
        """Should ensure cache is loaded."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        with patch.object(cache, "ensure_loaded") as mock_ensure:
            cache._oa_source_names = set()
            _ = cache.source_count
            mock_ensure.assert_called_once()


class TestCacheAgeHours:
    """Tests for cache_age_hours property."""

    def test_returns_infinity_if_never_updated(self, tmp_path):
        """Should return infinity if never updated."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._last_updated = 0

        assert cache.cache_age_hours == float("inf")

    def test_returns_hours_since_update(self, tmp_path):
        """Should return hours since last update."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._last_updated = time.time() - 7200  # 2 hours ago

        age = cache.cache_age_hours
        assert 1.9 < age < 2.1  # Allow some tolerance


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_oa_cache_returns_singleton(self, tmp_path):
        """get_oa_cache should return singleton instance."""
        OASourcesCache._instance = None
        cache1 = get_oa_cache(cache_dir=tmp_path)
        cache2 = get_oa_cache()
        assert cache1 is cache2

    def test_is_oa_journal_cached(self, tmp_path):
        """is_oa_journal_cached should check OA status."""
        OASourcesCache._instance = None
        cache = OASourcesCache.get_instance(cache_dir=tmp_path)
        cache._oa_source_names = {"plos one"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert is_oa_journal_cached("PLOS ONE") is True
        assert is_oa_journal_cached("Nature") is False

    def test_refresh_oa_cache(self, tmp_path):
        """refresh_oa_cache should force refresh."""
        OASourcesCache._instance = None
        cache = OASourcesCache.get_instance(cache_dir=tmp_path)

        with patch.object(cache, "ensure_loaded") as mock_ensure:
            refresh_oa_cache()
            mock_ensure.assert_called_once_with(force_refresh=True)


class TestOASourcesCacheEdgeCases:
    """Edge case tests for OASourcesCache."""

    def test_handles_unicode_journal_names(self, tmp_path):
        """Should handle unicode characters in journal names."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"日本語ジャーナル", "révue française"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_source("日本語ジャーナル") is True
        assert cache.is_oa_source("Révue Française") is True

    def test_handles_special_characters_in_names(self, tmp_path):
        """Should handle special characters in journal names."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"journal (online)", "bio-medical & health"}
        cache._loaded = True
        cache._last_updated = time.time()
        cache._cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache._cache_file, "w") as f:
            json.dump({"timestamp": time.time()}, f)

        assert cache.is_oa_source("Journal (Online)") is True
        assert cache.is_oa_source("Bio-Medical & Health") is True

    def test_concurrent_singleton_access(self, tmp_path):
        """Singleton should be consistent across accesses."""
        OASourcesCache._instance = None

        instances = []
        for _ in range(10):
            instances.append(OASourcesCache.get_instance(cache_dir=tmp_path))

        # All instances should be the same object
        assert all(inst is instances[0] for inst in instances)

    def test_cache_file_permissions_error(self, tmp_path):
        """Should handle permission errors gracefully."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._oa_source_names = {"journal"}

        with patch("builtins.open", side_effect=OSError("Permission denied")):
            # Should not raise, just log warning
            cache._save_to_cache()

    def test_very_large_cache(self, tmp_path):
        """Should handle large number of sources."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)

        # Create 10000 journal names
        large_set = {f"journal {i}" for i in range(10000)}
        cache._oa_source_names = large_set
        cache._oa_issns = {f"{i:04d}-{i:04d}" for i in range(1000)}

        cache._save_to_cache()

        # Reset and reload
        cache._oa_source_names = set()
        cache._oa_issns = set()
        cache._loaded = False

        cache._load_from_cache()
        assert len(cache._oa_source_names) == 10000
        assert cache.is_oa_source("journal 5000") is True


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_full_lifecycle(self, tmp_path):
        """Test complete lifecycle: init -> save -> load -> query."""
        OASourcesCache._instance = None

        # Initialize and populate
        cache1 = OASourcesCache(cache_dir=tmp_path)
        cache1._oa_source_names = {"nature", "science", "plos one"}
        cache1._oa_issns = {"1234-5678"}
        cache1._last_updated = time.time()
        cache1._loaded = True
        cache1._save_to_cache()

        # Create new instance (simulating restart)
        OASourcesCache._instance = None
        cache2 = OASourcesCache(cache_dir=tmp_path)

        # Should load from cache
        cache2._load_from_cache()

        assert cache2.is_oa_source("Nature") is True
        assert cache2.is_oa_source("Science") is True
        assert cache2.is_oa_issn("1234-5678") is True

    def test_cache_expiry_triggers_refresh(self, tmp_path):
        """Test that expired cache triggers API refresh."""
        OASourcesCache._instance = None
        cache = OASourcesCache(cache_dir=tmp_path)
        cache._cache_dir.mkdir(parents=True, exist_ok=True)

        # Write expired cache
        old_time = time.time() - CACHE_TTL_SECONDS - 3600
        with open(cache._cache_file, "w") as f:
            json.dump(
                {
                    "timestamp": old_time,
                    "source_names": ["old journal"],
                    "issns": [],
                },
                f,
            )

        with patch.object(cache, "_fetch_oa_sources_sync") as mock_fetch:
            cache.ensure_loaded()
            mock_fetch.assert_called_once()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
