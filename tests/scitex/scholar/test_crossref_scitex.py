#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: tests/scitex/scholar/test_crossref_scitex.py
"""Tests for scitex.scholar.crossref_scitex delegation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCrossrefScitexImport:
    """Test module import behavior."""

    def test_import_crossref_scitex_module(self):
        """Test that crossref_scitex module can be imported."""
        from scitex.scholar import crossref_scitex

        assert crossref_scitex is not None

    def test_module_has_expected_exports(self):
        """Test that module exports expected functions."""
        from scitex.scholar import crossref_scitex

        expected = [
            "search",
            "count",
            "get",
            "get_many",
            "exists",
            "enrich",
            "enrich_dois",
            "configure",
            "configure_http",
            "get_mode",
            "info",
            "is_available",
            "get_citing",
            "get_cited",
            "get_citation_count",
        ]
        for func_name in expected:
            assert hasattr(crossref_scitex, func_name), f"Missing: {func_name}"


class TestEnsureCrossrefLocal:
    """Test _ensure_crossref_local helper."""

    def test_ensure_crossref_local_returns_module_when_installed(self):
        """Test that _ensure_crossref_local returns crossref_local when installed."""
        from scitex.scholar import crossref_scitex

        # This will either work or raise ImportError depending on environment
        try:
            result = crossref_scitex._ensure_crossref_local()
            assert result is not None
        except ImportError:
            pytest.skip("crossref-local not installed")

    def test_ensure_crossref_local_raises_when_not_installed(self):
        """Test that _ensure_crossref_local raises ImportError when not installed."""
        from scitex.scholar import crossref_scitex

        with patch.dict("sys.modules", {"crossref_local": None}):
            with patch("scitex.scholar.crossref_scitex._ensure_crossref_local") as mock:
                mock.side_effect = ImportError("crossref-local not installed")
                with pytest.raises(ImportError, match="crossref-local"):
                    mock()


class TestSearchFunction:
    """Test search function delegation."""

    def test_search_delegates_to_crossref_local(self):
        """Test that search() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_result = MagicMock()
        mock_result.total = 100
        mock_result.__iter__ = lambda self: iter([])
        mock_crl.search.return_value = mock_result

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.search("deep learning", limit=10)

            mock_crl.search.assert_called_once_with("deep learning", limit=10, offset=0)
            assert result == mock_result

    def test_search_passes_offset_parameter(self):
        """Test that search() passes offset parameter."""
        mock_crl = MagicMock()
        mock_crl.search.return_value = MagicMock()

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            crossref_scitex.search("query", limit=20, offset=100)

            mock_crl.search.assert_called_once_with("query", limit=20, offset=100)


class TestCountFunction:
    """Test count function delegation."""

    def test_count_delegates_to_crossref_local(self):
        """Test that count() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_crl.count.return_value = 12345

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.count("machine learning")

            mock_crl.count.assert_called_once_with("machine learning")
            assert result == 12345


class TestGetFunction:
    """Test get function delegation."""

    def test_get_delegates_to_crossref_local(self):
        """Test that get() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_work = MagicMock()
        mock_work.title = "Test Paper"
        mock_crl.get.return_value = mock_work

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.get("10.1038/nature12373")

            mock_crl.get.assert_called_once_with("10.1038/nature12373")
            assert result.title == "Test Paper"

    def test_get_returns_none_for_nonexistent_doi(self):
        """Test that get() returns None for non-existent DOI."""
        mock_crl = MagicMock()
        mock_crl.get.return_value = None

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.get("10.0000/nonexistent")

            assert result is None


class TestGetManyFunction:
    """Test get_many function delegation."""

    def test_get_many_delegates_to_crossref_local(self):
        """Test that get_many() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_works = [MagicMock(), MagicMock()]
        mock_crl.get_many.return_value = mock_works

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            dois = ["10.1038/1", "10.1038/2"]
            result = crossref_scitex.get_many(dois)

            mock_crl.get_many.assert_called_once_with(dois)
            assert len(result) == 2


class TestExistsFunction:
    """Test exists function delegation."""

    def test_exists_delegates_to_crossref_local(self):
        """Test that exists() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_crl.exists.return_value = True

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.exists("10.1038/nature12373")

            mock_crl.exists.assert_called_once_with("10.1038/nature12373")
            assert result is True


class TestCitationFunctions:
    """Test citation-related function delegation."""

    def test_get_citing_delegates_to_crossref_local(self):
        """Test that get_citing() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_crl.get_citing.return_value = ["10.1038/1", "10.1038/2"]

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.get_citing("10.1038/nature12373")

            mock_crl.get_citing.assert_called_once_with("10.1038/nature12373")
            assert result == ["10.1038/1", "10.1038/2"]

    def test_get_cited_delegates_to_crossref_local(self):
        """Test that get_cited() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_crl.get_cited.return_value = ["10.1016/1", "10.1016/2"]

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.get_cited("10.1038/nature12373")

            mock_crl.get_cited.assert_called_once_with("10.1038/nature12373")
            assert result == ["10.1016/1", "10.1016/2"]

    def test_get_citation_count_delegates_to_crossref_local(self):
        """Test that get_citation_count() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_crl.get_citation_count.return_value = 42

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.get_citation_count("10.1038/nature12373")

            mock_crl.get_citation_count.assert_called_once_with("10.1038/nature12373")
            assert result == 42


class TestConfigurationFunctions:
    """Test configuration function delegation."""

    def test_configure_delegates_to_crossref_local(self):
        """Test that configure() delegates to crossref-local."""
        mock_crl = MagicMock()

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            crossref_scitex.configure("/path/to/db.sqlite")

            mock_crl.configure.assert_called_once_with("/path/to/db.sqlite")

    def test_configure_http_delegates_to_crossref_local(self):
        """Test that configure_http() delegates to crossref-local."""
        mock_crl = MagicMock()

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            crossref_scitex.configure_http("http://localhost:31291")

            mock_crl.configure_http.assert_called_once_with("http://localhost:31291")

    def test_get_mode_delegates_to_crossref_local(self):
        """Test that get_mode() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_crl.get_mode.return_value = "http"

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.get_mode()

            mock_crl.get_mode.assert_called_once()
            assert result == "http"

    def test_info_delegates_to_crossref_local(self):
        """Test that info() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_crl.info.return_value = {"status": "ok", "work_count": 167000000}

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.info()

            mock_crl.info.assert_called_once()
            assert result["status"] == "ok"
            assert result["work_count"] == 167000000


class TestIsAvailableFunction:
    """Test is_available function."""

    def test_is_available_returns_true_when_ok(self):
        """Test that is_available() returns True when status is ok."""
        mock_crl = MagicMock()
        mock_crl.info.return_value = {"status": "ok"}

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.is_available()

            assert result is True

    def test_is_available_returns_false_when_error(self):
        """Test that is_available() returns False when error occurs."""
        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            side_effect=ImportError("not installed"),
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.is_available()

            assert result is False


class TestEnrichmentFunctions:
    """Test enrichment function delegation."""

    def test_enrich_delegates_to_crossref_local(self):
        """Test that enrich() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_input = MagicMock()
        mock_output = MagicMock()
        mock_crl.enrich.return_value = mock_output

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            result = crossref_scitex.enrich(mock_input)

            mock_crl.enrich.assert_called_once_with(mock_input)
            assert result == mock_output

    def test_enrich_dois_delegates_to_crossref_local(self):
        """Test that enrich_dois() delegates to crossref-local."""
        mock_crl = MagicMock()
        mock_works = [MagicMock(), MagicMock()]
        mock_crl.enrich_dois.return_value = mock_works

        with patch(
            "scitex.scholar.crossref_scitex._ensure_crossref_local",
            return_value=mock_crl,
        ):
            from scitex.scholar import crossref_scitex

            dois = ["10.1038/1", "10.1038/2"]
            result = crossref_scitex.enrich_dois(dois)

            mock_crl.enrich_dois.assert_called_once_with(dois)
            assert result == mock_works


# EOF
