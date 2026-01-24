#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: tests/scitex/scholar/_mcp/test_crossref_handlers.py
"""Tests for scitex.scholar._mcp.crossref_handlers MCP handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCrossrefHandlersImport:
    """Test handlers import behavior."""

    def test_import_crossref_handlers_module(self):
        """Test that crossref_handlers module can be imported."""
        from scitex.scholar._mcp import crossref_handlers

        assert crossref_handlers is not None

    def test_module_has_expected_exports(self):
        """Test that module exports expected handlers."""
        from scitex.scholar._mcp import crossref_handlers

        expected = [
            "crossref_search_handler",
            "crossref_get_handler",
            "crossref_count_handler",
            "crossref_citations_handler",
            "crossref_info_handler",
        ]
        for handler_name in expected:
            assert hasattr(crossref_handlers, handler_name), f"Missing: {handler_name}"


class TestCrossrefSearchHandler:
    """Test crossref_search_handler async function."""

    @pytest.mark.asyncio
    async def test_search_handler_returns_success_dict(self):
        """Test that search handler returns success dict."""
        mock_crossref = MagicMock()
        mock_result = MagicMock()
        mock_result.total = 100
        mock_result.__iter__ = lambda self: iter([])
        mock_crossref.search.return_value = mock_result

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_search_handler

            result = await crossref_search_handler("deep learning", limit=10)

            assert result["success"] is True
            assert result["query"] == "deep learning"
            assert result["source"] == "crossref_local"
            assert "papers" in result
            assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_search_handler_with_year_filters(self):
        """Test that search handler applies year filters."""
        mock_crossref = MagicMock()

        mock_work1 = MagicMock()
        mock_work1.doi = "10.1038/1"
        mock_work1.title = "Paper 2020"
        mock_work1.year = 2020
        mock_work1.authors = ["Author A"]
        mock_work1.journal = "Nature"
        mock_work1.abstract = "Abstract"
        mock_work1.citation_count = 10
        mock_work1.reference_count = 5
        mock_work1.type = "journal-article"

        mock_work2 = MagicMock()
        mock_work2.doi = "10.1038/2"
        mock_work2.title = "Paper 2018"
        mock_work2.year = 2018
        mock_work2.authors = ["Author B"]
        mock_work2.journal = "Science"
        mock_work2.abstract = "Abstract 2"
        mock_work2.citation_count = 20
        mock_work2.reference_count = 10
        mock_work2.type = "journal-article"

        mock_result = MagicMock()
        mock_result.total = 2
        mock_result.__iter__ = lambda self: iter([mock_work1, mock_work2])
        mock_crossref.search.return_value = mock_result

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_search_handler

            result = await crossref_search_handler("test", limit=10, year_min=2019)

            assert result["success"] is True
            # Only 2020 paper should pass the filter
            assert len(result["papers"]) == 1
            assert result["papers"][0]["year"] == 2020

    @pytest.mark.asyncio
    async def test_search_handler_returns_error_on_exception(self):
        """Test that search handler returns error dict on exception."""
        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            side_effect=RuntimeError("test error"),
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_search_handler

            result = await crossref_search_handler("test")

            assert result["success"] is False
            assert "error" in result


class TestCrossrefGetHandler:
    """Test crossref_get_handler async function."""

    @pytest.mark.asyncio
    async def test_get_handler_returns_paper(self):
        """Test that get handler returns paper details."""
        mock_crossref = MagicMock()
        mock_work = MagicMock()
        mock_work.doi = "10.1038/nature12373"
        mock_work.title = "Test Paper"
        mock_work.authors = ["Author A", "Author B"]
        mock_work.year = 2020
        mock_work.journal = "Nature"
        mock_work.abstract = "Test abstract"
        mock_work.citation_count = 100
        mock_work.reference_count = 50
        mock_work.type = "journal-article"
        mock_work.publisher = "Nature Publishing"
        mock_work.url = "https://doi.org/10.1038/nature12373"
        mock_crossref.get.return_value = mock_work

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_get_handler

            result = await crossref_get_handler("10.1038/nature12373")

            assert result["success"] is True
            assert result["paper"]["doi"] == "10.1038/nature12373"
            assert result["paper"]["title"] == "Test Paper"
            assert result["source"] == "crossref_local"

    @pytest.mark.asyncio
    async def test_get_handler_with_citations(self):
        """Test that get handler includes citations when requested."""
        mock_crossref = MagicMock()
        mock_work = MagicMock()
        mock_work.doi = "10.1038/nature12373"
        mock_work.title = "Test Paper"
        mock_work.authors = []
        mock_work.year = 2020
        mock_work.journal = "Nature"
        mock_work.abstract = None
        mock_work.citation_count = 100
        mock_work.reference_count = 50
        mock_work.type = "journal-article"
        mock_work.publisher = "Nature Publishing"
        mock_work.url = None
        mock_crossref.get.return_value = mock_work
        mock_crossref.get_citing.return_value = ["10.1016/1", "10.1016/2"]

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_get_handler

            result = await crossref_get_handler(
                "10.1038/nature12373", include_citations=True
            )

            assert result["success"] is True
            assert "citing_dois" in result["paper"]
            assert len(result["paper"]["citing_dois"]) == 2

    @pytest.mark.asyncio
    async def test_get_handler_returns_not_found_for_missing_doi(self):
        """Test that get handler returns not found for missing DOI."""
        mock_crossref = MagicMock()
        mock_crossref.get.return_value = None

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_get_handler

            result = await crossref_get_handler("10.0000/nonexistent")

            assert result["success"] is False
            assert "not found" in result["error"].lower()


class TestCrossrefCountHandler:
    """Test crossref_count_handler async function."""

    @pytest.mark.asyncio
    async def test_count_handler_returns_count(self):
        """Test that count handler returns count."""
        mock_crossref = MagicMock()
        mock_crossref.count.return_value = 12345

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_count_handler

            result = await crossref_count_handler("machine learning")

            assert result["success"] is True
            assert result["count"] == 12345
            assert result["query"] == "machine learning"
            assert result["source"] == "crossref_local"


class TestCrossrefCitationsHandler:
    """Test crossref_citations_handler async function."""

    @pytest.mark.asyncio
    async def test_citations_handler_returns_citing(self):
        """Test that citations handler returns citing papers."""
        mock_crossref = MagicMock()
        mock_crossref.get_citing.return_value = ["10.1016/1", "10.1016/2", "10.1016/3"]

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_citations_handler

            result = await crossref_citations_handler(
                "10.1038/nature12373", direction="citing"
            )

            assert result["success"] is True
            assert result["direction"] == "citing"
            assert len(result["citing_dois"]) == 3
            assert result["citing_count"] == 3

    @pytest.mark.asyncio
    async def test_citations_handler_returns_cited(self):
        """Test that citations handler returns cited papers."""
        mock_crossref = MagicMock()
        mock_crossref.get_cited.return_value = ["10.1016/a", "10.1016/b"]

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_citations_handler

            result = await crossref_citations_handler(
                "10.1038/nature12373", direction="cited"
            )

            assert result["success"] is True
            assert result["direction"] == "cited"
            assert len(result["cited_dois"]) == 2
            assert result["cited_count"] == 2

    @pytest.mark.asyncio
    async def test_citations_handler_returns_both(self):
        """Test that citations handler returns both directions."""
        mock_crossref = MagicMock()
        mock_crossref.get_citing.return_value = ["10.1016/1"]
        mock_crossref.get_cited.return_value = ["10.1016/a", "10.1016/b"]

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_citations_handler

            result = await crossref_citations_handler(
                "10.1038/nature12373", direction="both"
            )

            assert result["success"] is True
            assert result["direction"] == "both"
            assert "citing_dois" in result
            assert "cited_dois" in result


class TestCrossrefInfoHandler:
    """Test crossref_info_handler async function."""

    @pytest.mark.asyncio
    async def test_info_handler_returns_info(self):
        """Test that info handler returns database info."""
        mock_crossref = MagicMock()
        mock_crossref.info.return_value = {
            "status": "ok",
            "version": "1.0.0",
            "work_count": 167000000,
        }
        mock_crossref.get_mode.return_value = "http"

        with patch(
            "scitex.scholar._mcp.crossref_handlers._ensure_crossref",
            return_value=mock_crossref,
        ):
            from scitex.scholar._mcp.crossref_handlers import crossref_info_handler

            result = await crossref_info_handler()

            assert result["success"] is True
            assert result["info"]["status"] == "ok"
            assert result["mode"] == "http"
            assert "timestamp" in result


# EOF
