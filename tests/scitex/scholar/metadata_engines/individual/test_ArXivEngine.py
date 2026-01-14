#!/usr/bin/env python3
"""Tests for ArXivEngine - ArXiv metadata retrieval engine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.metadata_engines.individual import ArXivEngine


class TestArXivEngineInit:
    """Tests for ArXivEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = ArXivEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = ArXivEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"

    def test_base_url(self):
        """Should have correct ArXiv API URL."""
        engine = ArXivEngine()
        assert engine.base_url == "http://export.arxiv.org/api/query"


class TestArXivEngineProperties:
    """Tests for ArXivEngine properties."""

    def test_name_property(self):
        """Name property should return 'arXiv'."""
        engine = ArXivEngine()
        assert engine.name == "arXiv"

    def test_rate_limit_delay(self):
        """Rate limit delay should be 3.0 seconds."""
        engine = ArXivEngine()
        assert engine.rate_limit_delay == 3.0


class TestArXivEngineSearch:
    """Tests for search method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ArXivEngine()

    def test_search_by_doi_calls_correct_method(self, engine):
        """Should call _search_by_doi when DOI provided."""
        with patch.object(engine, "_search_by_doi") as mock_method:
            mock_method.return_value = {"id": {"doi": "10.48550/arxiv.1234.5678"}}
            engine.search(doi="10.48550/arxiv.1234.5678")
            mock_method.assert_called_once()

    def test_search_by_title_calls_correct_method(self, engine):
        """Should call _search_by_metadata when title provided."""
        with patch.object(engine, "_search_by_metadata") as mock_method:
            mock_method.return_value = {"basic": {"title": "Test Paper"}}
            engine.search(title="Test Paper")
            mock_method.assert_called_once()

    def test_search_doi_takes_priority(self, engine):
        """DOI should take priority over title when both provided."""
        with patch.object(engine, "_search_by_doi") as mock_doi:
            with patch.object(engine, "_search_by_metadata") as mock_meta:
                mock_doi.return_value = {"id": {"doi": "10.48550/arxiv.1234.5678"}}
                engine.search(doi="10.48550/arxiv.1234.5678", title="Test Paper")
                mock_doi.assert_called_once()
                mock_meta.assert_not_called()


class TestArXivEngineSearchQuery:
    """Tests for _build_arxiv_search_query method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ArXivEngine()

    def test_builds_query_from_title(self, engine):
        """Should build query from title keywords."""
        query = engine._build_arxiv_search_query("Deep Learning for Image Recognition")
        assert "ti:deep" in query.lower()
        assert "ti:learning" in query.lower()
        assert "ti:image" in query.lower()

    def test_filters_short_words(self, engine):
        """Should filter words with 3 or fewer characters."""
        query = engine._build_arxiv_search_query("A Deep Learning for AI")
        # 'a' and 'for' should be filtered out
        assert "ti:a" not in query.lower().split()
        assert "ti:for" not in query.lower()

    def test_filters_stop_words(self, engine):
        """Should filter common stop words."""
        query = engine._build_arxiv_search_query(
            "The Deep Learning Method Using Neural Networks"
        )
        assert "ti:the" not in query.lower()
        assert "ti:using" not in query.lower()

    def test_includes_first_author(self, engine):
        """Should include first author surname in query."""
        query = engine._build_arxiv_search_query(
            "Deep Learning", authors=["John Smith", "Jane Doe"]
        )
        assert "au:Smith" in query


class TestArXivEngineSearchByDOI:
    """Tests for _search_by_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ArXivEngine()

    def test_cleans_doi_url(self, engine):
        """Should clean DOI URL prefix."""
        # Mock feedparser response
        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[])
            mock_response = MagicMock()
            mock_response.text = "<feed></feed>"
            mock_response.raise_for_status = MagicMock()

            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            engine._session = mock_session

            engine._search_by_doi("https://doi.org/10.48550/arXiv.1706.03762", "dict")
            call_params = mock_session.get.call_args[1]["params"]
            # Should extract arXiv ID
            assert "1706.03762" in call_params["search_query"]

    def test_handles_arxiv_doi_format(self, engine):
        """Should handle arXiv DOI format correctly."""
        with patch("feedparser.parse") as mock_parse:
            mock_parse.return_value = MagicMock(entries=[])
            mock_response = MagicMock()
            mock_response.text = "<feed></feed>"
            mock_response.raise_for_status = MagicMock()

            mock_session = MagicMock()
            mock_session.get.return_value = mock_response
            engine._session = mock_session

            engine._search_by_doi("10.48550/arXiv.2301.12345", "dict")
            call_params = mock_session.get.call_args[1]["params"]
            assert "2301.12345" in call_params["search_query"]

    def test_failed_doi_search_returns_minimal(self, engine):
        """Should return minimal metadata on failure."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Network error")
        engine._session = mock_session

        result = engine._search_by_doi("10.48550/arXiv.1234.5678", "dict")
        assert result["id"]["doi"] == "10.48550/arXiv.1234.5678"


class TestArXivEngineSearchByMetadata:
    """Tests for _search_by_metadata method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ArXivEngine()

    def test_returns_minimal_without_title(self, engine):
        """Should return minimal metadata when no title provided."""
        result = engine._search_by_metadata(title=None)
        assert result is not None
        assert "id" in result

    def test_handles_api_error(self, engine):
        """Should handle API errors gracefully."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("API Error")
        engine._session = mock_session

        result = engine._search_by_metadata(title="Test Paper")
        assert result is not None
        assert result["basic"]["title"] == "Test Paper"


class TestArXivEngineExtractMetadata:
    """Tests for _extract_metadata_from_entry method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ArXivEngine()

    @pytest.fixture
    def mock_entry(self):
        """Create mock ArXiv entry."""
        entry = MagicMock()
        entry.id = "http://arxiv.org/abs/1706.03762v1"
        entry.get.side_effect = lambda key, default="": {
            "title": "Attention Is All You Need",
            "summary": "This paper proposes the Transformer architecture...",
            "published": "2017-06-12T17:57:34Z",
            "link": "http://arxiv.org/abs/1706.03762",
            "authors": [
                MagicMock(**{"get.return_value": "Ashish Vaswani"}),
                MagicMock(**{"get.return_value": "Noam Shazeer"}),
            ],
        }.get(key, default)
        return entry

    def test_extracts_arxiv_id(self, engine, mock_entry):
        """Should extract arXiv ID from entry."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert result["id"]["arxiv_id"] == "1706.03762"

    def test_extracts_title(self, engine, mock_entry):
        """Should extract title from entry."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert result["basic"]["title"] == "Attention Is All You Need"

    def test_removes_trailing_period(self, engine):
        """Should remove trailing period from title."""
        mock_entry = MagicMock()
        mock_entry.id = "http://arxiv.org/abs/1234.5678v1"
        mock_entry.get.side_effect = lambda key, default="": {
            "title": "Test Paper Title.",
            "published": "2023-01-01",
            "authors": [],
        }.get(key, default)

        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert result["basic"]["title"] == "Test Paper Title"

    def test_extracts_year(self, engine, mock_entry):
        """Should extract publication year."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert result["basic"]["year"] == "2017"

    def test_extracts_authors(self, engine, mock_entry):
        """Should extract author names."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert "Ashish Vaswani" in result["basic"]["authors"]
            assert "Noam Shazeer" in result["basic"]["authors"]

    def test_extracts_abstract(self, engine, mock_entry):
        """Should extract abstract."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert "Transformer" in result["basic"]["abstract"]

    def test_generates_arxiv_doi(self, engine, mock_entry):
        """Should generate arXiv DOI if not scraped."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert "10.48550/arxiv.1706.03762" in result["id"]["doi"]

    def test_uses_scraped_doi(self, engine, mock_entry):
        """Should use scraped DOI if available."""
        scraped_doi = "10.5555/3295222.3295349"
        with patch.object(engine, "_scrape_doi", return_value=scraped_doi):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert result["id"]["doi"] == scraped_doi

    def test_sets_journal_to_arxiv(self, engine, mock_entry):
        """Should set journal to arXiv."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert result["publication"]["journal"] == "arXiv"

    def test_tracks_engine_source(self, engine, mock_entry):
        """Should track arXiv as source engine."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert result["id"]["arxiv_id_engines"] == ["arXiv"]
            assert result["basic"]["title_engines"] == ["arXiv"]
            assert result["system"]["searched_by_arXiv"] is True

    def test_return_as_json(self, engine, mock_entry):
        """Should return JSON string when return_as='json'."""
        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "json")
            assert isinstance(result, str)
            parsed = json.loads(result)
            assert "id" in parsed


class TestArXivEngineScrapeDoI:
    """Tests for _scrape_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ArXivEngine()

    def test_scrapes_doi_from_page(self, engine):
        """Should scrape DOI from arxiv page."""
        mock_response = MagicMock()
        mock_response.content = b"""
        <html>
            <a href="https://doi.org/10.5555/123456">DOI Link</a>
        </html>
        """

        with patch("requests.get", return_value=mock_response):
            doi = engine._scrape_doi("http://arxiv.org/abs/1234.5678")
            assert doi == "10.5555/123456"

    def test_handles_missing_doi(self, engine):
        """Should handle pages without DOI link."""
        mock_response = MagicMock()
        mock_response.content = b"<html><body>No DOI here</body></html>"

        with patch("requests.get", return_value=mock_response):
            doi = engine._scrape_doi("http://arxiv.org/abs/1234.5678")
            assert doi is None

    def test_handles_network_error(self, engine):
        """Should handle network errors gracefully."""
        with patch("requests.get", side_effect=Exception("Network error")):
            doi = engine._scrape_doi("http://arxiv.org/abs/1234.5678")
            assert doi is None


class TestArXivEngineEdgeCases:
    """Edge case tests for ArXivEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return ArXivEngine()

    def test_handles_old_arxiv_id_format(self, engine):
        """Should handle old arXiv ID format (e.g., hep-th/9802109)."""
        mock_entry = MagicMock()
        mock_entry.id = "http://arxiv.org/abs/hep-th/9802109v1"
        mock_entry.get.side_effect = lambda key, default="": {
            "title": "Test Paper",
            "published": "1998-02-17",
            "authors": [],
        }.get(key, default)

        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert "hep-th" in result["id"]["arxiv_id"]

    def test_handles_unicode_title(self, engine):
        """Should handle unicode in titles."""
        mock_entry = MagicMock()
        mock_entry.id = "http://arxiv.org/abs/1234.5678v1"
        mock_entry.get.side_effect = lambda key, default="": {
            "title": "Etude sur le reseau neuronal",
            "published": "2023-01-01",
            "authors": [],
        }.get(key, default)

        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            assert "Etude" in result["basic"]["title"]

    def test_handles_newlines_in_abstract(self, engine):
        """Should handle newlines in abstract."""
        mock_entry = MagicMock()
        mock_entry.id = "http://arxiv.org/abs/1234.5678v1"
        mock_entry.get.side_effect = lambda key, default="": {
            "title": "Test",
            "summary": "Line 1\nLine 2\nLine 3",
            "published": "2023-01-01",
            "authors": [],
        }.get(key, default)

        with patch.object(engine, "_scrape_doi", return_value=None):
            result = engine._extract_metadata_from_entry(mock_entry, "dict")
            # Newlines should be replaced with spaces
            assert "\n" not in result["basic"]["abstract"]
            assert "Line 1 Line 2 Line 3" == result["basic"]["abstract"]


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
