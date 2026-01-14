#!/usr/bin/env python3
"""Tests for CrossRefEngine - CrossRef metadata retrieval engine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.metadata_engines.individual import CrossRefEngine


class TestCrossRefEngineInit:
    """Tests for CrossRefEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = CrossRefEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = CrossRefEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"

    def test_base_url(self):
        """Should have correct CrossRef API URL."""
        engine = CrossRefEngine()
        assert engine.base_url == "https://api.crossref.org/works"


class TestCrossRefEngineProperties:
    """Tests for CrossRefEngine properties."""

    def test_name_property(self):
        """Name property should return 'CrossRef'."""
        engine = CrossRefEngine()
        assert engine.name == "CrossRef"

    def test_rate_limit_delay(self):
        """Rate limit delay should be 0.1 seconds."""
        engine = CrossRefEngine()
        assert engine.rate_limit_delay == 0.1


class TestCrossRefEngineSearch:
    """Tests for search method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefEngine()

    def test_search_by_doi_calls_correct_method(self, engine):
        """Should call _search_by_doi when DOI provided."""
        with patch.object(engine, "_search_by_doi") as mock_method:
            mock_method.return_value = {"id": {"doi": "10.1038/test"}}
            engine.search(doi="10.1038/test")
            mock_method.assert_called_once_with("10.1038/test", "dict")

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
                mock_doi.return_value = {"id": {"doi": "10.1038/test"}}
                engine.search(doi="10.1038/test", title="Test Paper")
                mock_doi.assert_called_once()
                mock_meta.assert_not_called()


class TestCrossRefEngineSearchByDOI:
    """Tests for _search_by_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefEngine()

    def test_cleans_doi_url(self, engine):
        """Should remove https://doi.org/ prefix from DOI."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"DOI": "10.1038/test", "title": ["Test"]}
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_doi("https://doi.org/10.1038/test", "dict")
        call_url = mock_session.get.call_args[0][0]
        assert "https://doi.org/" not in call_url

    def test_successful_doi_search(self, engine):
        """Should return metadata for valid DOI."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "DOI": "10.1038/nature12373",
                "title": ["Test Paper Title"],
                "published-print": {"date-parts": [[2023]]},
                "author": [{"given": "John", "family": "Doe"}],
                "container-title": ["Nature"],
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_doi("10.1038/nature12373", "dict")
        assert result["id"]["doi"] == "10.1038/nature12373"
        assert result["basic"]["title"] == "Test Paper Title"

    def test_failed_doi_search_returns_minimal(self, engine):
        """Should return minimal metadata on failure."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Network error")
        engine._session = mock_session

        result = engine._search_by_doi("10.1038/test", "dict")
        assert result["id"]["doi"] == "10.1038/test"

    def test_return_as_json(self, engine):
        """Should return JSON string when return_as='json'."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"DOI": "10.1038/test", "title": ["Test"]}
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_doi("10.1038/test", "json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "id" in parsed


class TestCrossRefEngineSearchByMetadata:
    """Tests for _search_by_metadata method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefEngine()

    def test_returns_none_without_title(self, engine):
        """Should return None when no title provided."""
        result = engine._search_by_metadata(title=None)
        assert result is None

    def test_builds_correct_params(self, engine):
        """Should build correct query parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"items": []}}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_metadata(title="Test Paper", year=2023)
        call_params = mock_session.get.call_args[1]["params"]
        assert call_params["query.title"] == "Test Paper"
        assert "from-pub-date:2023" in call_params["filter"]

    def test_matches_title_substring(self, engine):
        """Should match when search title is substring of result."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1038/test",
                        "title": ["Test Paper: A Comprehensive Study"],
                    }
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_metadata(title="Test Paper")
        assert result is not None
        assert result["id"]["doi"] == "10.1038/test"

    def test_handles_api_error(self, engine):
        """Should handle API errors gracefully."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("API Error")
        engine._session = mock_session

        result = engine._search_by_metadata(title="Test")
        assert result is not None
        assert result["basic"]["title"] == "Test"


class TestCrossRefEngineExtractMetadata:
    """Tests for _extract_metadata_from_item method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefEngine()

    def test_extracts_title(self, engine):
        """Should extract title from item."""
        item = {"title": ["Test Paper Title"]}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["basic"]["title"] == "Test Paper Title"

    def test_removes_trailing_period(self, engine):
        """Should remove trailing period from title."""
        item = {"title": ["Test Paper Title."]}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["basic"]["title"] == "Test Paper Title"

    def test_extracts_year_from_published_print(self, engine):
        """Should extract year from published-print."""
        item = {"published-print": {"date-parts": [[2023, 5, 15]]}}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["basic"]["year"] == 2023

    def test_extracts_year_from_published_online(self, engine):
        """Should fall back to published-online for year."""
        item = {"published-online": {"date-parts": [[2022]]}}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["basic"]["year"] == 2022

    def test_extracts_authors(self, engine):
        """Should extract author names correctly."""
        item = {
            "author": [
                {"given": "John", "family": "Doe"},
                {"given": "Jane", "family": "Smith"},
                {"family": "Anonymous"},
            ]
        }
        result = engine._extract_metadata_from_item(item, "dict")
        assert "John Doe" in result["basic"]["authors"]
        assert "Jane Smith" in result["basic"]["authors"]
        assert "Anonymous" in result["basic"]["authors"]

    def test_extracts_doi(self, engine):
        """Should extract DOI."""
        item = {"DOI": "10.1038/nature12373"}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["id"]["doi"] == "10.1038/nature12373"

    def test_extracts_journal(self, engine):
        """Should extract journal name."""
        item = {"container-title": ["Nature Communications"]}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["publication"]["journal"] == "Nature Communications"

    def test_extracts_citation_count(self, engine):
        """Should extract citation count."""
        item = {"is-referenced-by-count": 150}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["citation_count"]["total"] == 150

    def test_builds_doi_url(self, engine):
        """Should build DOI URL."""
        item = {"DOI": "10.1038/test"}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["url"]["doi"] == "https://doi.org/10.1038/test"

    def test_tracks_engine_source(self, engine):
        """Should track CrossRef as source engine."""
        item = {"DOI": "10.1038/test", "title": ["Test"]}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["id"]["doi_engines"] == ["CrossRef"]
        assert result["basic"]["title_engines"] == ["CrossRef"]
        assert result["system"]["searched_by_CrossRef"] is True


class TestCrossRefEngineEdgeCases:
    """Edge case tests for CrossRefEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefEngine()

    def test_empty_title_list(self, engine):
        """Should handle empty title list."""
        item = {"title": []}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["basic"]["title"] is None

    def test_missing_date_parts(self, engine):
        """Should handle missing date parts."""
        item = {"published-print": {}}
        result = engine._extract_metadata_from_item(item, "dict")
        assert result["basic"]["year"] is None

    def test_handles_unicode_title(self, engine):
        """Should handle unicode in titles."""
        item = {"title": ["Etude sur les donnees medicales"]}
        result = engine._extract_metadata_from_item(item, "dict")
        assert "Etude" in result["basic"]["title"]

    def test_handles_special_characters(self, engine):
        """Should handle special characters in metadata."""
        item = {
            "title": ["Machine Learning & AI: A Study"],
            "author": [{"given": "Jose", "family": "Garcia-Lopez"}],
        }
        result = engine._extract_metadata_from_item(item, "dict")
        assert "&" in result["basic"]["title"]
        assert "Garcia-Lopez" in result["basic"]["authors"][0]


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
