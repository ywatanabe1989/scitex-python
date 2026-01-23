#!/usr/bin/env python3
"""Tests for CrossRefLocalEngine - CrossRef Local API metadata retrieval engine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.metadata_engines.individual import CrossRefLocalEngine


class TestCrossRefLocalEngineInit:
    """Tests for CrossRefLocalEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = CrossRefLocalEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = CrossRefLocalEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"

    def test_init_default_api_url(self):
        """Should have default API URL."""
        engine = CrossRefLocalEngine()
        assert engine.api_url == "http://127.0.0.1:3333"

    def test_init_custom_api_url(self):
        """Should accept custom API URL."""
        engine = CrossRefLocalEngine(api_url="http://custom:8080")
        assert engine.api_url == "http://custom:8080"

    def test_strips_trailing_slash(self):
        """Should strip trailing slash from API URL."""
        engine = CrossRefLocalEngine(api_url="http://custom:8080/")
        assert engine.api_url == "http://custom:8080"


class TestCrossRefLocalEngineProperties:
    """Tests for CrossRefLocalEngine properties."""

    def test_name_property(self):
        """Name property should return 'CrossRefLocal'."""
        engine = CrossRefLocalEngine()
        assert engine.name == "CrossRefLocal"

    def test_rate_limit_delay(self):
        """Rate limit delay should be 0.01 seconds."""
        engine = CrossRefLocalEngine()
        assert engine.rate_limit_delay == 0.01


class TestCrossRefLocalEngineAPIDetection:
    """Tests for API type detection."""

    def test_detects_internal_api(self):
        """Should detect internal API (Docker/local)."""
        engine = CrossRefLocalEngine(api_url="http://crossref:3333")
        assert engine._is_external_api is False

    def test_detects_external_api_by_path(self):
        """Should detect external API by path."""
        engine = CrossRefLocalEngine(api_url="https://scitex.ai/scholar/api/crossref")
        assert engine._is_external_api is True

    def test_detects_external_api_by_domain(self):
        """Should detect external API by domain."""
        engine = CrossRefLocalEngine(api_url="https://scitex.ai/api")
        assert engine._is_external_api is True


class TestCrossRefLocalEngineBuildEndpoint:
    """Tests for _build_endpoint_url method."""

    def test_builds_internal_endpoint(self):
        """Should build internal API endpoint correctly."""
        engine = CrossRefLocalEngine(api_url="http://crossref:3333")
        url = engine._build_endpoint_url("search")
        assert url == "http://crossref:3333/api/search/"

    def test_builds_external_endpoint(self):
        """Should build external API endpoint correctly."""
        engine = CrossRefLocalEngine(api_url="https://scitex.ai/scholar/api/crossref")
        url = engine._build_endpoint_url("search")
        assert url == "https://scitex.ai/scholar/api/crossref/search/"


class TestCrossRefLocalEngineSearch:
    """Tests for search method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefLocalEngine()

    def test_search_with_doi(self, engine):
        """Should search with DOI parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "DOI": "10.1038/test",
            "title": ["Test Paper"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine.search(doi="10.1038/test")
        call_params = mock_session.get.call_args[1]["params"]
        assert call_params["doi"] == "10.1038/test"

    def test_search_cleans_doi_url(self, engine):
        """Should clean DOI URL prefix."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "DOI": "10.1038/test",
            "title": ["Test Paper"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine.search(doi="https://doi.org/10.1038/test")
        call_params = mock_session.get.call_args[1]["params"]
        assert call_params["doi"] == "10.1038/test"

    def test_search_with_title(self, engine):
        """Should search with title parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine.search(title="Test Paper")
        call_params = mock_session.get.call_args[1]["params"]
        assert call_params["title"] == "Test Paper"

    def test_search_with_year(self, engine):
        """Should search with year parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine.search(title="Test", year=2023)
        call_params = mock_session.get.call_args[1]["params"]
        assert call_params["year"] == "2023"

    def test_search_with_authors(self, engine):
        """Should search with authors parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine.search(title="Test", authors=["John Doe", "Jane Smith"])
        call_params = mock_session.get.call_args[1]["params"]
        assert call_params["authors"] == "John Doe Jane Smith"

    def test_search_no_params_returns_minimal(self, engine):
        """Should return minimal metadata when no params."""
        result = engine.search()
        assert result is not None
        assert "id" in result


class TestCrossRefLocalEngineMakeSearchRequest:
    """Tests for _make_search_request method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefLocalEngine()

    def test_successful_doi_search(self, engine):
        """Should return metadata for valid DOI response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "DOI": "10.1038/nature12373",
            "title": ["Test Paper Title"],
            "published-print": {"date-parts": [[2023, 5, 15]]},
            "author": [{"given": "John", "family": "Doe"}],
            "container-title": ["Nature"],
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._make_search_request({"doi": "10.1038/nature12373"}, "dict")
        assert result["id"]["doi"] == "10.1038/nature12373"
        assert result["basic"]["title"] == "Test Paper Title"

    def test_handles_results_array(self, engine):
        """Should handle results array format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"doi": "10.1038/test"}]}
        mock_response.raise_for_status = MagicMock()

        # Second call for DOI lookup
        doi_response = MagicMock()
        doi_response.json.return_value = {
            "DOI": "10.1038/test",
            "title": ["Found Paper"],
        }
        doi_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.side_effect = [mock_response, doi_response]
        engine._session = mock_session

        result = engine._make_search_request({"title": "Test"}, "dict")
        assert result is not None

    def test_handles_api_error(self, engine):
        """Should handle API errors gracefully."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Connection refused")
        engine._session = mock_session

        result = engine._make_search_request({"doi": "10.1038/test"}, "dict")
        assert result is not None

    def test_handles_error_response(self, engine):
        """Should handle error in response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "Not found"}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._make_search_request({"doi": "10.1038/test"}, "dict")
        assert result is not None


class TestCrossRefLocalEngineExtractMetadata:
    """Tests for _extract_metadata_from_crossref_data method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefLocalEngine()

    def test_extracts_title(self, engine):
        """Should extract title from data."""
        data = {"title": ["Test Paper Title"]}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["basic"]["title"] == "Test Paper Title"

    def test_removes_trailing_period(self, engine):
        """Should remove trailing period from title."""
        data = {"title": ["Test Paper Title."]}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["basic"]["title"] == "Test Paper Title"

    def test_extracts_year_from_published_print(self, engine):
        """Should extract year from published-print."""
        data = {"published-print": {"date-parts": [[2023, 5, 15]]}}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["basic"]["year"] == 2023

    def test_extracts_year_from_published_online(self, engine):
        """Should fall back to published-online for year."""
        data = {"published-online": {"date-parts": [[2022, 3, 10]]}}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["basic"]["year"] == 2022

    def test_extracts_authors(self, engine):
        """Should extract author names correctly."""
        data = {
            "author": [
                {"given": "John", "family": "Doe"},
                {"given": "Jane", "family": "Smith"},
                {"family": "Anonymous"},
            ]
        }
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert "John Doe" in result["basic"]["authors"]
        assert "Jane Smith" in result["basic"]["authors"]
        assert "Anonymous" in result["basic"]["authors"]

    def test_extracts_doi(self, engine):
        """Should extract DOI."""
        data = {"DOI": "10.1038/nature12373"}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["id"]["doi"] == "10.1038/nature12373"

    def test_extracts_journal(self, engine):
        """Should extract journal name."""
        data = {"container-title": ["Nature Communications"]}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["publication"]["journal"] == "Nature Communications"

    def test_extracts_short_journal(self, engine):
        """Should extract short journal name."""
        data = {"short-container-title": ["Nat Commun"]}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["publication"]["short_journal"] == "Nat Commun"

    def test_extracts_publisher(self, engine):
        """Should extract publisher."""
        data = {"publisher": "Nature Publishing Group"}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["publication"]["publisher"] == "Nature Publishing Group"

    def test_extracts_volume_issue(self, engine):
        """Should extract volume and issue."""
        data = {"volume": "10", "issue": "5"}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["publication"]["volume"] == "10"
        assert result["publication"]["issue"] == "5"

    def test_extracts_issn(self, engine):
        """Should extract ISSN."""
        data = {"ISSN": ["2041-1723", "2041-1731"]}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["publication"]["issn"] == "2041-1723"

    def test_builds_doi_url(self, engine):
        """Should build DOI URL."""
        data = {"DOI": "10.1038/test"}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["url"]["doi"] == "https://doi.org/10.1038/test"

    def test_tracks_engine_source(self, engine):
        """Should track CrossRefLocal as source engine."""
        data = {"DOI": "10.1038/test", "title": ["Test"]}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["id"]["doi_engines"] == ["CrossRefLocal"]
        assert result["basic"]["title_engines"] == ["CrossRefLocal"]
        assert result["system"]["searched_by_CrossRefLocal"] is True

    def test_return_as_json(self, engine):
        """Should return JSON string when return_as='json'."""
        data = {"DOI": "10.1038/test", "title": ["Test"]}
        result = engine._extract_metadata_from_crossref_data(data, "json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "id" in parsed


class TestCrossRefLocalEngineEdgeCases:
    """Edge case tests for CrossRefLocalEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return CrossRefLocalEngine()

    def test_handles_empty_data(self, engine):
        """Should handle empty data."""
        result = engine._extract_metadata_from_crossref_data({}, "dict")
        assert result is not None

    def test_handles_error_in_data(self, engine):
        """Should handle error in data."""
        result = engine._extract_metadata_from_crossref_data(
            {"error": "Not found"}, "dict"
        )
        assert result is not None

    def test_handles_none_data(self, engine):
        """Should handle None data."""
        result = engine._extract_metadata_from_crossref_data(None, "dict")
        assert result is not None

    def test_handles_empty_title_list(self, engine):
        """Should handle empty title list."""
        data = {"title": []}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["basic"]["title"] is None

    def test_handles_missing_date_parts(self, engine):
        """Should handle missing date parts."""
        data = {"published-print": {}}
        result = engine._extract_metadata_from_crossref_data(data, "dict")
        assert result["basic"]["year"] is None

    def test_handles_connection_refused(self, engine):
        """Should handle connection refused error."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception(
            "Max retries exceeded with url: /api/search/"
        )
        engine._session = mock_session

        result = engine.search(doi="10.1038/test")
        assert result is not None


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
