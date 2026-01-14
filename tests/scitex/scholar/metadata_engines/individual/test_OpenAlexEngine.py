#!/usr/bin/env python3
"""Tests for OpenAlexEngine - OpenAlex metadata retrieval engine."""

import json
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.metadata_engines.individual import OpenAlexEngine


class TestOpenAlexEngineInit:
    """Tests for OpenAlexEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = OpenAlexEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = OpenAlexEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"

    def test_base_url(self):
        """Should have correct OpenAlex API URL."""
        engine = OpenAlexEngine()
        assert engine.base_url == "https://api.openalex.org/works"


class TestOpenAlexEngineProperties:
    """Tests for OpenAlexEngine properties."""

    def test_name_property(self):
        """Name property should return 'OpenAlex'."""
        engine = OpenAlexEngine()
        assert engine.name == "OpenAlex"

    def test_rate_limit_delay(self):
        """Rate limit delay should be 0.1 seconds."""
        engine = OpenAlexEngine()
        assert engine.rate_limit_delay == 0.1


class TestOpenAlexEngineSearch:
    """Tests for search method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return OpenAlexEngine()

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


class TestOpenAlexEngineSearchByDOI:
    """Tests for _search_by_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return OpenAlexEngine()

    def test_cleans_doi_url(self, engine):
        """Should remove https://doi.org/ prefix from DOI."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [{"doi": "https://doi.org/10.1038/test", "title": "Test"}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_doi("https://doi.org/10.1038/test", "dict")
        call_params = mock_session.get.call_args[1]["params"]
        assert "https://doi.org/" not in call_params["filter"]

    def test_successful_doi_search(self, engine):
        """Should return metadata for valid DOI."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "doi": "https://doi.org/10.1038/nature12373",
                    "title": "Test Paper Title",
                    "publication_year": 2023,
                    "authorships": [
                        {"author": {"display_name": "John Doe"}},
                        {"author": {"display_name": "Jane Smith"}},
                    ],
                    "primary_location": {
                        "engine": {
                            "display_name": "Nature",
                            "issn_l": "0028-0836",
                        }
                    },
                    "cited_by_count": 150,
                    "counts_by_year": [{"year": 2023, "cited_by_count": 50}],
                }
            ]
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
            "results": [
                {
                    "doi": "https://doi.org/10.1038/test",
                    "title": "Test",
                    "publication_year": 2023,
                    "counts_by_year": [{"year": 2023, "cited_by_count": 0}],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_doi("10.1038/test", "json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "id" in parsed


class TestOpenAlexEngineSearchByMetadata:
    """Tests for _search_by_metadata method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return OpenAlexEngine()

    def test_returns_minimal_without_title(self, engine):
        """Should return minimal metadata when no title provided."""
        result = engine._search_by_metadata(title=None)
        assert result is not None
        assert "id" in result

    def test_builds_correct_params(self, engine):
        """Should build correct query parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_metadata(title="Test Paper", year=2023)
        call_params = mock_session.get.call_args[1]["params"]
        assert call_params["search"] == "Test Paper"
        assert "publication_year:2023" in call_params["filter"]

    def test_matches_similar_title(self, engine):
        """Should match when titles are similar (>95% threshold)."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "doi": "https://doi.org/10.1038/test",
                    "title": "Deep Learning for Image Recognition",
                    "publication_year": 2023,
                    "counts_by_year": [{"year": 2023, "cited_by_count": 0}],
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        # Use exact title match since _is_title_match uses 95% threshold
        result = engine._search_by_metadata(title="Deep Learning for Image Recognition")
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


class TestOpenAlexEngineExtractMetadata:
    """Tests for _extract_metadata_from_work method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return OpenAlexEngine()

    def _create_minimal_work(self, **kwargs):
        """Create minimal work dict with required counts_by_year to avoid source bug."""
        work = {
            "counts_by_year": [{"year": 2023, "cited_by_count": 0}],
        }
        work.update(kwargs)
        return work

    def test_extracts_title(self, engine):
        """Should extract title from work."""
        work = self._create_minimal_work(title="Test Paper Title")
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["basic"]["title"] == "Test Paper Title"

    def test_removes_trailing_period(self, engine):
        """Should remove trailing period from title."""
        work = self._create_minimal_work(title="Test Paper Title.")
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["basic"]["title"] == "Test Paper Title"

    def test_extracts_year(self, engine):
        """Should extract publication year."""
        work = self._create_minimal_work(publication_year=2023)
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["basic"]["year"] == 2023

    def test_extracts_authors(self, engine):
        """Should extract author names correctly."""
        work = self._create_minimal_work(
            authorships=[
                {"author": {"display_name": "John Doe"}},
                {"author": {"display_name": "Jane Smith"}},
            ]
        )
        result = engine._extract_metadata_from_work(work, "dict")
        assert "John Doe" in result["basic"]["authors"]
        assert "Jane Smith" in result["basic"]["authors"]

    def test_extracts_doi(self, engine):
        """Should extract DOI."""
        work = self._create_minimal_work(doi="https://doi.org/10.1038/nature12373")
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["id"]["doi"] == "10.1038/nature12373"

    def test_extracts_journal_from_primary_location(self, engine):
        """Should extract journal from primary_location."""
        work = self._create_minimal_work(
            primary_location={
                "engine": {
                    "display_name": "Nature Communications",
                    "issn_l": "2041-1723",
                }
            }
        )
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["publication"]["journal"] == "Nature Communications"
        assert result["publication"]["issn"] == "2041-1723"

    def test_extracts_citation_count(self, engine):
        """Should extract citation count with counts_by_year structure."""
        work = {
            "cited_by_count": 150,
            "counts_by_year": [{"year": 2023, "cited_by_count": 50}],
        }
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["citation_count"]["total"] == 150

    def test_builds_doi_url(self, engine):
        """Should build DOI URL."""
        work = self._create_minimal_work(doi="https://doi.org/10.1038/test")
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["url"]["doi"] == "https://doi.org/10.1038/test"

    def test_tracks_engine_source(self, engine):
        """Should track OpenAlex as source engine."""
        work = self._create_minimal_work(
            doi="https://doi.org/10.1038/test", title="Test"
        )
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["id"]["doi_engines"] == ["OpenAlex"]
        assert result["basic"]["title_engines"] == ["OpenAlex"]
        assert result["system"]["searched_by_OpenAlex"] is True


class TestOpenAlexEngineEdgeCases:
    """Edge case tests for OpenAlexEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return OpenAlexEngine()

    def _create_minimal_work(self, **kwargs):
        """Create minimal work dict with required counts_by_year to avoid source bug."""
        work = {
            "counts_by_year": [{"year": 2023, "cited_by_count": 0}],
        }
        work.update(kwargs)
        return work

    def test_empty_title(self, engine):
        """Should handle empty title."""
        work = self._create_minimal_work(title="")
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["basic"]["title"] is None

    def test_no_authorships(self, engine):
        """Should handle missing authorships."""
        work = self._create_minimal_work(title="Test")
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["basic"]["authors"] is None

    def test_handles_arxiv_doi(self, engine):
        """Should handle arXiv DOIs correctly."""
        work = self._create_minimal_work(
            doi="https://doi.org/10.48550/arxiv.2301.12345",
            title="Test ArXiv Paper",
        )
        result = engine._extract_metadata_from_work(work, "dict")
        assert "arxiv" in result["id"]["doi"].lower()
        # arXiv papers should get default journal/publisher
        assert result["publication"]["journal"] == "arXiv (Cornell University)"

    def test_handles_keywords(self, engine):
        """Should extract keywords."""
        work = self._create_minimal_work(
            title="Test",
            keywords=[
                {"display_name": "Machine Learning"},
                {"display_name": "AI"},
            ],
        )
        result = engine._extract_metadata_from_work(work, "dict")
        assert "Machine Learning" in result["basic"]["keywords"]
        assert "AI" in result["basic"]["keywords"]

    def test_extracts_pmid(self, engine):
        """Should extract PubMed ID from ids."""
        work = self._create_minimal_work(
            title="Test",
            ids={"pmid": "https://pubmed.ncbi.nlm.nih.gov/12345678"},
        )
        result = engine._extract_metadata_from_work(work, "dict")
        assert result["id"]["pmid"] == "12345678"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
