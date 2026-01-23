#!/usr/bin/env python3
"""Tests for URLDOIEngine - Extract DOIs from URL fields."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from scitex.scholar.metadata_engines.individual import URLDOIEngine


class TestURLDOIEngineInit:
    """Tests for URLDOIEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = URLDOIEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = URLDOIEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"

    def test_init_api_key_from_env(self):
        """Should read API key from environment variable."""
        with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "env_api_key"}):
            engine = URLDOIEngine()
            assert engine.api_key == "env_api_key"

    def test_ieee_patterns_defined(self):
        """Should have IEEE patterns defined."""
        engine = URLDOIEngine()
        assert len(engine.ieee_patterns) > 0
        assert any("ieeexplore" in p for p in engine.ieee_patterns)

    def test_pubmed_patterns_defined(self):
        """Should have PubMed patterns defined."""
        engine = URLDOIEngine()
        assert len(engine.pubmed_patterns) > 0
        assert any("pubmed" in p for p in engine.pubmed_patterns)

    def test_semantic_patterns_defined(self):
        """Should have Semantic Scholar patterns defined."""
        engine = URLDOIEngine()
        assert len(engine.semantic_patterns) > 0
        assert any("semanticscholar" in p for p in engine.semantic_patterns)


class TestURLDOIEngineProperties:
    """Tests for URLDOIEngine properties."""

    def test_name_property(self):
        """Name property should return 'URL'."""
        engine = URLDOIEngine()
        assert engine.name == "URL"

    def test_rate_limit_delay(self):
        """Rate limit delay should be 0.0 seconds."""
        engine = URLDOIEngine()
        assert engine.rate_limit_delay == 0.0


class TestURLDOIEngineSearch:
    """Tests for search method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_search_by_doi_calls_correct_method(self, engine):
        """Should call _search_by_doi when DOI provided."""
        with patch.object(engine, "_search_by_doi") as mock_method:
            mock_method.return_value = {"id": {"doi": "10.1038/test"}}
            engine.search(doi="10.1038/test")
            mock_method.assert_called_once_with("10.1038/test", "dict")

    def test_search_by_url_calls_correct_method(self, engine):
        """Should call _search_by_url when URL provided."""
        with patch.object(engine, "_search_by_url") as mock_method:
            mock_method.return_value = {"id": {"doi": "10.1038/test"}}
            engine.search(title="Test", url="https://doi.org/10.1038/test")
            mock_method.assert_called_once()

    def test_search_doi_takes_priority(self, engine):
        """DOI should take priority over URL."""
        with patch.object(engine, "_search_by_doi") as mock_doi:
            with patch.object(engine, "_search_by_url") as mock_url:
                mock_doi.return_value = {"id": {"doi": "10.1038/test"}}
                engine.search(doi="10.1038/test", url="https://example.com")
                mock_doi.assert_called_once()
                mock_url.assert_not_called()


class TestURLDOIEngineSearchByDOI:
    """Tests for _search_by_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_cleans_doi_url_https(self, engine):
        """Should remove https://doi.org/ prefix from DOI."""
        result = engine._search_by_doi("https://doi.org/10.1038/test", "dict")
        assert result["id"]["doi"] == "10.1038/test"

    def test_cleans_doi_url_http(self, engine):
        """Should remove http://doi.org/ prefix from DOI."""
        result = engine._search_by_doi("http://doi.org/10.1038/test", "dict")
        assert result["id"]["doi"] == "10.1038/test"

    def test_successful_doi_search(self, engine):
        """Should return metadata for valid DOI."""
        result = engine._search_by_doi("10.1038/nature12373", "dict")

        assert result["id"]["doi"] == "10.1038/nature12373"
        assert result["url"]["doi"] == "https://doi.org/10.1038/nature12373"
        assert result["id"]["doi_engines"] == ["URL"]

    def test_tracks_engine_source(self, engine):
        """Should track URL as source engine."""
        result = engine._search_by_doi("10.1038/test", "dict")
        assert result["id"]["doi_engines"] == ["URL"]
        assert result["system"]["searched_by_URL"] is True

    def test_return_as_json(self, engine):
        """Should return JSON string when return_as='json'."""
        result = engine._search_by_doi("10.1038/test", "json")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["id"]["doi"] == "10.1038/test"

    def test_invalid_return_as_returns_none(self, engine):
        """Should return None for invalid return_as (caught by exception handler)."""
        # The assertion error is caught by the try/except block which returns None
        result = engine._search_by_doi("10.1038/test", "invalid")
        assert result is None


class TestURLDOIEngineSearchByURL:
    """Tests for _search_by_url method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_returns_minimal_without_url(self, engine):
        """Should return minimal metadata when no URL provided."""
        result = engine._search_by_url()
        assert result is not None

    def test_extracts_doi_from_doi_url(self, engine):
        """Should extract DOI from doi.org URL."""
        mock_extractor = MagicMock()
        mock_extractor.extract_doi_from_url.return_value = "10.1038/test"
        engine._url_doi_extractor = mock_extractor

        result = engine._search_by_url(
            title="Test Paper", url="https://doi.org/10.1038/test"
        )

        assert result is not None
        assert result["id"]["doi"] == "10.1038/test"

    def test_preserves_title_from_input(self, engine):
        """Should preserve title from input."""
        mock_extractor = MagicMock()
        mock_extractor.extract_doi_from_url.return_value = "10.1038/test"
        engine._url_doi_extractor = mock_extractor

        result = engine._search_by_url(
            title="My Test Paper", url="https://doi.org/10.1038/test"
        )

        assert result["basic"]["title"] == "My Test Paper"


class TestURLDOIEngineExtractPubmedId:
    """Tests for _extract_pubmed_id method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_extracts_from_pubmed_url(self, engine):
        """Should extract PMID from pubmed URL."""
        result = engine._extract_pubmed_id("https://pubmed/12345678")
        assert result == "12345678"

    def test_extracts_from_ncbi_url(self, engine):
        """Should extract PMID from NCBI URL."""
        result = engine._extract_pubmed_id(
            "https://www.ncbi.nlm.nih.gov/pubmed/12345678"
        )
        assert result == "12345678"

    def test_extracts_from_pmid_prefix(self, engine):
        """Should extract from PMID: prefix."""
        result = engine._extract_pubmed_id("PMID:12345678")
        assert result == "12345678"

    def test_returns_none_for_non_pubmed(self, engine):
        """Should return None for non-PubMed URLs."""
        result = engine._extract_pubmed_id("https://example.com/article")
        assert result is None


class TestURLDOIEngineExtractIEEEId:
    """Tests for _extract_ieee_id method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_extracts_from_ieee_document(self, engine):
        """Should extract ID from IEEE document URL."""
        result = engine._extract_ieee_id(
            "https://ieeexplore.ieee.org/document/12345678"
        )
        assert result == "12345678"

    def test_extracts_from_ieee_abstract(self, engine):
        """Should extract ID from IEEE abstract URL."""
        result = engine._extract_ieee_id(
            "https://ieeexplore.ieee.org/abstract/document/12345678"
        )
        assert result == "12345678"

    def test_extracts_from_ieee_stamp(self, engine):
        """Should extract ID from IEEE stamp URL."""
        result = engine._extract_ieee_id(
            "https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=12345678"
        )
        assert result == "12345678"

    def test_returns_none_for_non_ieee(self, engine):
        """Should return None for non-IEEE URLs."""
        result = engine._extract_ieee_id("https://example.com/article")
        assert result is None


class TestURLDOIEngineExtractSemanticCorpusId:
    """Tests for _extract_semantic_corpus_id method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_extracts_from_semantic_scholar_url(self, engine):
        """Should extract corpus ID from Semantic Scholar URL."""
        result = engine._extract_semantic_corpus_id(
            "https://www.semanticscholar.org/paper/12345678"
        )
        assert result == "12345678"

    def test_extracts_from_corpus_id_prefix(self, engine):
        """Should extract from CorpusId: prefix."""
        result = engine._extract_semantic_corpus_id("CorpusId:12345678")
        assert result == "12345678"

    def test_returns_none_for_non_semantic(self, engine):
        """Should return None for non-Semantic Scholar URLs."""
        result = engine._extract_semantic_corpus_id("https://example.com/article")
        assert result is None


class TestURLDOIEngineLookupIEEEDOI:
    """Tests for _lookup_ieee_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_successful_ieee_lookup(self, engine):
        """Should extract DOI from IEEE page."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '"doi":"10.1109/TEST.2023.12345"'

        with patch("requests.get", return_value=mock_response):
            result = engine._lookup_ieee_doi("12345678")
            assert result == "10.1109/TEST.2023.12345"

    def test_failed_ieee_lookup(self, engine):
        """Should return None on failure."""
        with patch("requests.get", side_effect=Exception("Network error")):
            result = engine._lookup_ieee_doi("12345678")
            assert result is None

    def test_ieee_404_returns_none(self, engine):
        """Should return None for 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            result = engine._lookup_ieee_doi("99999999")
            assert result is None


class TestURLDOIEngineLookupSemanticScholarDOI:
    """Tests for _lookup_semantic_scholar_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_successful_semantic_lookup(self, engine):
        """Should extract DOI from Semantic Scholar API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "externalIds": {"DOI": "10.1038/nature12373"},
            "title": "Test Paper",
        }

        with patch("requests.get", return_value=mock_response):
            result = engine._lookup_semantic_scholar_doi("12345678")
            assert result == "10.1038/nature12373"

    def test_semantic_404_returns_none(self, engine):
        """Should return None for 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("requests.get", return_value=mock_response):
            result = engine._lookup_semantic_scholar_doi("99999999")
            assert result is None

    def test_semantic_no_doi_returns_none(self, engine):
        """Should return None when no DOI in response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"externalIds": {}, "title": "Test Paper"}

        with patch("requests.get", return_value=mock_response):
            result = engine._lookup_semantic_scholar_doi("12345678")
            assert result is None

    def test_semantic_rate_limit_retry(self, engine):
        """Should retry on rate limit (429)."""
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {
            "externalIds": {"DOI": "10.1038/test"},
        }

        with patch("requests.get", side_effect=[rate_limit_response, success_response]):
            with patch("time.sleep"):
                result = engine._lookup_semantic_scholar_doi("12345678")
                assert result == "10.1038/test"

    def test_uses_corpus_id_format_for_digits(self, engine):
        """Should use CorpusId format for numeric IDs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"externalIds": {"DOI": "10.1038/test"}}

        with patch("requests.get", return_value=mock_response) as mock_get:
            engine._lookup_semantic_scholar_doi("12345678")
            call_url = mock_get.call_args[0][0]
            assert "CorpusId:12345678" in call_url

    def test_uses_direct_format_for_non_digits(self, engine):
        """Should use direct format for non-numeric IDs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"externalIds": {"DOI": "10.1038/test"}}

        with patch("requests.get", return_value=mock_response) as mock_get:
            engine._lookup_semantic_scholar_doi("abc123def")
            call_url = mock_get.call_args[0][0]
            assert "abc123def" in call_url
            assert "CorpusId:" not in call_url


class TestURLDOIEngineCleanDOI:
    """Tests for _clean_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_strips_whitespace(self, engine):
        """Should strip whitespace from DOI."""
        result = engine._clean_doi("  10.1038/test  ")
        assert result == "10.1038/test"

    def test_handles_none(self, engine):
        """Should handle None DOI."""
        result = engine._clean_doi(None)
        assert result is None


class TestURLDOIEngineEdgeCases:
    """Edge case tests for URLDOIEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_handles_empty_url(self, engine):
        """Should handle empty URL string."""
        result = engine._search_by_url(title="Test", url="")
        assert result is not None

    def test_handles_none_url(self, engine):
        """Should handle None URL."""
        result = engine._search_by_url(title="Test", url=None)
        assert result is not None

    def test_handles_malformed_ieee_url(self, engine):
        """Should handle malformed IEEE URL."""
        result = engine._extract_ieee_id("https://ieeexplore.ieee.org/")
        assert result is None

    def test_handles_unicode_in_url(self, engine):
        """Should handle unicode in URL."""
        result = engine._extract_pubmed_id("https://pubmed/12345678?title=étude")
        assert result == "12345678"


class TestURLDOIEngineIntegration:
    """Integration tests for URLDOIEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return URLDOIEngine()

    def test_full_doi_workflow(self, engine):
        """Test complete DOI search workflow."""
        result = engine.search(doi="10.1038/nature12373")

        assert result is not None
        assert result["id"]["doi"] == "10.1038/nature12373"
        assert result["url"]["doi"] == "https://doi.org/10.1038/nature12373"
        assert result["system"]["searched_by_URL"] is True

    def test_full_url_workflow_with_doi_extractor(self, engine):
        """Test complete URL workflow with DOI extractor."""
        mock_extractor = MagicMock()
        mock_extractor.extract_doi_from_url.return_value = "10.1038/nature12373"
        engine._url_doi_extractor = mock_extractor

        result = engine.search(
            title="Test Paper", url="https://doi.org/10.1038/nature12373"
        )

        assert result is not None
        assert result["id"]["doi"] == "10.1038/nature12373"

    def test_fallback_to_minimal_metadata(self, engine):
        """Should return minimal metadata when all extraction fails."""
        mock_extractor = MagicMock()
        mock_extractor.extract_doi_from_url.return_value = None
        engine._url_doi_extractor = mock_extractor

        result = engine.search(
            title="Unknown Paper",
            year=2023,
            url="https://unknown-site.com/article",
        )

        assert result is not None


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
