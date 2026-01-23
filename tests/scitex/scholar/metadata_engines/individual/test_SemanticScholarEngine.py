#!/usr/bin/env python3
"""Tests for SemanticScholarEngine - Semantic Scholar metadata retrieval engine."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from tenacity import RetryError

from scitex.scholar.metadata_engines.individual import SemanticScholarEngine


class TestSemanticScholarEngineInit:
    """Tests for SemanticScholarEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = SemanticScholarEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = SemanticScholarEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"

    def test_init_without_api_key(self):
        """Should initialize without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var if present
            os.environ.pop("SEMANTIC_SCHOLAR_API_KEY", None)
            engine = SemanticScholarEngine()
            assert engine.api_key is None

    def test_init_with_api_key_param(self):
        """Should accept API key parameter."""
        engine = SemanticScholarEngine(api_key="test-api-key")
        assert engine.api_key == "test-api-key"

    def test_init_with_env_api_key(self):
        """Should read API key from environment variable."""
        with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "env-api-key"}):
            engine = SemanticScholarEngine()
            assert engine.api_key == "env-api-key"

    def test_init_param_overrides_env(self):
        """Parameter API key should override environment variable."""
        with patch.dict(os.environ, {"SEMANTIC_SCHOLAR_API_KEY": "env-api-key"}):
            engine = SemanticScholarEngine(api_key="param-api-key")
            assert engine.api_key == "param-api-key"

    def test_base_url(self):
        """Should have correct Semantic Scholar API URL."""
        engine = SemanticScholarEngine()
        assert engine.base_url == "https://api.semanticscholar.org/graph/v1"


class TestSemanticScholarEngineProperties:
    """Tests for SemanticScholarEngine properties."""

    def test_name_property(self):
        """Name property should return 'Semantic_Scholar'."""
        engine = SemanticScholarEngine()
        assert engine.name == "Semantic_Scholar"

    def test_rate_limit_delay_without_api_key(self):
        """Rate limit delay should be 1.2 seconds without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SEMANTIC_SCHOLAR_API_KEY", None)
            engine = SemanticScholarEngine()
            assert engine.rate_limit_delay == 1.2

    def test_rate_limit_delay_with_api_key(self):
        """Rate limit delay should be 0.5 seconds with API key."""
        engine = SemanticScholarEngine(api_key="test-key")
        assert engine.rate_limit_delay == 0.5

    def test_user_agent(self):
        """Should return correct user agent string."""
        engine = SemanticScholarEngine(email="test@example.com")
        assert engine._get_user_agent() == "SciTeX/1.0 (mailto:test@example.com)"


class TestSemanticScholarEngineSession:
    """Tests for session property and headers."""

    def test_session_sets_headers_without_api_key(self):
        """Session should set basic headers without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SEMANTIC_SCHOLAR_API_KEY", None)
            engine = SemanticScholarEngine()
            session = engine.session
            assert "User-Agent" in session.headers
            assert "Accept" in session.headers
            assert "x-api-key" not in session.headers

    def test_session_sets_api_key_header(self):
        """Session should include API key in headers when provided."""
        engine = SemanticScholarEngine(api_key="test-api-key")
        session = engine.session
        assert session.headers.get("x-api-key") == "test-api-key"

    def test_session_is_cached(self):
        """Session should be cached after first access."""
        engine = SemanticScholarEngine()
        session1 = engine.session
        session2 = engine.session
        assert session1 is session2


class TestSemanticScholarEngineSearch:
    """Tests for search method routing."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SemanticScholarEngine()

    def test_search_by_doi_calls_correct_method(self, engine):
        """Should call _search_by_doi when DOI provided."""
        with patch.object(engine, "_search_by_doi") as mock_method:
            mock_method.return_value = {"id": {"doi": "10.1038/test"}}
            engine.search(doi="10.1038/test")
            mock_method.assert_called_once_with("10.1038/test", "dict")

    def test_search_by_corpus_id_calls_correct_method(self, engine):
        """Should call _search_by_corpus_id when corpus_id provided."""
        with patch.object(engine, "_search_by_corpus_id") as mock_method:
            mock_method.return_value = {"id": {"corpus_id": "12345678"}}
            engine.search(corpus_id="12345678")
            mock_method.assert_called_once_with("12345678", "dict")

    def test_search_by_title_calls_correct_method(self, engine):
        """Should call _search_by_metadata when title provided."""
        with patch.object(engine, "_search_by_metadata") as mock_method:
            mock_method.return_value = {"basic": {"title": "Test Paper"}}
            engine.search(title="Test Paper")
            mock_method.assert_called_once()

    def test_doi_takes_priority_over_corpus_id(self, engine):
        """DOI should take priority over corpus_id when both provided."""
        with patch.object(engine, "_search_by_doi") as mock_doi:
            with patch.object(engine, "_search_by_corpus_id") as mock_corpus:
                mock_doi.return_value = {"id": {"doi": "10.1038/test"}}
                engine.search(doi="10.1038/test", corpus_id="12345678")
                mock_doi.assert_called_once()
                mock_corpus.assert_not_called()

    def test_corpus_id_takes_priority_over_title(self, engine):
        """Corpus ID should take priority over title when both provided."""
        with patch.object(engine, "_search_by_corpus_id") as mock_corpus:
            with patch.object(engine, "_search_by_metadata") as mock_meta:
                mock_corpus.return_value = {"id": {"corpus_id": "12345678"}}
                engine.search(corpus_id="12345678", title="Test Paper")
                mock_corpus.assert_called_once()
                mock_meta.assert_not_called()


class TestSemanticScholarEngineSearchByDOI:
    """Tests for _search_by_doi method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SemanticScholarEngine()

    def test_cleans_doi_url(self, engine):
        """Should remove https://doi.org/ prefix from DOI."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "title": "Test Paper",
            "year": 2023,
            "externalIds": {"DOI": "10.1038/test"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_doi("https://doi.org/10.1038/test", "dict")
        call_url = mock_session.get.call_args[0][0]
        assert "https://doi.org/" not in call_url
        assert "10.1038/test" in call_url

    def test_successful_doi_search(self, engine):
        """Should return metadata for valid DOI."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "title": "Test Paper Title",
            "year": 2023,
            "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
            "externalIds": {
                "DOI": "10.1038/nature12373",
                "CorpusId": "12345678",
            },
            "venue": "Nature",
            "url": "https://www.semanticscholar.org/paper/12345678",
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
            "title": "Test",
            "year": 2023,
            "externalIds": {"DOI": "10.1038/test"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_doi("10.1038/test", "json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "id" in parsed


class TestSemanticScholarEngineSearchByCorpusId:
    """Tests for _search_by_corpus_id method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SemanticScholarEngine()

    def test_cleans_corpus_id_prefix(self, engine):
        """Should clean CorpusId: prefix from corpus_id."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Test Paper",
            "year": 2023,
            "externalIds": {"CorpusId": "12345678"},
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_corpus_id("CorpusId:12345678", "dict")
        call_url = mock_session.get.call_args[0][0]
        assert "CorpusId:12345678" in call_url

    def test_successful_corpus_id_search(self, engine):
        """Should return metadata for valid corpus_id."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Test Paper Title",
            "year": 2023,
            "authors": [{"name": "John Doe"}],
            "externalIds": {
                "DOI": "10.1038/test",
                "CorpusId": "12345678",
            },
            "venue": "Nature",
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_corpus_id("12345678", "dict")
        assert result["id"]["corpus_id"] == "12345678"

    def test_returns_none_for_404(self, engine):
        """Should return None when corpus_id not found (404)."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_corpus_id("99999999", "dict")
        assert result is None

    def test_handles_rate_limit(self, engine):
        """Should raise RetryError (wrapping ConnectionError) on 429 rate limit after exhausting retries."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        # tenacity wraps ConnectionError in RetryError after exhausting retries
        with pytest.raises(RetryError):
            engine._search_by_corpus_id("12345678", "dict")


class TestSemanticScholarEngineSearchByMetadata:
    """Tests for _search_by_metadata method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SemanticScholarEngine()

    def test_returns_none_without_title(self, engine):
        """Should return None when no title provided."""
        result = engine._search_by_metadata(title=None)
        assert result is None

    def test_builds_correct_params(self, engine):
        """Should build correct query parameters."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        engine._search_by_metadata(title="Test Paper", year=2023)
        call_params = mock_session.get.call_args[1]["params"]
        assert "Test Paper" in call_params["query"]
        assert call_params["limit"] == 10

    def test_matches_similar_title(self, engine):
        """Should match when titles are similar."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "title": "Deep Learning for Image Recognition",
                    "year": 2023,
                    "authors": [{"name": "John Doe"}],
                    "externalIds": {"DOI": "10.1038/test", "CorpusId": "12345"},
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        # Use exact title match
        result = engine._search_by_metadata(title="Deep Learning for Image Recognition")
        assert result is not None
        assert result["id"]["doi"] == "10.1038/test"

    def test_filters_by_year(self, engine):
        """Should filter results by year when provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "title": "Test Paper",
                    "year": 2022,  # Wrong year
                    "authors": [],
                    "externalIds": {"DOI": "10.1038/wrong"},
                },
                {
                    "title": "Test Paper",
                    "year": 2023,  # Correct year
                    "authors": [],
                    "externalIds": {"DOI": "10.1038/correct", "CorpusId": "12345"},
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_metadata(title="Test Paper", year=2023)
        assert result["id"]["doi"] == "10.1038/correct"

    def test_filters_by_authors(self, engine):
        """Should filter results by authors when provided."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {
                    "title": "Test Paper",
                    "year": 2023,
                    "authors": [{"name": "Wrong Author"}],
                    "externalIds": {"DOI": "10.1038/wrong"},
                },
                {
                    "title": "Test Paper",
                    "year": 2023,
                    "authors": [{"name": "John Doe"}, {"name": "Jane Smith"}],
                    "externalIds": {"DOI": "10.1038/correct", "CorpusId": "12345"},
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        result = engine._search_by_metadata(title="Test Paper", authors=["John Doe"])
        assert result["id"]["doi"] == "10.1038/correct"

    def test_handles_api_error(self, engine):
        """Should handle API errors gracefully."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("API Error")
        engine._session = mock_session

        result = engine._search_by_metadata(title="Test")
        assert result is not None
        assert result["basic"]["title"] == "Test"

    def test_handles_rate_limit_429(self, engine):
        """Should raise RetryError (wrapping ConnectionError) on 429 rate limit after exhausting retries."""
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        engine._session = mock_session

        # tenacity wraps ConnectionError in RetryError after exhausting retries
        with pytest.raises(RetryError):
            engine._search_by_metadata(title="Test Paper")


class TestSemanticScholarEngineExtractMetadata:
    """Tests for _extract_metadata_from_paper method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SemanticScholarEngine()

    def _create_paper(self, **kwargs):
        """Create minimal paper dict."""
        paper = {
            "title": "",
            "year": None,
            "authors": [],
            "externalIds": {},
            "venue": "",
            "url": "",
            "abstract": "",
        }
        paper.update(kwargs)
        return paper

    def test_extracts_title(self, engine):
        """Should extract title from paper."""
        paper = self._create_paper(title="Test Paper Title")
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["basic"]["title"] == "Test Paper Title"

    def test_extracts_year(self, engine):
        """Should extract publication year."""
        paper = self._create_paper(year=2023)
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["basic"]["year"] == 2023

    def test_extracts_authors(self, engine):
        """Should extract author names correctly."""
        paper = self._create_paper(
            authors=[
                {"name": "John Doe"},
                {"name": "Jane Smith"},
            ]
        )
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert "John Doe" in result["basic"]["authors"]
        assert "Jane Smith" in result["basic"]["authors"]

    def test_extracts_doi(self, engine):
        """Should extract DOI from externalIds."""
        paper = self._create_paper(externalIds={"DOI": "10.1038/nature12373"})
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["id"]["doi"] == "10.1038/nature12373"

    def test_extracts_corpus_id(self, engine):
        """Should extract CorpusId from externalIds."""
        paper = self._create_paper(externalIds={"CorpusId": "12345678"})
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["id"]["corpus_id"] == "12345678"

    def test_extracts_arxiv_id(self, engine):
        """Should extract ArXiv ID from externalIds."""
        paper = self._create_paper(externalIds={"ArXiv": "2301.12345"})
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["id"]["arxiv_id"] == "2301.12345"

    def test_extracts_pmid(self, engine):
        """Should extract PubMed ID from externalIds."""
        paper = self._create_paper(externalIds={"PubMed": "12345678"})
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["id"]["pmid"] == "12345678"

    def test_extracts_venue_as_journal(self, engine):
        """Should extract venue as journal."""
        paper = self._create_paper(venue="Nature Communications")
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["publication"]["journal"] == "Nature Communications"

    def test_extracts_abstract(self, engine):
        """Should extract abstract."""
        paper = self._create_paper(abstract="This is the abstract of the paper.")
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["basic"]["abstract"] == "This is the abstract of the paper."

    def test_builds_doi_url(self, engine):
        """Should build DOI URL."""
        paper = self._create_paper(externalIds={"DOI": "10.1038/test"})
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["url"]["doi"] == "https://doi.org/10.1038/test"

    def test_builds_arxiv_url(self, engine):
        """Should build arXiv URL."""
        paper = self._create_paper(externalIds={"ArXiv": "2301.12345"})
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["url"]["arxiv"] == "https://arxiv.org/abs/2301.12345"

    def test_builds_corpus_id_url(self, engine):
        """Should build Semantic Scholar corpus URL."""
        paper = self._create_paper(externalIds={"CorpusId": "12345678"})
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert (
            result["url"]["corpus_id"]
            == "https://www.semanticscholar.org/paper/12345678"
        )

    def test_extracts_publisher_url(self, engine):
        """Should extract publisher URL."""
        paper = self._create_paper(url="https://www.semanticscholar.org/paper/12345678")
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert (
            result["url"]["publisher"]
            == "https://www.semanticscholar.org/paper/12345678"
        )

    def test_tracks_engine_source(self, engine):
        """Should track Semantic_Scholar as source engine."""
        paper = self._create_paper(
            title="Test Paper",
            externalIds={"DOI": "10.1038/test", "CorpusId": "12345"},
        )
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["id"]["doi_engines"] == ["Semantic_Scholar"]
        assert result["id"]["corpus_id_engines"] == ["Semantic_Scholar"]
        assert result["basic"]["title_engines"] == ["Semantic_Scholar"]
        assert result["system"]["searched_by_Semantic_Scholar"] is True

    def test_return_as_json(self, engine):
        """Should return JSON string when return_as='json'."""
        paper = self._create_paper(title="Test", externalIds={"DOI": "10.1038/test"})
        result = engine._extract_metadata_from_paper(paper, "json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "id" in parsed


class TestSemanticScholarEngineEdgeCases:
    """Edge case tests for SemanticScholarEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SemanticScholarEngine()

    def test_handles_empty_title(self, engine):
        """Should handle empty title."""
        paper = {
            "title": "",
            "year": 2023,
            "authors": [],
            "externalIds": {},
        }
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["basic"]["title"] is None

    def test_handles_missing_authors(self, engine):
        """Should handle missing authors."""
        paper = {
            "title": "Test",
            "year": 2023,
            "externalIds": {},
        }
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["basic"]["authors"] is None

    def test_handles_authors_without_names(self, engine):
        """Should handle authors without name field."""
        paper = {
            "title": "Test",
            "year": 2023,
            "authors": [{"authorId": "123"}, {"authorId": "456"}],
            "externalIds": {},
        }
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["basic"]["authors"] is None

    def test_handles_missing_external_ids(self, engine):
        """Should handle missing externalIds."""
        paper = {
            "title": "Test",
            "year": 2023,
            "authors": [],
        }
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["id"]["doi"] is None
        assert result["id"]["corpus_id"] is None

    def test_handles_empty_venue(self, engine):
        """Should handle empty venue."""
        paper = {
            "title": "Test",
            "year": 2023,
            "authors": [],
            "externalIds": {},
            "venue": "",
        }
        result = engine._extract_metadata_from_paper(paper, "dict")
        assert result["publication"]["journal"] is None

    def test_handles_connection_error(self, engine):
        """Should raise RetryError after exhausting retries on ConnectionError."""
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.ConnectionError("Connection failed")
        engine._session = mock_session

        # tenacity wraps ConnectionError in RetryError after exhausting retries
        with pytest.raises(RetryError):
            engine._search_by_doi("10.1038/test", "dict")

    def test_clean_doi_strips_whitespace(self, engine):
        """Should strip whitespace from DOI."""
        result = engine._clean_doi("  10.1038/test  ")
        assert result == "10.1038/test"

    def test_clean_doi_handles_none(self, engine):
        """Should handle None DOI."""
        result = engine._clean_doi(None)
        assert result is None


class TestSemanticScholarEngineRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return SemanticScholarEngine()

    def test_handle_rate_limit_sleeps_when_needed(self, engine):
        """Should sleep when requests are too fast."""
        import time

        engine.last_request_time = time.time()  # Just requested

        with patch("time.sleep") as mock_sleep:
            with patch("time.time") as mock_time:
                # Simulate time passing
                mock_time.side_effect = [
                    engine.last_request_time + 0.1,  # First call (too soon)
                    engine.last_request_time + 1.5,  # After sleep
                ]
                engine._handle_rate_limit()
                # Should have slept for the difference
                mock_sleep.assert_called_once()

    def test_handle_rate_limit_no_sleep_when_delay_passed(self, engine):
        """Should not sleep when enough time has passed."""
        import time

        engine.last_request_time = time.time() - 10  # 10 seconds ago

        with patch("time.sleep") as mock_sleep:
            engine._handle_rate_limit()
            mock_sleep.assert_not_called()


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
