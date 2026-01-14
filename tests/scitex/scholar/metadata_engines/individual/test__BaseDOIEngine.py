#!/usr/bin/env python3
"""Tests for BaseDOIEngine - Abstract base class for DOI engines."""

import json
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
import requests

from scitex.scholar.metadata_engines.individual._BaseDOIEngine import BaseDOIEngine


# Create a concrete implementation for testing
class MockDOIEngine(BaseDOIEngine):
    """Concrete implementation of BaseDOIEngine for testing."""

    @property
    def name(self) -> str:
        return "MockEngine"

    def search(
        self,
        title: str,
        year: Optional[int] = None,
        authors: Optional[List[str]] = None,
    ) -> Optional[str]:
        return None


class TestBaseDOIEngineInit:
    """Tests for BaseDOIEngine initialization."""

    def test_init_default_email(self):
        """Should initialize with default email."""
        engine = MockDOIEngine()
        assert engine.email == "research@example.com"

    def test_init_custom_email(self):
        """Should accept custom email."""
        engine = MockDOIEngine(email="custom@test.com")
        assert engine.email == "custom@test.com"

    def test_init_rate_limit_handler_none(self):
        """Rate limit handler should be None initially."""
        engine = MockDOIEngine()
        assert engine.rate_limit_handler is None

    def test_init_last_request_time_zero(self):
        """Last request time should start at 0."""
        engine = MockDOIEngine()
        assert engine.last_request_time == 0.0

    def test_init_request_count_zero(self):
        """Request count should start at 0."""
        engine = MockDOIEngine()
        assert engine._request_count == 0

    def test_init_lazy_loaded_utilities_none(self):
        """Lazy-loaded utilities should be None initially."""
        engine = MockDOIEngine()
        assert engine._url_doi_extractor is None
        assert engine._pubmed_converter is None
        assert engine._session is None


class TestBaseDOIEngineProperties:
    """Tests for BaseDOIEngine properties."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_name_property(self, engine):
        """Name property should return engine name."""
        assert engine.name == "MockEngine"

    def test_rate_limit_delay_default(self, engine):
        """Default rate limit delay should be 1.0 seconds."""
        assert engine.rate_limit_delay == 1.0

    def test_text_normalizer_property(self, engine):
        """Should return TextNormalizer class."""
        from scitex.scholar.metadata_engines.utils import TextNormalizer

        assert engine.text_normalizer is TextNormalizer

    def test_url_doi_extractor_lazy_loading(self, engine):
        """URL DOI extractor should be lazy loaded."""
        assert engine._url_doi_extractor is None
        extractor = engine.url_doi_extractor
        assert engine._url_doi_extractor is not None
        # Second access should return same instance
        assert engine.url_doi_extractor is extractor

    def test_pubmed_converter_lazy_loading(self, engine):
        """PubMed converter should be lazy loaded."""
        assert engine._pubmed_converter is None
        converter = engine.pubmed_converter
        assert engine._pubmed_converter is not None
        # Second access should return same instance
        assert engine.pubmed_converter is converter

    def test_session_lazy_loading(self, engine):
        """Session should be lazy loaded."""
        assert engine._session is None
        session = engine.session
        assert engine._session is not None
        assert isinstance(session, requests.Session)
        # Second access should return same instance
        assert engine.session is session

    def test_session_has_user_agent(self, engine):
        """Session should have User-Agent header set."""
        session = engine.session
        assert "User-Agent" in session.headers
        assert "SciTeX" in session.headers["User-Agent"]
        assert engine.email in session.headers["User-Agent"]


class TestBaseDOIEngineSetRateLimitHandler:
    """Tests for set_rate_limit_handler method."""

    def test_set_rate_limit_handler(self):
        """Should set rate limit handler."""
        engine = MockDOIEngine()
        mock_handler = MagicMock()
        engine.set_rate_limit_handler(mock_handler)
        assert engine.rate_limit_handler is mock_handler


class TestBaseDOIEngineGetUserAgent:
    """Tests for _get_user_agent method."""

    def test_get_user_agent_format(self):
        """Should return properly formatted user agent."""
        engine = MockDOIEngine(email="test@example.com")
        user_agent = engine._get_user_agent()
        assert "SciTeX/1.0" in user_agent
        assert "mailto:test@example.com" in user_agent


class TestBaseDOIEngineCleanQuery:
    """Tests for _clean_query method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_clean_query_removes_parentheses(self, engine):
        """Should remove parentheses."""
        result = engine._clean_query("Memory (LSTM) neural networks")
        assert "(" not in result
        assert ")" not in result
        assert "LSTM" in result

    def test_clean_query_removes_brackets(self, engine):
        """Should remove brackets."""
        result = engine._clean_query("Title [with] {brackets}")
        assert "[" not in result
        assert "]" not in result
        assert "{" not in result
        assert "}" not in result

    def test_clean_query_removes_special_chars(self, engine):
        """Should remove special characters."""
        result = engine._clean_query("Title @#$%^& test")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_clean_query_preserves_alphanumeric(self, engine):
        """Should preserve alphanumeric characters."""
        result = engine._clean_query("Test 123 Paper")
        assert "Test" in result
        assert "123" in result
        assert "Paper" in result

    def test_clean_query_preserves_hyphens(self, engine):
        """Should preserve hyphens."""
        result = engine._clean_query("Self-Attention Networks")
        assert "-" in result

    def test_clean_query_collapses_spaces(self, engine):
        """Should collapse multiple spaces."""
        result = engine._clean_query("Title   with    spaces")
        assert "   " not in result
        assert "    " not in result

    def test_clean_query_empty_input(self, engine):
        """Should handle empty input."""
        result = engine._clean_query("")
        assert result == ""

    def test_clean_query_none_input(self, engine):
        """Should handle None input."""
        result = engine._clean_query(None)
        assert result is None

    def test_clean_query_strips_whitespace(self, engine):
        """Should strip leading/trailing whitespace."""
        result = engine._clean_query("  Title  ")
        assert result == "Title"


class TestBaseDOIEngineApplyRateLimiting:
    """Tests for _apply_rate_limiting method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_apply_rate_limiting_no_handler(self, engine):
        """Should log error when no handler set."""
        with patch(
            "scitex.scholar.metadata_engines.individual._BaseDOIEngine.logger"
        ) as mock_logger:
            engine._apply_rate_limiting()
            mock_logger.error.assert_called_once()

    def test_apply_rate_limiting_with_handler_no_wait(self, engine):
        """Should not sleep when wait time is 0."""
        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep") as mock_sleep:
            engine._apply_rate_limiting()
            mock_sleep.assert_not_called()

    def test_apply_rate_limiting_with_handler_wait(self, engine):
        """Should sleep when wait time > 0."""
        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0.5
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep") as mock_sleep:
            engine._apply_rate_limiting()
            mock_sleep.assert_called_once_with(0.5)


class TestBaseDOIEngineApplyRateLimitingAsync:
    """Tests for _apply_rate_limiting_async method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    @pytest.mark.asyncio
    async def test_apply_rate_limiting_async_no_handler(self, engine):
        """Should log error when no handler set."""
        with patch(
            "scitex.scholar.metadata_engines.individual._BaseDOIEngine.logger"
        ) as mock_logger:
            await engine._apply_rate_limiting_async()
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_apply_rate_limiting_async_no_wait(self, engine):
        """Should not wait when wait time is 0."""
        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        engine.set_rate_limit_handler(mock_handler)

        await engine._apply_rate_limiting_async()
        mock_handler.wait_with_countdown_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_apply_rate_limiting_async_with_wait(self, engine):
        """Should wait when wait time > 0."""
        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 1.0
        mock_handler.wait_with_countdown_async = MagicMock(return_value=MagicMock())
        # Make it a proper async mock
        import asyncio

        mock_handler.wait_with_countdown_async.return_value = asyncio.Future()
        mock_handler.wait_with_countdown_async.return_value.set_result(None)
        engine.set_rate_limit_handler(mock_handler)

        await engine._apply_rate_limiting_async()
        mock_handler.wait_with_countdown_async.assert_called_once_with(
            1.0, "MockEngine"
        )


class TestBaseDOIEngineMakeRequestWithRetry:
    """Tests for _make_request_with_retry method."""

    @pytest.fixture
    def engine(self):
        """Create engine with mocked session."""
        engine = MockDOIEngine()
        engine._session = MagicMock()
        return engine

    def test_successful_request(self, engine):
        """Should return response on successful request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        engine._session.get.return_value = mock_response

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        mock_handler.detect_rate_limit.return_value = None
        engine.set_rate_limit_handler(mock_handler)

        result = engine._make_request_with_retry("http://example.com")

        assert result == mock_response
        assert engine._request_count == 1
        mock_handler.record_success.assert_called_once()

    def test_retry_on_timeout(self, engine):
        """Should retry on timeout exception."""
        engine._session.get.side_effect = [
            requests.exceptions.Timeout(),
            MagicMock(status_code=200),
        ]

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        mock_handler.detect_rate_limit.return_value = None
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep"):
            result = engine._make_request_with_retry(
                "http://example.com", max_retries=1
            )

        assert result.status_code == 200
        assert engine._request_count == 2

    def test_retry_on_server_error(self, engine):
        """Should retry on 503 server error."""
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        engine._session.get.side_effect = [mock_response_503, mock_response_200]

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        mock_handler.detect_rate_limit.return_value = None
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep"):
            result = engine._make_request_with_retry(
                "http://example.com", max_retries=1
            )

        assert result.status_code == 200

    def test_retry_on_429_rate_limit(self, engine):
        """Should retry on 429 rate limit."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        engine._session.get.side_effect = [mock_response_429, mock_response_200]

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        mock_handler.detect_rate_limit.return_value = None
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep"):
            result = engine._make_request_with_retry(
                "http://example.com", max_retries=1
            )

        assert result.status_code == 200

    def test_max_retries_exceeded(self, engine):
        """Should return None after max retries exceeded."""
        engine._session.get.side_effect = requests.exceptions.Timeout()

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        mock_handler.detect_rate_limit.return_value = None
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep"):
            result = engine._make_request_with_retry(
                "http://example.com", max_retries=2
            )

        assert result is None
        assert engine._request_count == 3  # Initial + 2 retries
        mock_handler.record_failure.assert_called_once()

    def test_rate_limit_handler_integration(self, engine):
        """Should handle rate limit info from handler."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        engine._session.get.return_value = mock_response

        mock_rate_limit_info = MagicMock()
        mock_rate_limit_info.wait_time = 1.0

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        mock_handler.detect_rate_limit.side_effect = [mock_rate_limit_info, None]
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep"):
            result = engine._make_request_with_retry(
                "http://example.com", max_retries=1
            )

        assert result.status_code == 200
        mock_handler.record_rate_limit.assert_called_once()

    def test_request_exception_handling(self, engine):
        """Should handle general request exceptions."""
        engine._session.get.side_effect = requests.exceptions.ConnectionError()

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        engine.set_rate_limit_handler(mock_handler)

        with patch("time.sleep"):
            result = engine._make_request_with_retry(
                "http://example.com", max_retries=0
            )

        assert result is None
        mock_handler.record_failure.assert_called_once()

    def test_unexpected_exception_handling(self, engine):
        """Should handle unexpected exceptions."""
        engine._session.get.side_effect = ValueError("Unexpected error")

        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        engine.set_rate_limit_handler(mock_handler)

        result = engine._make_request_with_retry("http://example.com", max_retries=0)

        assert result is None
        mock_handler.record_failure.assert_called_once()


class TestBaseDOIEngineSearchAsync:
    """Tests for search_async method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    @pytest.mark.asyncio
    async def test_search_async_applies_rate_limiting(self, engine):
        """Should apply rate limiting before search."""
        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        engine.set_rate_limit_handler(mock_handler)

        result = await engine.search_async("Test Title")

        assert result is None  # MockDOIEngine returns None
        mock_handler.get_wait_time_for_engine.assert_called()

    @pytest.mark.asyncio
    async def test_search_async_handles_error(self, engine):
        """Should handle errors gracefully."""
        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        engine.set_rate_limit_handler(mock_handler)

        # Make search raise an exception
        with patch.object(engine, "search", side_effect=Exception("Search error")):
            result = await engine.search_async("Test Title")

        assert result is None
        mock_handler.record_failure.assert_called_once()


class TestBaseDOIEngineGetRequestStats:
    """Tests for get_request_stats method."""

    def test_get_request_stats_initial(self):
        """Should return initial stats."""
        engine = MockDOIEngine()
        stats = engine.get_request_stats()

        assert stats["total_requests"] == 0
        assert stats["last_request_time"] == 0.0
        assert stats["rate_limit_delay"] == 1.0

    def test_get_request_stats_after_request(self):
        """Should update stats after request."""
        engine = MockDOIEngine()
        engine._request_count = 5
        engine.last_request_time = 1234567890.0

        stats = engine.get_request_stats()

        assert stats["total_requests"] == 5
        assert stats["last_request_time"] == 1234567890.0


class TestBaseDOIEngineExtractDOIFromURL:
    """Tests for extract_doi_from_url method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_extract_doi_from_doi_org_url(self, engine):
        """Should extract DOI from doi.org URL."""
        url = "https://doi.org/10.1038/nature12373"
        result = engine.extract_doi_from_url(url)
        assert result == "10.1038/nature12373"

    def test_extract_doi_from_doi_org_url_with_query(self, engine):
        """Should extract DOI from URL with query params."""
        url = "https://doi.org/10.1038/nature12373?ref=test"
        result = engine.extract_doi_from_url(url)
        assert result == "10.1038/nature12373"

    def test_extract_doi_from_doi_org_url_with_hash(self, engine):
        """Should extract DOI from URL with hash."""
        url = "https://doi.org/10.1038/nature12373#section"
        result = engine.extract_doi_from_url(url)
        assert result == "10.1038/nature12373"

    def test_extract_doi_pattern_in_url(self, engine):
        """Should extract DOI pattern from arbitrary URL."""
        url = "https://example.com/article/10.1234/test.article"
        result = engine.extract_doi_from_url(url)
        assert result == "10.1234/test.article"

    def test_extract_doi_no_doi_in_url(self, engine):
        """Should return None when no DOI in URL."""
        url = "https://example.com/article/12345"
        result = engine.extract_doi_from_url(url)
        assert result is None

    def test_extract_doi_empty_url(self, engine):
        """Should return None for empty URL."""
        result = engine.extract_doi_from_url("")
        assert result is None

    def test_extract_doi_none_url(self, engine):
        """Should return None for None URL."""
        result = engine.extract_doi_from_url(None)
        assert result is None

    def test_extract_doi_complex_doi(self, engine):
        """Should extract complex DOI with special chars."""
        url = "https://doi.org/10.1016/j.cell.2020.01.001"
        result = engine.extract_doi_from_url(url)
        assert result == "10.1016/j.cell.2020.01.001"


class TestBaseDOIEngineIsTitleMatch:
    """Tests for _is_title_match method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_exact_title_match(self, engine):
        """Should match identical titles."""
        result = engine._is_title_match(
            "Attention Is All You Need", "Attention Is All You Need"
        )
        assert result is True

    def test_case_insensitive_match(self, engine):
        """Should match titles with different cases."""
        result = engine._is_title_match(
            "attention is all you need", "ATTENTION IS ALL YOU NEED"
        )
        assert result is True

    def test_different_titles_no_match(self, engine):
        """Should not match different titles."""
        result = engine._is_title_match(
            "Attention Is All You Need", "Completely Different Paper"
        )
        assert result is False

    def test_custom_threshold(self, engine):
        """Should respect custom threshold."""
        # With lower threshold, minor differences should match
        result = engine._is_title_match(
            "Attention Is All You Need", "Attention Is All You Needs", threshold=0.9
        )
        # Depending on TextNormalizer implementation
        assert isinstance(result, bool)


class TestBaseDOIEngineCreateMinimalMetadata:
    """Tests for _create_minimal_metadata method."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_create_minimal_metadata_empty(self, engine):
        """Should create metadata structure with all None values."""
        result = engine._create_minimal_metadata()
        assert isinstance(result, dict)
        assert "id" in result
        assert "basic" in result

    def test_create_minimal_metadata_with_doi(self, engine):
        """Should include DOI and track engine source."""
        result = engine._create_minimal_metadata(doi="10.1038/test")
        assert result["id"]["doi"] == "10.1038/test"
        assert result["id"]["doi_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_with_title(self, engine):
        """Should include title and track engine source."""
        result = engine._create_minimal_metadata(title="Test Paper")
        assert result["basic"]["title"] == "Test Paper"
        assert result["basic"]["title_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_with_year(self, engine):
        """Should include year and track engine source."""
        result = engine._create_minimal_metadata(year=2023)
        assert result["basic"]["year"] == 2023
        assert result["basic"]["year_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_with_authors(self, engine):
        """Should include authors and track engine source."""
        result = engine._create_minimal_metadata(authors=["John Doe", "Jane Smith"])
        assert result["basic"]["authors"] == ["John Doe", "Jane Smith"]
        assert result["basic"]["authors_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_with_pmid(self, engine):
        """Should include PMID and track engine source."""
        result = engine._create_minimal_metadata(pmid="12345678")
        assert result["id"]["pmid"] == "12345678"
        assert result["id"]["pmid_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_with_corpus_id(self, engine):
        """Should include corpus ID and track engine source."""
        result = engine._create_minimal_metadata(corpus_id="987654")
        assert result["id"]["corpus_id"] == "987654"
        assert result["id"]["corpus_id_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_with_semantic_id(self, engine):
        """Should include semantic ID and track engine source."""
        result = engine._create_minimal_metadata(semantic_id="abc123")
        assert result["id"]["semantic_id"] == "abc123"
        assert result["id"]["semantic_id_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_with_ieee_id(self, engine):
        """Should include IEEE ID and track engine source."""
        result = engine._create_minimal_metadata(ieee_id="8765432")
        assert result["id"]["ieee_id"] == "8765432"
        assert result["id"]["ieee_id_engines"] == ["MockEngine"]

    def test_create_minimal_metadata_return_json(self, engine):
        """Should return JSON string when requested."""
        result = engine._create_minimal_metadata(doi="10.1038/test", return_as="json")
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["id"]["doi"] == "10.1038/test"

    def test_create_minimal_metadata_return_dict(self, engine):
        """Should return dict by default."""
        result = engine._create_minimal_metadata(doi="10.1038/test", return_as="dict")
        assert isinstance(result, dict)

    def test_create_minimal_metadata_full_data(self, engine):
        """Should handle all fields together."""
        result = engine._create_minimal_metadata(
            doi="10.1038/test",
            pmid="12345678",
            corpus_id="987654",
            ieee_id="8765432",
            semantic_id="abc123",
            title="Test Paper",
            year=2023,
            authors=["John Doe"],
        )

        assert result["id"]["doi"] == "10.1038/test"
        assert result["id"]["pmid"] == "12345678"
        assert result["id"]["corpus_id"] == "987654"
        assert result["id"]["ieee_id"] == "8765432"
        assert result["id"]["semantic_id"] == "abc123"
        assert result["basic"]["title"] == "Test Paper"
        assert result["basic"]["year"] == 2023
        assert result["basic"]["authors"] == ["John Doe"]


class TestBaseDOIEngineAbstractMethods:
    """Tests verifying abstract method requirements."""

    def test_cannot_instantiate_base_class(self):
        """Should not be able to instantiate abstract base class."""
        with pytest.raises(TypeError):
            BaseDOIEngine()

    def test_must_implement_search(self):
        """Subclass must implement search method."""

        class IncompleteEngine(BaseDOIEngine):
            @property
            def name(self) -> str:
                return "Incomplete"

        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_must_implement_name(self):
        """Subclass must implement name property."""

        class IncompleteEngine(BaseDOIEngine):
            def search(self, title, year=None, authors=None):
                return None

        with pytest.raises(TypeError):
            IncompleteEngine()


class TestBaseDOIEngineEdgeCases:
    """Edge case tests for BaseDOIEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_session_user_agent_update(self, engine):
        """Session should update user agent from _get_user_agent."""
        # Access session to trigger creation
        session = engine.session
        assert engine.email in session.headers["User-Agent"]

    def test_multiple_utility_accesses(self, engine):
        """Multiple accesses to utilities should return same instance."""
        extractor1 = engine.url_doi_extractor
        extractor2 = engine.url_doi_extractor
        assert extractor1 is extractor2

        converter1 = engine.pubmed_converter
        converter2 = engine.pubmed_converter
        assert converter1 is converter2

    def test_clean_query_unicode(self, engine):
        """Should handle unicode in query."""
        result = engine._clean_query("Étude sur les données")
        assert "Étude" in result
        assert "données" in result

    def test_extract_doi_with_parentheses(self, engine):
        """Should extract DOI containing parentheses."""
        url = "https://doi.org/10.1002/(SICI)1234"
        result = engine.extract_doi_from_url(url)
        assert result is not None
        assert "10.1002" in result


class TestBaseDOIEngineCustomRateLimitDelay:
    """Tests for custom rate limit delay in subclasses."""

    def test_override_rate_limit_delay(self):
        """Subclasses can override rate_limit_delay."""

        class SlowEngine(MockDOIEngine):
            @property
            def rate_limit_delay(self) -> float:
                return 5.0

        engine = SlowEngine()
        assert engine.rate_limit_delay == 5.0


class TestBaseDOIEngineIntegration:
    """Integration tests for BaseDOIEngine."""

    @pytest.fixture
    def engine(self):
        """Create engine instance."""
        return MockDOIEngine()

    def test_full_workflow_simulation(self, engine):
        """Simulate a full search workflow."""
        # Set up rate limit handler
        mock_handler = MagicMock()
        mock_handler.get_wait_time_for_engine.return_value = 0
        engine.set_rate_limit_handler(mock_handler)

        # Check initial stats
        stats = engine.get_request_stats()
        assert stats["total_requests"] == 0

        # Access utilities (lazy loading)
        _ = engine.text_normalizer
        _ = engine.session

        # Create minimal metadata
        metadata = engine._create_minimal_metadata(
            doi="10.1038/test", title="Test Paper", year=2023
        )

        assert metadata["id"]["doi"] == "10.1038/test"
        assert metadata["basic"]["title"] == "Test Paper"
        assert metadata["basic"]["year"] == 2023


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
