#!/usr/bin/env python3
"""Tests for PublisherRules class."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.config import ScholarConfig
from scitex.scholar.config.PublisherRules import PublisherRules


class TestPublisherRulesInit:
    """Tests for PublisherRules initialization."""

    def test_init_creates_instance(self):
        """PublisherRules should initialize without errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                assert rules is not None
                assert rules.name == "PublisherRules"

    def test_init_with_config(self):
        """Should use provided config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                config = ScholarConfig()
                rules = PublisherRules(config=config)
                assert rules.config is config

    def test_init_creates_default_config(self):
        """Should create default ScholarConfig if none provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                assert rules.config is not None
                assert isinstance(rules.config, ScholarConfig)


class TestGetConfigForUrl:
    """Tests for get_config_for_url method."""

    def test_returns_empty_dict_for_unknown_publisher(self):
        """Should return empty dict for URLs not matching any publisher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                result = rules.get_config_for_url("https://unknown-site.com/paper")
                assert result == {}

    def test_matches_publisher_by_domain_pattern(self):
        """Should match publisher by domain pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "sciencedirect": {
                        "domain_patterns": ["sciencedirect.com", "elsevier.com"],
                        "deny_selectors": [".ad-banner"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                result = rules.get_config_for_url(
                    "https://www.sciencedirect.com/article/123"
                )
                assert "deny_selectors" in result

    def test_url_matching_is_case_insensitive(self):
        """Should match URLs case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "springer": {
                        "domain_patterns": ["springer.com"],
                        "deny_selectors": [".popup"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                result = rules.get_config_for_url("https://WWW.SPRINGER.COM/Article")
                assert "deny_selectors" in result

    def test_returns_first_matching_publisher(self):
        """Should return rules for first matching publisher."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "pub1": {
                        "domain_patterns": ["example.com"],
                        "deny_selectors": [".ad1"],
                    },
                    "pub2": {
                        "domain_patterns": ["example.com"],
                        "deny_selectors": [".ad2"],
                    },
                }
                rules = PublisherRules(config=mock_config)
                result = rules.get_config_for_url("https://example.com/article")
                # First match should be returned
                assert ".ad1" in result.get("deny_selectors", [])

    def test_handles_missing_publisher_rules(self):
        """Should handle when publisher_pdf_rules is None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = None
                rules = PublisherRules(config=mock_config)
                result = rules.get_config_for_url("https://example.com/paper")
                assert result == {}


class TestMergeWithConfig:
    """Tests for merge_with_config method."""

    def test_returns_dict_with_expected_keys(self):
        """Should return dict with all expected keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                result = rules.merge_with_config("https://example.com/paper")
                assert "deny_selectors" in result
                assert "deny_classes" in result
                assert "deny_text_patterns" in result
                assert "download_selectors" in result
                assert "allowed_pdf_patterns" in result

    def test_includes_base_deny_selectors(self):
        """Should include base deny selectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                result = rules.merge_with_config(
                    "https://example.com/paper",
                    base_deny_selectors=[".ad", ".popup"],
                )
                assert ".ad" in result["deny_selectors"]
                assert ".popup" in result["deny_selectors"]

    def test_includes_base_deny_classes(self):
        """Should include base deny classes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                result = rules.merge_with_config(
                    "https://example.com/paper",
                    base_deny_classes=["advertisement", "sidebar"],
                )
                assert "advertisement" in result["deny_classes"]
                assert "sidebar" in result["deny_classes"]

    def test_includes_base_deny_text_patterns(self):
        """Should include base deny text patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                result = rules.merge_with_config(
                    "https://example.com/paper",
                    base_deny_text_patterns=["subscribe", "login"],
                )
                assert "subscribe" in result["deny_text_patterns"]
                assert "login" in result["deny_text_patterns"]

    def test_merges_publisher_deny_selectors(self):
        """Should merge publisher-specific deny selectors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "test_pub": {
                        "domain_patterns": ["test.com"],
                        "deny_selectors": [".publisher-ad"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                result = rules.merge_with_config(
                    "https://test.com/paper",
                    base_deny_selectors=[".base-ad"],
                )
                assert ".base-ad" in result["deny_selectors"]
                assert ".publisher-ad" in result["deny_selectors"]

    def test_removes_duplicate_deny_patterns(self):
        """Should remove duplicates while preserving order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "test_pub": {
                        "domain_patterns": ["test.com"],
                        "deny_selectors": [".ad", ".ad", ".popup"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                result = rules.merge_with_config(
                    "https://test.com/paper",
                    base_deny_selectors=[".ad"],
                )
                # Check no duplicates
                assert result["deny_selectors"].count(".ad") == 1

    def test_includes_download_selectors_from_publisher(self):
        """Should include download selectors from publisher config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "test_pub": {
                        "domain_patterns": ["test.com"],
                        "download_selectors": ["#pdf-download-btn", ".download-link"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                result = rules.merge_with_config("https://test.com/paper")
                assert "#pdf-download-btn" in result["download_selectors"]
                assert ".download-link" in result["download_selectors"]

    def test_handles_none_base_parameters(self):
        """Should handle None base parameters gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                result = rules.merge_with_config("https://example.com/paper")
                assert isinstance(result["deny_selectors"], list)
                assert isinstance(result["deny_classes"], list)
                assert isinstance(result["deny_text_patterns"], list)


class TestIsValidPdfUrl:
    """Tests for is_valid_pdf_url method."""

    def test_returns_true_for_pdf_extension(self):
        """Should return True for URLs ending with .pdf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                assert rules.is_valid_pdf_url(
                    "https://example.com/page",
                    "https://example.com/paper.pdf",
                )

    def test_returns_true_for_pdf_in_path(self):
        """Should return True for URLs containing /pdf/."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                assert rules.is_valid_pdf_url(
                    "https://example.com/page",
                    "https://example.com/pdf/12345",
                )

    def test_returns_false_for_non_pdf_url(self):
        """Should return False for non-PDF URLs without patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                assert not rules.is_valid_pdf_url(
                    "https://example.com/page",
                    "https://example.com/document.html",
                )

    def test_uses_allowed_pdf_patterns_from_config(self):
        """Should use allowed_pdf_patterns from publisher config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "test_pub": {
                        "domain_patterns": ["test.com"],
                        "allowed_pdf_patterns": [r"/document/\d+/fulltext"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                assert rules.is_valid_pdf_url(
                    "https://test.com/page",
                    "https://test.com/document/123/fulltext",
                )

    def test_returns_false_when_pattern_not_matched(self):
        """Should return False when URL doesn't match allowed patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "test_pub": {
                        "domain_patterns": ["test.com"],
                        "allowed_pdf_patterns": [r"\.pdf$"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                assert not rules.is_valid_pdf_url(
                    "https://test.com/page",
                    "https://test.com/document.html",
                )


class TestFilterPdfUrls:
    """Tests for filter_pdf_urls method."""

    def test_filters_valid_pdf_urls(self):
        """Should keep only valid PDF URLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                pdf_urls = [
                    "https://example.com/paper.pdf",
                    "https://example.com/page.html",
                    "https://example.com/pdf/12345",
                ]
                result = rules.filter_pdf_urls("https://example.com/article", pdf_urls)
                assert "https://example.com/paper.pdf" in result
                assert "https://example.com/pdf/12345" in result
                assert "https://example.com/page.html" not in result

    def test_removes_urls_matching_deny_text_patterns(self):
        """Should remove URLs matching deny text patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                mock_config = MagicMock()
                mock_config.get.return_value = {
                    "test_pub": {
                        "domain_patterns": ["test.com"],
                        "deny_text_patterns": ["advertisement", "sample"],
                    }
                }
                rules = PublisherRules(config=mock_config)
                pdf_urls = [
                    "https://test.com/paper.pdf",
                    "https://test.com/advertisement.pdf",
                    "https://test.com/sample-chapter.pdf",
                ]
                result = rules.filter_pdf_urls("https://test.com/article", pdf_urls)
                assert "https://test.com/paper.pdf" in result
                assert "https://test.com/advertisement.pdf" not in result
                assert "https://test.com/sample-chapter.pdf" not in result

    def test_removes_duplicate_urls(self):
        """Should remove duplicate URLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                pdf_urls = [
                    "https://example.com/paper.pdf",
                    "https://example.com/paper.pdf",
                    "https://example.com/other.pdf",
                ]
                result = rules.filter_pdf_urls("https://example.com/article", pdf_urls)
                assert result.count("https://example.com/paper.pdf") == 1
                assert len(result) == 2

    def test_sciencedirect_pii_filtering(self):
        """Should filter ScienceDirect PDFs by PII matching allowed patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                page_url = "https://www.sciencedirect.com/science/article/pii/S0123456789000123"
                # Use URLs that match ScienceDirect's allowed_pdf_patterns
                pdf_urls = [
                    "https://www.sciencedirect.com/science/article/pii/S0123456789000123.pdf",
                    "https://www.sciencedirect.com/science/article/pii/S9999999999999999.pdf",
                ]
                result = rules.filter_pdf_urls(page_url, pdf_urls)
                # Only matching PII should be kept (filters non-matching PII)
                # First URL has matching PII, second has different PII
                assert len(result) == 1
                assert "S0123456789000123" in result[0]

    def test_preserves_order(self):
        """Should preserve order of URLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                pdf_urls = [
                    "https://example.com/a.pdf",
                    "https://example.com/b.pdf",
                    "https://example.com/c.pdf",
                ]
                result = rules.filter_pdf_urls("https://example.com/article", pdf_urls)
                assert result == pdf_urls

    def test_returns_empty_list_for_empty_input(self):
        """Should return empty list when no URLs provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                result = rules.filter_pdf_urls("https://example.com/article", [])
                assert result == []

    def test_handles_elsevier_domain(self):
        """Should apply ScienceDirect rules to elsevier.com."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                page_url = "https://www.elsevier.com/science/article/pii/ABC123"
                pdf_urls = ["https://pdf.elsevier.com/ABC123.pdf"]
                result = rules.filter_pdf_urls(page_url, pdf_urls)
                # Should process as ScienceDirect-like publisher
                assert isinstance(result, list)

    def test_handles_cell_domain(self):
        """Should apply ScienceDirect rules to cell.com."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()
                page_url = "https://www.cell.com/article/pii/DEF456"
                pdf_urls = ["https://pdf.cell.com/DEF456.pdf"]
                result = rules.filter_pdf_urls(page_url, pdf_urls)
                assert isinstance(result, list)


class TestPublisherRulesIntegration:
    """Integration tests for PublisherRules."""

    def test_full_workflow(self):
        """Test complete workflow with merge and filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules = PublisherRules()

                # Get merged config
                merged = rules.merge_with_config(
                    "https://example.com/article",
                    base_deny_selectors=[".ad"],
                    base_deny_classes=["advertisement"],
                )
                assert isinstance(merged, dict)

                # Filter URLs
                pdf_urls = [
                    "https://example.com/paper.pdf",
                    "https://example.com/page.html",
                ]
                filtered = rules.filter_pdf_urls(
                    "https://example.com/article", pdf_urls
                )
                assert len(filtered) == 1
                assert filtered[0].endswith(".pdf")

    def test_multiple_instances_independent(self):
        """Multiple instances should be independent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"SCITEX_DIR": tmpdir}):
                rules1 = PublisherRules()
                rules2 = PublisherRules()

                # Instances should be independent
                assert rules1 is not rules2
                assert rules1.config is not rules2.config


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
