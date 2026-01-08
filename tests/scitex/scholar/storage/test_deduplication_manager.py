#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/scholar/storage/test_deduplication_manager.py

"""Tests for DeduplicationManager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDeduplicationManager:
    """Test suite for DeduplicationManager."""

    @pytest.fixture
    def temp_library(self, tmp_path):
        """Create a temporary library structure."""
        library_dir = tmp_path / "library"
        master_dir = library_dir / "MASTER"
        master_dir.mkdir(parents=True)
        return library_dir, master_dir

    @pytest.fixture
    def mock_config(self, temp_library):
        """Create mock config pointing to temp library."""
        library_dir, master_dir = temp_library

        config = MagicMock()
        config.path_manager.library_dir = library_dir
        config.path_manager.get_library_master_dir.return_value = master_dir
        return config

    @pytest.fixture
    def manager(self, mock_config):
        """Create DeduplicationManager with mock config."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        return DeduplicationManager(config=mock_config)

    def create_paper(self, master_dir, paper_id, metadata):
        """Helper to create a paper directory with metadata."""
        paper_dir = master_dir / paper_id
        paper_dir.mkdir()
        with open(paper_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        return paper_dir

    # ================================================================
    # Fingerprint Generation Tests
    # ================================================================

    def test_fingerprint_with_doi(self, manager):
        """Test fingerprint generation with DOI."""
        metadata = {
            "doi": "10.1038/s41586-021-03819-2",
            "title": "Some Paper Title",
            "authors": [{"name": "Smith, John"}],
            "year": 2021,
        }
        fingerprint = manager._generate_paper_fingerprint(metadata)
        assert fingerprint.startswith("DOI:")
        assert "10.1038/s41586-021-03819-2" in fingerprint.lower()

    def test_fingerprint_with_doi_url_prefix(self, manager):
        """Test fingerprint strips DOI URL prefix."""
        metadata = {"doi": "https://doi.org/10.1038/s41586-021-03819-2"}
        fingerprint = manager._generate_paper_fingerprint(metadata)
        assert "doi.org" not in fingerprint
        assert "10.1038/s41586-021-03819-2" in fingerprint.lower()

    def test_fingerprint_without_doi(self, manager):
        """Test fingerprint fallback to title+author+year."""
        metadata = {
            "title": "Machine Learning for Neuroscience",
            "authors": [{"name": "Watanabe, Yusuke"}],
            "year": 2024,
        }
        fingerprint = manager._generate_paper_fingerprint(metadata)
        assert fingerprint.startswith("META:")
        assert "watanabe" in fingerprint.lower()
        assert "2024" in fingerprint

    def test_fingerprint_no_metadata(self, manager):
        """Test fingerprint with minimal metadata returns None."""
        metadata = {}
        fingerprint = manager._generate_paper_fingerprint(metadata)
        assert fingerprint is None

    def test_fingerprint_title_only(self, manager):
        """Test fingerprint with only title."""
        metadata = {"title": "Some Research Paper"}
        fingerprint = manager._generate_paper_fingerprint(metadata)
        assert fingerprint is not None
        assert fingerprint.startswith("META:")

    # ================================================================
    # Title Normalization Tests
    # ================================================================

    def test_normalize_title_lowercase(self, manager):
        """Test title normalization converts to lowercase."""
        result = manager._normalize_title("UPPERCASE TITLE")
        assert result == result.lower()

    def test_normalize_title_removes_punctuation(self, manager):
        """Test title normalization removes punctuation."""
        result = manager._normalize_title("Title: A Sub-title (2024)")
        assert ":" not in result
        assert "-" not in result
        assert "(" not in result

    def test_normalize_title_removes_stopwords(self, manager):
        """Test title normalization removes common words."""
        result = manager._normalize_title("The Analysis and the Brain")
        assert "the" not in result.split()
        assert "and" not in result.split()

    def test_normalize_title_same_titles_match(self, manager):
        """Test that similar titles normalize to same value."""
        title1 = "The Machine Learning Approach"
        title2 = "A Machine Learning Approach"
        title3 = "machine learning approach"

        norm1 = manager._normalize_title(title1)
        norm2 = manager._normalize_title(title2)
        norm3 = manager._normalize_title(title3)

        assert norm1 == norm2 == norm3

    # ================================================================
    # DOI Normalization Tests
    # ================================================================

    def test_normalize_doi_strips_https(self, manager):
        """Test DOI normalization strips https prefix."""
        result = manager._normalize_doi("https://doi.org/10.1234/test")
        assert result == "10.1234/test"

    def test_normalize_doi_strips_http(self, manager):
        """Test DOI normalization strips http prefix."""
        result = manager._normalize_doi("http://dx.doi.org/10.1234/test")
        assert result == "10.1234/test"

    def test_normalize_doi_lowercase(self, manager):
        """Test DOI normalization converts to lowercase."""
        result = manager._normalize_doi("10.1234/TEST")
        assert result == "10.1234/test"

    # ================================================================
    # Author Normalization Tests
    # ================================================================

    def test_normalize_author_last_first(self, manager):
        """Test author normalization with Last, First format."""
        result = manager._normalize_author("Smith, John")
        assert result == "smith"

    def test_normalize_author_first_last(self, manager):
        """Test author normalization with First Last format."""
        result = manager._normalize_author("John Smith")
        assert result == "smith"

    def test_normalize_author_single_name(self, manager):
        """Test author normalization with single name."""
        result = manager._normalize_author("Aristotle")
        assert result == "aristotle"

    # ================================================================
    # Duplicate Detection Tests
    # ================================================================

    def test_find_duplicates_by_doi(self, manager, temp_library):
        """Test finding duplicates by matching DOI."""
        _, master_dir = temp_library

        # Create two papers with same DOI
        self.create_paper(
            master_dir, "PAPER001", {"doi": "10.1038/test123", "title": "Paper One"}
        )
        self.create_paper(
            master_dir,
            "PAPER002",
            {"doi": "10.1038/test123", "title": "Paper One (Copy)"},
        )

        duplicates = manager.find_duplicate_papers()
        assert len(duplicates) == 1
        assert len(list(duplicates.values())[0]) == 2

    def test_find_duplicates_by_title(self, manager, temp_library):
        """Test finding duplicates by matching title."""
        _, master_dir = temp_library

        # Create two papers with same title but no DOI
        self.create_paper(
            master_dir,
            "PAPER001",
            {
                "title": "Machine Learning Analysis",
                "authors": [{"name": "Smith, John"}],
                "year": 2024,
            },
        )
        self.create_paper(
            master_dir,
            "PAPER002",
            {
                "title": "The Machine Learning Analysis",  # Same after normalization
                "authors": [{"name": "John Smith"}],
                "year": 2024,
            },
        )

        duplicates = manager.find_duplicate_papers()
        assert len(duplicates) == 1

    def test_no_duplicates(self, manager, temp_library):
        """Test no duplicates found for unique papers."""
        _, master_dir = temp_library

        self.create_paper(
            master_dir, "PAPER001", {"doi": "10.1038/paper1", "title": "First Paper"}
        )
        self.create_paper(
            master_dir, "PAPER002", {"doi": "10.1038/paper2", "title": "Second Paper"}
        )

        duplicates = manager.find_duplicate_papers()
        assert len(duplicates) == 0

    def test_find_duplicates_empty_library(self, manager, temp_library):
        """Test finding duplicates in empty library."""
        duplicates = manager.find_duplicate_papers()
        assert len(duplicates) == 0

    # ================================================================
    # Metadata Scoring Tests
    # ================================================================

    def test_score_paper_with_doi(self, manager, temp_library):
        """Test paper with DOI scores higher."""
        _, master_dir = temp_library
        paper_dir = self.create_paper(master_dir, "PAPER001", {})

        with_doi = {"doi": "10.1038/test"}
        without_doi = {"title": "Test Paper"}

        score_with = manager._score_paper_metadata(with_doi, paper_dir)
        score_without = manager._score_paper_metadata(without_doi, paper_dir)

        assert score_with > score_without

    def test_score_paper_with_abstract(self, manager, temp_library):
        """Test paper with abstract scores higher."""
        _, master_dir = temp_library
        paper_dir = self.create_paper(master_dir, "PAPER001", {})

        with_abstract = {"title": "Test", "abstract": "This is the abstract."}
        without_abstract = {"title": "Test"}

        score_with = manager._score_paper_metadata(with_abstract, paper_dir)
        score_without = manager._score_paper_metadata(without_abstract, paper_dir)

        assert score_with > score_without

    def test_score_paper_with_pdf(self, manager, temp_library):
        """Test paper with PDF file scores higher."""
        _, master_dir = temp_library

        paper_with_pdf = master_dir / "PAPER001"
        paper_with_pdf.mkdir()
        (paper_with_pdf / "paper.pdf").touch()

        paper_no_pdf = master_dir / "PAPER002"
        paper_no_pdf.mkdir()

        metadata = {"title": "Test"}

        score_with = manager._score_paper_metadata(metadata, paper_with_pdf)
        score_without = manager._score_paper_metadata(metadata, paper_no_pdf)

        assert score_with > score_without

    def test_score_paper_with_citations(self, manager, temp_library):
        """Test paper with citation count scores higher."""
        _, master_dir = temp_library
        paper_dir = self.create_paper(master_dir, "PAPER001", {})

        high_citations = {"title": "Test", "citation_count": 1000}
        low_citations = {"title": "Test", "citation_count": 10}
        no_citations = {"title": "Test"}

        score_high = manager._score_paper_metadata(high_citations, paper_dir)
        score_low = manager._score_paper_metadata(low_citations, paper_dir)
        score_none = manager._score_paper_metadata(no_citations, paper_dir)

        assert score_high > score_low > score_none

    # ================================================================
    # Metadata Merging Tests
    # ================================================================

    def test_merge_metadata_keeps_best_doi(self, manager, temp_library):
        """Test merging keeps DOI from any source."""
        _, master_dir = temp_library
        paper1 = master_dir / "P1"
        paper2 = master_dir / "P2"

        scored_papers = [
            (100, paper1, {"title": "Test", "doi": "10.1038/best"}),
            (50, paper2, {"title": "Test"}),
        ]

        merged = manager._merge_metadata(scored_papers)
        assert merged["doi"] == "10.1038/best"

    def test_merge_metadata_fills_missing(self, manager, temp_library):
        """Test merging fills in missing fields from duplicates."""
        _, master_dir = temp_library
        paper1 = master_dir / "P1"
        paper2 = master_dir / "P2"

        scored_papers = [
            (100, paper1, {"title": "Test", "year": 2024}),
            (
                50,
                paper2,
                {"title": "Test", "abstract": "This is abstract", "journal": "Nature"},
            ),
        ]

        merged = manager._merge_metadata(scored_papers)
        assert merged["year"] == 2024
        assert merged["abstract"] == "This is abstract"
        assert merged["journal"] == "Nature"

    def test_merge_metadata_takes_higher_citations(self, manager, temp_library):
        """Test merging takes higher citation count."""
        _, master_dir = temp_library
        paper1 = master_dir / "P1"
        paper2 = master_dir / "P2"

        scored_papers = [
            (100, paper1, {"title": "Test", "citation_count": 50}),
            (50, paper2, {"title": "Test", "citation_count": 200}),
        ]

        merged = manager._merge_metadata(scored_papers)
        assert merged["citation_count"] == 200

    def test_merge_metadata_tracks_sources(self, manager, temp_library):
        """Test merging tracks source papers."""
        _, master_dir = temp_library
        paper1 = master_dir / "P1"
        paper2 = master_dir / "P2"

        scored_papers = [
            (100, paper1, {"title": "Test"}),
            (50, paper2, {"title": "Test"}),
        ]

        merged = manager._merge_metadata(scored_papers)
        assert "_deduplication" in merged
        assert len(merged["_deduplication"]["merged_from"]) == 2

    # ================================================================
    # Check Existing Paper Tests
    # ================================================================

    def test_check_existing_by_doi(self, manager, temp_library):
        """Test finding existing paper by DOI."""
        _, master_dir = temp_library

        existing = self.create_paper(
            master_dir,
            "EXISTING01",
            {"doi": "10.1038/existing", "title": "Existing Paper"},
        )

        result = manager.check_for_existing_paper(
            {
                "doi": "10.1038/existing",
                "title": "Different Title",  # Should still match by DOI
            }
        )

        assert result is not None
        assert result == existing

    def test_check_existing_by_title(self, manager, temp_library):
        """Test finding existing paper by title."""
        _, master_dir = temp_library

        existing = self.create_paper(
            master_dir,
            "EXISTING01",
            {
                "title": "Machine Learning Study",
                "authors": [{"name": "Smith"}],
                "year": 2024,
            },
        )

        result = manager.check_for_existing_paper(
            {
                "title": "The Machine Learning Study",
                "authors": [{"name": "Smith"}],
                "year": 2024,
            }
        )

        assert result is not None
        assert result == existing

    def test_check_existing_not_found(self, manager, temp_library):
        """Test no match when paper doesn't exist."""
        _, master_dir = temp_library

        self.create_paper(
            master_dir,
            "EXISTING01",
            {"doi": "10.1038/different", "title": "Different Paper"},
        )

        result = manager.check_for_existing_paper(
            {"doi": "10.1038/new", "title": "New Paper"}
        )

        assert result is None


class TestDeduplicationIntegration:
    """Integration tests for deduplication workflow."""

    @pytest.fixture
    def temp_library(self, tmp_path):
        """Create a temporary library structure."""
        library_dir = tmp_path / "library"
        master_dir = library_dir / "MASTER"
        project_dir = library_dir / "test_project"
        master_dir.mkdir(parents=True)
        project_dir.mkdir(parents=True)
        return library_dir, master_dir, project_dir

    @pytest.fixture
    def mock_config(self, temp_library):
        """Create mock config pointing to temp library."""
        library_dir, master_dir, _ = temp_library

        config = MagicMock()
        config.path_manager.library_dir = library_dir
        config.path_manager.get_library_master_dir.return_value = master_dir
        return config

    def create_paper_with_pdf(
        self, master_dir, paper_id, metadata, pdf_name="paper.pdf"
    ):
        """Helper to create a paper with PDF."""
        paper_dir = master_dir / paper_id
        paper_dir.mkdir()
        with open(paper_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
        (paper_dir / pdf_name).write_text("dummy pdf content")
        return paper_dir

    def test_full_deduplication_workflow_dry_run(self, mock_config, temp_library):
        """Test complete deduplication workflow in dry run mode."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        library_dir, master_dir, _ = temp_library
        manager = DeduplicationManager(config=mock_config)

        # Create duplicate papers
        self.create_paper_with_pdf(
            master_dir,
            "PAPER001",
            {"doi": "10.1038/test123", "title": "Test Paper", "citation_count": 100},
        )
        self.create_paper_with_pdf(
            master_dir,
            "PAPER002",
            {
                "doi": "10.1038/test123",
                "title": "Test Paper Copy",
                "citation_count": 50,
            },
        )

        # Run dry run
        stats = manager.deduplicate_library(dry_run=True)

        assert stats["groups_found"] == 1
        assert stats["duplicates_found"] == 1
        # Dry run should not remove anything
        assert stats["dirs_removed"] == 0

        # Both papers should still exist
        assert (master_dir / "PAPER001").exists()
        assert (master_dir / "PAPER002").exists()

    def test_merge_keeps_all_pdfs(self, mock_config, temp_library):
        """Test that merging keeps PDFs from all duplicates."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        library_dir, master_dir, _ = temp_library
        manager = DeduplicationManager(config=mock_config)

        # Create papers with different PDFs
        paper1 = self.create_paper_with_pdf(
            master_dir,
            "PAPER001",
            {"doi": "10.1038/test123", "title": "Test Paper", "citation_count": 100},
            pdf_name="paper_v1.pdf",
        )

        paper2 = self.create_paper_with_pdf(
            master_dir,
            "PAPER002",
            {"doi": "10.1038/test123", "title": "Test Paper", "citation_count": 50},
            pdf_name="paper_v2.pdf",
        )

        # Merge papers
        keep_dir, remove_dirs = manager.merge_duplicate_papers([paper1, paper2])

        # Higher citation count paper should be kept
        assert keep_dir == paper1

        # Both PDFs should be in kept directory
        pdfs = list(keep_dir.glob("*.pdf"))
        assert len(pdfs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
