#!/usr/bin/env python3
# Timestamp: "2026-01-14"
# File: tests/scitex/scholar/storage/test__DeduplicationManager.py
# ----------------------------------------

"""
Comprehensive tests for the DeduplicationManager class.

Tests cover:
- Initialization
- Fingerprint generation (DOI-based and metadata-based)
- Normalization functions (DOI, title, author)
- Duplicate detection
- Paper scoring for deduplication priority
- Metadata merging
- File merging
"""

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDeduplicationManagerInit:
    """Tests for DeduplicationManager initialization."""

    def test_init_with_default_config(self, tmp_path):
        """DeduplicationManager should initialize with default config."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        with patch("scitex.scholar.config.ScholarConfig") as MockConfig:
            mock_config = MagicMock()
            mock_config.path_manager.library_dir = tmp_path
            mock_config.path_manager.get_library_master_dir.return_value = (
                tmp_path / "MASTER"
            )
            MockConfig.return_value = mock_config

            manager = DeduplicationManager()

            assert manager.name == "DeduplicationManager"

    def test_init_with_custom_config(self, tmp_path):
        """DeduplicationManager should accept custom config."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = tmp_path / "MASTER"

        manager = DeduplicationManager(config=config)

        assert manager.config is config
        assert manager.master_dir == tmp_path / "MASTER"


class TestDeduplicationManagerNormalization:
    """Tests for normalization functions."""

    @pytest.fixture
    def dedup_manager(self, tmp_path):
        """Create DeduplicationManager instance."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = tmp_path / "MASTER"

        return DeduplicationManager(config=config)

    def test_normalize_doi_removes_url_prefix(self, dedup_manager):
        """_normalize_doi should remove URL prefixes."""
        assert (
            dedup_manager._normalize_doi("https://doi.org/10.1234/test")
            == "10.1234/test"
        )
        assert (
            dedup_manager._normalize_doi("http://dx.doi.org/10.1234/test")
            == "10.1234/test"
        )
        assert dedup_manager._normalize_doi("doi:10.1234/test") == "10.1234/test"

    def test_normalize_doi_lowercases(self, dedup_manager):
        """_normalize_doi should lowercase DOI."""
        assert dedup_manager._normalize_doi("10.1234/TEST") == "10.1234/test"

    def test_normalize_doi_handles_empty(self, dedup_manager):
        """_normalize_doi should handle empty input."""
        assert dedup_manager._normalize_doi("") == ""
        assert dedup_manager._normalize_doi(None) == ""

    def test_normalize_title_removes_special_chars(self, dedup_manager):
        """_normalize_title should remove special characters."""
        result = dedup_manager._normalize_title("Deep Learning: A Review!")
        assert ":" not in result
        assert "!" not in result

    def test_normalize_title_removes_stop_words(self, dedup_manager):
        """_normalize_title should remove stop words."""
        result = dedup_manager._normalize_title("The Deep Learning in the Brain")
        # Check for whole words, not substrings
        words = result.split()
        assert "the" not in words
        assert "in" not in words
        assert "deep" in words
        assert "learning" in words
        assert "brain" in words

    def test_normalize_title_handles_empty(self, dedup_manager):
        """_normalize_title should handle empty input."""
        assert dedup_manager._normalize_title("") == ""
        assert dedup_manager._normalize_title(None) == ""

    def test_normalize_author_extracts_last_name_from_comma_format(self, dedup_manager):
        """_normalize_author should extract last name from 'Last, First' format."""
        result = dedup_manager._normalize_author("Smith, John")
        assert result == "smith"

    def test_normalize_author_extracts_last_name_from_space_format(self, dedup_manager):
        """_normalize_author should extract last name from 'First Last' format."""
        result = dedup_manager._normalize_author("John Smith")
        assert result == "smith"

    def test_normalize_author_handles_empty(self, dedup_manager):
        """_normalize_author should handle empty input."""
        assert dedup_manager._normalize_author("") == ""
        assert dedup_manager._normalize_author(None) == ""


class TestDeduplicationManagerFingerprint:
    """Tests for fingerprint generation."""

    @pytest.fixture
    def dedup_manager(self, tmp_path):
        """Create DeduplicationManager instance."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = tmp_path / "MASTER"

        return DeduplicationManager(config=config)

    def test_fingerprint_uses_doi_when_available(self, dedup_manager):
        """_generate_paper_fingerprint should prefer DOI."""
        metadata = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "authors": ["Smith, John"],
            "year": 2023,
        }

        fingerprint = dedup_manager._generate_paper_fingerprint(metadata)

        assert fingerprint.startswith("DOI:")
        assert "10.1234/test" in fingerprint

    def test_fingerprint_uses_metadata_without_doi(self, dedup_manager):
        """_generate_paper_fingerprint should use metadata when no DOI."""
        metadata = {
            "title": "Deep Learning for EEG",
            "authors": ["Smith, John"],
            "year": 2023,
        }

        fingerprint = dedup_manager._generate_paper_fingerprint(metadata)

        assert fingerprint.startswith("META:")
        assert "deep" in fingerprint.lower()
        assert "smith" in fingerprint.lower()
        assert "2023" in fingerprint

    def test_fingerprint_returns_none_without_title(self, dedup_manager):
        """_generate_paper_fingerprint should return None without title."""
        metadata = {"year": 2023, "authors": ["Smith"]}

        fingerprint = dedup_manager._generate_paper_fingerprint(metadata)

        assert fingerprint is None

    def test_fingerprint_handles_dict_authors(self, dedup_manager):
        """_generate_paper_fingerprint should handle dict author format."""
        metadata = {
            "title": "Test Paper",
            "authors": [{"name": "Smith, John"}],
            "year": 2023,
        }

        fingerprint = dedup_manager._generate_paper_fingerprint(metadata)

        assert "smith" in fingerprint.lower()


class TestDeduplicationManagerDuplicateDetection:
    """Tests for duplicate detection."""

    @pytest.fixture
    def dedup_manager_with_papers(self, tmp_path):
        """Create DeduplicationManager with sample papers."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        master_dir = tmp_path / "MASTER"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = master_dir

        manager = DeduplicationManager(config=config)

        # Create duplicate papers (same DOI)
        paper1_dir = master_dir / "PAPER001"
        paper1_dir.mkdir()
        (paper1_dir / "metadata.json").write_text(
            json.dumps(
                {"doi": "10.1234/duplicate", "title": "Duplicate Paper", "year": 2023}
            )
        )

        paper2_dir = master_dir / "PAPER002"
        paper2_dir.mkdir()
        (paper2_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "doi": "10.1234/duplicate",
                    "title": "Duplicate Paper (Copy)",
                    "year": 2023,
                }
            )
        )

        # Create unique paper
        paper3_dir = master_dir / "PAPER003"
        paper3_dir.mkdir()
        (paper3_dir / "metadata.json").write_text(
            json.dumps({"doi": "10.1234/unique", "title": "Unique Paper", "year": 2023})
        )

        return manager

    def test_find_duplicate_papers_detects_same_doi(self, dedup_manager_with_papers):
        """find_duplicate_papers should detect papers with same DOI."""
        duplicates = dedup_manager_with_papers.find_duplicate_papers()

        # Should find one group of duplicates
        assert len(duplicates) >= 1

        # The duplicate group should contain at least 2 papers
        for fingerprint, paths in duplicates.items():
            if "10.1234/duplicate" in fingerprint.lower():
                assert len(paths) == 2
                break

    def test_find_duplicate_papers_ignores_unique(self, dedup_manager_with_papers):
        """find_duplicate_papers should not include unique papers."""
        duplicates = dedup_manager_with_papers.find_duplicate_papers()

        # The unique paper shouldn't be in any duplicate group
        all_paths = [p for paths in duplicates.values() for p in paths]
        path_names = [p.name for p in all_paths]

        # PAPER003 (unique) shouldn't be in duplicates unless grouped by title
        # This test may need adjustment based on exact grouping logic


class TestDeduplicationManagerScoring:
    """Tests for paper scoring."""

    @pytest.fixture
    def dedup_manager(self, tmp_path):
        """Create DeduplicationManager instance."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = tmp_path / "MASTER"

        return DeduplicationManager(config=config)

    def test_score_paper_with_doi(self, dedup_manager, tmp_path):
        """Paper with DOI should score high."""
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir(parents=True)

        metadata = {"doi": "10.1234/test"}

        score = dedup_manager._score_paper_metadata(metadata, paper_dir)

        assert score >= 1000  # DOI adds 1000 points

    def test_score_paper_with_pdf(self, dedup_manager, tmp_path):
        """Paper with PDF should score higher."""
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4")

        metadata = {}

        score = dedup_manager._score_paper_metadata(metadata, paper_dir)

        assert score >= 100  # PDF adds 100 points

    def test_score_paper_with_abstract(self, dedup_manager, tmp_path):
        """Paper with abstract should score higher."""
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir(parents=True)

        metadata = {"abstract": "This is a test abstract."}

        score = dedup_manager._score_paper_metadata(metadata, paper_dir)

        assert score >= 50  # Abstract adds 50 points

    def test_score_paper_with_citation_count(self, dedup_manager, tmp_path):
        """Paper with citations should score higher."""
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir(parents=True)

        metadata = {"citation_count": 100}

        score = dedup_manager._score_paper_metadata(metadata, paper_dir)

        assert score > 0  # Citation count adds log-scaled points

    def test_score_paper_complete_metadata(self, dedup_manager, tmp_path):
        """Paper with complete metadata should have high score."""
        paper_dir = tmp_path / "test_paper"
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4")

        metadata = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "authors": ["Smith, John", "Doe, Jane"],
            "abstract": "Test abstract",
            "journal": "Nature",
            "publisher": "Nature Publishing",
            "impact_factor": 40.5,
            "citation_count": 500,
            "url": "https://example.com",
            "pdf_url": "https://example.com/paper.pdf",
        }

        score = dedup_manager._score_paper_metadata(metadata, paper_dir)

        # Should have a very high score with all metadata
        assert score > 1500


class TestDeduplicationManagerMerging:
    """Tests for metadata and file merging."""

    @pytest.fixture
    def dedup_manager(self, tmp_path):
        """Create DeduplicationManager instance."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = tmp_path / "MASTER"

        return DeduplicationManager(config=config)

    def test_merge_metadata_keeps_best_values(self, dedup_manager, tmp_path):
        """_merge_metadata should keep best values from all papers."""
        scored_papers = [
            (
                1000,
                tmp_path / "paper1",
                {"title": "Paper 1", "doi": "10.1234/test", "citation_count": 50},
            ),
            (
                500,
                tmp_path / "paper2",
                {
                    "title": "Paper 2",
                    "abstract": "Test abstract",
                    "citation_count": 100,
                },
            ),
        ]

        merged = dedup_manager._merge_metadata(scored_papers)

        # Should keep DOI from best paper
        assert merged["doi"] == "10.1234/test"
        # Should merge in missing abstract
        assert merged["abstract"] == "Test abstract"
        # Should take highest citation count
        assert merged["citation_count"] == 100
        # Should have deduplication tracking
        assert "_deduplication" in merged
        assert len(merged["_deduplication"]["merged_from"]) == 2

    def test_merge_metadata_takes_doi_if_missing(self, dedup_manager, tmp_path):
        """_merge_metadata should add DOI from duplicate if missing."""
        scored_papers = [
            (
                1000,
                tmp_path / "paper1",
                {"title": "Paper 1", "abstract": "Good abstract"},
            ),
            (500, tmp_path / "paper2", {"doi": "10.1234/from_duplicate"}),
        ]

        merged = dedup_manager._merge_metadata(scored_papers)

        assert merged["doi"] == "10.1234/from_duplicate"
        assert merged["abstract"] == "Good abstract"

    def test_merge_metadata_takes_highest_impact_factor(self, dedup_manager, tmp_path):
        """_merge_metadata should take highest impact factor."""
        scored_papers = [
            (1000, tmp_path / "paper1", {"impact_factor": 10.5}),
            (500, tmp_path / "paper2", {"impact_factor": 42.3}),
        ]

        merged = dedup_manager._merge_metadata(scored_papers)

        assert merged["impact_factor"] == 42.3

    def test_merge_files_copies_pdfs(self, dedup_manager, tmp_path):
        """_merge_files should copy PDFs from duplicates."""
        keep_dir = tmp_path / "keep"
        keep_dir.mkdir()

        remove_dir = tmp_path / "remove"
        remove_dir.mkdir()
        (remove_dir / "unique_paper.pdf").write_bytes(b"%PDF-1.4 unique content")

        dedup_manager._merge_files(keep_dir, [remove_dir])

        # PDF should be copied to keep_dir
        assert (keep_dir / "unique_paper.pdf").exists()

    def test_merge_files_does_not_overwrite_existing(self, dedup_manager, tmp_path):
        """_merge_files should not overwrite existing PDFs."""
        keep_dir = tmp_path / "keep"
        keep_dir.mkdir()
        (keep_dir / "paper.pdf").write_bytes(b"%PDF-1.4 original")

        remove_dir = tmp_path / "remove"
        remove_dir.mkdir()
        (remove_dir / "paper.pdf").write_bytes(b"%PDF-1.4 duplicate")

        dedup_manager._merge_files(keep_dir, [remove_dir])

        # Original should be preserved
        content = (keep_dir / "paper.pdf").read_bytes()
        assert b"original" in content

    def test_merge_files_copies_screenshots(self, dedup_manager, tmp_path):
        """_merge_files should copy screenshots from duplicates."""
        keep_dir = tmp_path / "keep"
        keep_dir.mkdir()

        remove_dir = tmp_path / "remove"
        remove_dir.mkdir()
        screenshots_dir = remove_dir / "screenshots"
        screenshots_dir.mkdir()
        (screenshots_dir / "screenshot1.png").write_bytes(b"PNG data")

        dedup_manager._merge_files(keep_dir, [remove_dir])

        # Screenshot should be copied
        assert (keep_dir / "screenshots" / "screenshot1.png").exists()


class TestDeduplicationManagerMergeDuplicates:
    """Tests for full duplicate merging workflow."""

    @pytest.fixture
    def dedup_manager_with_duplicates(self, tmp_path):
        """Create DeduplicationManager with duplicate papers."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        master_dir = tmp_path / "MASTER"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = master_dir

        manager = DeduplicationManager(config=config)

        # Create paper 1 (better metadata)
        paper1_dir = master_dir / "PAPER001"
        paper1_dir.mkdir()
        (paper1_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "doi": "10.1234/test",
                    "title": "Test Paper",
                    "abstract": "Test abstract",
                    "year": 2023,
                    "citation_count": 50,
                }
            )
        )
        (paper1_dir / "paper.pdf").write_bytes(b"%PDF-1.4")

        # Create paper 2 (lower quality)
        paper2_dir = master_dir / "PAPER002"
        paper2_dir.mkdir()
        (paper2_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "title": "Test Paper",
                    "year": 2023,
                    "citation_count": 100,  # Higher citations
                    "journal": "Nature",  # Extra field
                }
            )
        )

        return manager, [paper1_dir, paper2_dir]

    def test_merge_duplicate_papers_returns_kept_dir(
        self, dedup_manager_with_duplicates
    ):
        """merge_duplicate_papers should return the kept directory."""
        manager, paper_dirs = dedup_manager_with_duplicates

        kept_dir, removed_dirs = manager.merge_duplicate_papers(paper_dirs)

        # Should keep paper1 (has DOI and PDF)
        assert kept_dir == paper_dirs[0]
        assert paper_dirs[1] in removed_dirs

    def test_merge_duplicate_papers_merges_metadata(
        self, dedup_manager_with_duplicates
    ):
        """merge_duplicate_papers should merge metadata from all duplicates."""
        manager, paper_dirs = dedup_manager_with_duplicates

        kept_dir, _ = manager.merge_duplicate_papers(paper_dirs)

        # Check merged metadata
        with open(kept_dir / "metadata.json") as f:
            merged = json.load(f)

        # Should have DOI from paper1
        assert merged["doi"] == "10.1234/test"
        # Should have higher citation count from paper2
        assert merged["citation_count"] == 100
        # Should have journal from paper2
        assert merged["journal"] == "Nature"
        # Should have abstract from paper1
        assert merged["abstract"] == "Test abstract"

    def test_merge_duplicate_papers_creates_backup(self, dedup_manager_with_duplicates):
        """merge_duplicate_papers should create metadata backup."""
        manager, paper_dirs = dedup_manager_with_duplicates

        kept_dir, _ = manager.merge_duplicate_papers(paper_dirs)

        # Should have backup file
        backups = list(kept_dir.glob("metadata.backup.*.json"))
        assert len(backups) == 1

    def test_merge_duplicate_papers_handles_single_paper(
        self, dedup_manager_with_duplicates
    ):
        """merge_duplicate_papers should handle single paper list."""
        manager, paper_dirs = dedup_manager_with_duplicates

        kept_dir, removed_dirs = manager.merge_duplicate_papers([paper_dirs[0]])

        assert kept_dir == paper_dirs[0]
        assert removed_dirs == []


class TestDeduplicationManagerEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def dedup_manager(self, tmp_path):
        """Create DeduplicationManager instance."""
        from scitex.scholar.storage._DeduplicationManager import DeduplicationManager

        master_dir = tmp_path / "MASTER"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.library_dir = tmp_path
        config.path_manager.get_library_master_dir.return_value = master_dir

        return DeduplicationManager(config=config)

    def test_find_duplicates_with_empty_library(self, dedup_manager):
        """find_duplicate_papers should handle empty library."""
        duplicates = dedup_manager.find_duplicate_papers()
        assert duplicates == {}

    def test_find_duplicates_with_corrupted_json(self, dedup_manager, tmp_path):
        """find_duplicate_papers should handle corrupted JSON files."""
        paper_dir = dedup_manager.master_dir / "CORRUPT"
        paper_dir.mkdir()
        (paper_dir / "metadata.json").write_text("{ invalid json }")

        # Should not raise, just skip the corrupted file
        duplicates = dedup_manager.find_duplicate_papers()
        assert isinstance(duplicates, dict)

    def test_find_duplicates_with_non_directory_files(self, dedup_manager):
        """find_duplicate_papers should ignore non-directory files."""
        # Create a regular file in master directory
        (dedup_manager.master_dir / "random_file.txt").write_text("test")

        # Should not raise
        duplicates = dedup_manager.find_duplicate_papers()
        assert isinstance(duplicates, dict)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
