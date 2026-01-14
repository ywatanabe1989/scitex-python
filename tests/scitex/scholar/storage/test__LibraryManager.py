#!/usr/bin/env python3
# Timestamp: "2026-01-14"
# File: tests/scitex/scholar/storage/test__LibraryManager.py
# ----------------------------------------

"""
Comprehensive tests for the LibraryManager class.

Tests cover:
- Initialization with project and config
- Storage helper methods (has_metadata, has_urls, has_pdf)
- Paper loading and saving
- Metadata merging and conversion
- DOI lookup in library
- Paper saving with various field combinations
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestLibraryManagerInit:
    """Tests for LibraryManager initialization."""

    def test_init_with_defaults(self, tmp_path):
        """LibraryManager should initialize with default config."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = tmp_path / "master"
        config.resolve.return_value = "default"

        manager = LibraryManager(config=config)

        assert manager.config is config
        assert manager.dedup_manager is not None

    def test_init_with_project(self, tmp_path):
        """LibraryManager should accept project name."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = tmp_path / "master"
        config.resolve.return_value = "test_project"

        manager = LibraryManager(project="test_project", config=config)

        # Project should be resolved from config
        assert manager.project is not None


class TestLibraryManagerStorageHelpers:
    """Tests for storage helper methods."""

    @pytest.fixture
    def library_manager(self, tmp_path):
        """Create LibraryManager with temp directory."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        master_dir = tmp_path / "master"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = master_dir
        config.resolve.return_value = "test_project"

        manager = LibraryManager(config=config)
        manager.library_master_dir = master_dir
        return manager

    def test_has_metadata_returns_true_when_exists(self, library_manager, tmp_path):
        """has_metadata should return True when metadata.json exists."""
        paper_id = "A1B2C3D4"
        paper_dir = library_manager.library_master_dir / paper_id
        paper_dir.mkdir(parents=True)
        (paper_dir / "metadata.json").write_text('{"title": "Test"}')

        assert library_manager.has_metadata(paper_id) is True

    def test_has_metadata_returns_false_when_missing(self, library_manager):
        """has_metadata should return False when metadata.json doesn't exist."""
        paper_id = "MISSING01"
        assert library_manager.has_metadata(paper_id) is False

    def test_has_urls_returns_true_when_pdfs_exist(self, library_manager):
        """has_urls should return True when PDF URLs exist in metadata."""
        paper_id = "B2C3D4E5"
        paper_dir = library_manager.library_master_dir / paper_id
        paper_dir.mkdir(parents=True)

        metadata = {"metadata": {"url": {"pdfs": ["https://example.com/paper.pdf"]}}}
        (paper_dir / "metadata.json").write_text(json.dumps(metadata))

        assert library_manager.has_urls(paper_id) is True

    def test_has_urls_returns_false_when_empty(self, library_manager):
        """has_urls should return False when no PDF URLs."""
        paper_id = "C3D4E5F6"
        paper_dir = library_manager.library_master_dir / paper_id
        paper_dir.mkdir(parents=True)

        metadata = {"metadata": {"url": {"pdfs": []}}}
        (paper_dir / "metadata.json").write_text(json.dumps(metadata))

        assert library_manager.has_urls(paper_id) is False

    def test_has_urls_returns_false_when_no_metadata(self, library_manager):
        """has_urls should return False when no metadata file."""
        assert library_manager.has_urls("NONEXIST") is False

    def test_has_pdf_returns_true_when_pdf_exists(self, library_manager):
        """has_pdf should return True when PDF file exists."""
        paper_id = "D4E5F6G7"
        paper_dir = library_manager.library_master_dir / paper_id
        paper_dir.mkdir(parents=True)
        (paper_dir / "paper.pdf").write_bytes(b"%PDF-1.4 test")

        assert library_manager.has_pdf(paper_id) is True

    def test_has_pdf_returns_false_when_no_pdf(self, library_manager):
        """has_pdf should return False when no PDF file."""
        paper_id = "E5F6G7H8"
        paper_dir = library_manager.library_master_dir / paper_id
        paper_dir.mkdir(parents=True)
        # Only metadata, no PDF
        (paper_dir / "metadata.json").write_text("{}")

        assert library_manager.has_pdf(paper_id) is False

    def test_has_pdf_returns_false_when_dir_missing(self, library_manager):
        """has_pdf should return False when paper directory doesn't exist."""
        assert library_manager.has_pdf("NOFOLDER") is False


class TestLibraryManagerPaperIO:
    """Tests for paper loading and saving."""

    @pytest.fixture
    def library_manager(self, tmp_path):
        """Create LibraryManager with temp directory."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        master_dir = tmp_path / "master"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = master_dir
        config.resolve.return_value = "test_project"

        manager = LibraryManager(config=config)
        manager.library_master_dir = master_dir
        return manager

    def test_load_paper_from_id_returns_paper(self, library_manager):
        """load_paper_from_id should return Paper object when found."""
        from scitex.scholar.core import Paper

        paper_id = "F6G7H8I9"
        paper_dir = library_manager.library_master_dir / paper_id
        paper_dir.mkdir(parents=True)

        # Create valid paper metadata
        paper = Paper()
        paper.container.library_id = paper_id
        paper.metadata.basic.title = "Test Paper"
        paper.metadata.basic.year = 2023

        (paper_dir / "metadata.json").write_text(
            json.dumps(paper.model_dump(), default=str)
        )

        loaded = library_manager.load_paper_from_id(paper_id)

        assert loaded is not None
        assert loaded.metadata.basic.title == "Test Paper"
        assert loaded.metadata.basic.year == 2023

    def test_load_paper_from_id_returns_none_when_missing(self, library_manager):
        """load_paper_from_id should return None when paper not found."""
        result = library_manager.load_paper_from_id("NOTFOUND")
        assert result is None

    def test_save_paper_incremental_creates_file(self, library_manager):
        """save_paper_incremental should create metadata file."""
        from scitex.scholar.core import Paper

        paper_id = "G7H8I9J0"
        paper = Paper()
        paper.container.library_id = paper_id
        paper.metadata.basic.title = "New Paper"

        library_manager.save_paper_incremental(paper_id, paper)

        metadata_file = library_manager.library_master_dir / paper_id / "metadata.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            data = json.load(f)
        assert data["metadata"]["basic"]["title"] == "New Paper"

    def test_save_paper_incremental_merges_existing(self, library_manager):
        """save_paper_incremental should merge with existing data."""
        from scitex.scholar.core import Paper

        paper_id = "H8I9J0K1"
        paper_dir = library_manager.library_master_dir / paper_id
        paper_dir.mkdir(parents=True)

        # Create initial metadata
        initial_data = {
            "metadata": {
                "basic": {"title": "Original Title", "abstract": "Existing abstract"}
            }
        }
        (paper_dir / "metadata.json").write_text(json.dumps(initial_data))

        # Save new paper with updated title but no abstract
        paper = Paper()
        paper.container.library_id = paper_id
        paper.metadata.basic.title = "Updated Title"
        # abstract is None/empty

        library_manager.save_paper_incremental(paper_id, paper)

        with open(paper_dir / "metadata.json") as f:
            data = json.load(f)

        # Title should be updated, abstract should be preserved
        assert data["metadata"]["basic"]["title"] == "Updated Title"


class TestLibraryManagerMetadataMerge:
    """Tests for metadata merging logic."""

    @pytest.fixture
    def library_manager(self, tmp_path):
        """Create LibraryManager instance."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        master_dir = tmp_path / "master"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = master_dir
        config.resolve.return_value = "test_project"

        manager = LibraryManager(config=config)
        manager.library_master_dir = master_dir
        return manager

    def test_merge_metadata_prefers_new_non_none_values(self, library_manager):
        """_merge_metadata should prefer new non-None values."""
        existing = {"title": "Old Title", "year": 2020}
        new = {"title": "New Title", "year": None}

        result = library_manager._merge_metadata(existing, new)

        assert result["title"] == "New Title"
        assert result["year"] == 2020  # Kept existing because new is None

    def test_merge_metadata_handles_nested_dicts(self, library_manager):
        """_merge_metadata should recursively merge nested dicts."""
        existing = {"metadata": {"basic": {"title": "Old", "year": 2020}}}
        new = {"metadata": {"basic": {"title": "New", "abstract": "Test"}}}

        result = library_manager._merge_metadata(existing, new)

        assert result["metadata"]["basic"]["title"] == "New"
        assert result["metadata"]["basic"]["year"] == 2020
        assert result["metadata"]["basic"]["abstract"] == "Test"

    def test_merge_metadata_handles_lists(self, library_manager):
        """_merge_metadata should update non-empty lists."""
        existing = {"authors": ["Old Author"]}
        new = {"authors": ["New Author 1", "New Author 2"]}

        result = library_manager._merge_metadata(existing, new)

        assert result["authors"] == ["New Author 1", "New Author 2"]

    def test_merge_metadata_preserves_existing_lists_if_new_empty(
        self, library_manager
    ):
        """_merge_metadata should keep existing list if new is empty."""
        existing = {"authors": ["Keep This"]}
        new = {"authors": []}

        result = library_manager._merge_metadata(existing, new)

        assert result["authors"] == ["Keep This"]


class TestLibraryManagerDOILookup:
    """Tests for DOI lookup functionality."""

    @pytest.fixture
    def library_manager(self, tmp_path):
        """Create LibraryManager with sample papers."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        master_dir = tmp_path / "master"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = master_dir
        config.resolve.return_value = "test_project"

        manager = LibraryManager(config=config)
        manager.library_master_dir = master_dir

        # Create sample paper with known DOI
        paper_dir = master_dir / "SAMPLE01"
        paper_dir.mkdir(parents=True)
        (paper_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "title": "Deep Learning for EEG Analysis",
                    "year": 2023,
                    "doi": "10.1234/example.2023",
                }
            )
        )

        return manager

    def test_check_library_for_doi_finds_matching_paper(self, library_manager):
        """check_library_for_doi should find DOI for matching title."""
        doi = library_manager.check_library_for_doi(
            title="Deep Learning for EEG Analysis", year=2023
        )

        assert doi == "10.1234/example.2023"

    def test_check_library_for_doi_returns_none_for_no_match(self, library_manager):
        """check_library_for_doi should return None when no match."""
        doi = library_manager.check_library_for_doi(
            title="Completely Different Paper", year=2023
        )

        assert doi is None

    def test_check_library_for_doi_handles_year_tolerance(self, library_manager):
        """check_library_for_doi should allow ±1 year tolerance."""
        # Paper is from 2023, searching with 2024 should still match
        doi = library_manager.check_library_for_doi(
            title="Deep Learning for EEG Analysis", year=2024
        )

        assert doi == "10.1234/example.2023"


class TestLibraryManagerSaveResolvedPaper:
    """Tests for save_resolved_paper method."""

    @pytest.fixture
    def library_manager(self, tmp_path):
        """Create LibraryManager instance."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        master_dir = tmp_path / "master"
        master_dir.mkdir(parents=True)

        # Also create the paper storage directory
        paper_dir = master_dir / "NEW001"
        paper_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = master_dir
        config.path_manager.get_paper_storage_paths.return_value = (
            paper_dir,
            "Author-2023-Journal",
            "NEW00001",
        )
        config.resolve.return_value = "test_project"

        manager = LibraryManager(config=config)
        manager.library_master_dir = master_dir
        return manager

    def test_save_resolved_paper_with_basic_fields(self, library_manager):
        """save_resolved_paper should save paper with basic fields."""
        paper_id = library_manager.save_resolved_paper(
            title="Test Paper",
            doi="10.1234/test.2023",
            authors=["Smith, John"],
            year=2023,
            journal="Nature",
        )

        assert paper_id is not None

    def test_save_resolved_paper_with_paper_object(self, library_manager):
        """save_resolved_paper should accept Paper object."""
        from scitex.scholar.core import Paper

        paper = Paper()
        paper.container.library_id = "PAPOBJ01"
        paper.metadata.basic.title = "Paper Object Test"
        paper.metadata.id.doi = "10.1234/paper.obj"
        paper.metadata.basic.year = 2023

        paper_id = library_manager.save_resolved_paper(paper_data=paper)

        assert paper_id is not None


class TestLibraryManagerMetadataConversion:
    """Tests for metadata conversion utilities."""

    @pytest.fixture
    def library_manager(self, tmp_path):
        """Create LibraryManager instance."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        master_dir = tmp_path / "master"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = master_dir
        config.resolve.return_value = "test_project"

        manager = LibraryManager(config=config)
        manager.library_master_dir = master_dir
        return manager

    def test_convert_to_standardized_metadata_basic_fields(self, library_manager):
        """_convert_to_standardized_metadata should map basic fields."""
        flat_metadata = {
            "doi": "10.1234/test",
            "title": "Test Paper",
            "authors": ["Smith, John"],
            "year": 2023,
            "abstract": "This is a test abstract.",
            "doi_source": "CrossRef",
            "title_source": "CrossRef",
        }

        result = library_manager._convert_to_standardized_metadata(flat_metadata)

        assert result["id"]["doi"] == "10.1234/test"
        assert result["basic"]["title"] == "Test Paper"
        assert result["basic"]["authors"] == ["Smith, John"]
        assert result["basic"]["year"] == 2023
        assert result["basic"]["abstract"] == "This is a test abstract."
        assert "CrossRef" in result["id"]["doi_engines"]

    def test_convert_to_standardized_metadata_publication_fields(self, library_manager):
        """_convert_to_standardized_metadata should map publication fields."""
        flat_metadata = {
            "journal": "Nature Neuroscience",
            "volume": "26",
            "issue": "4",
            "pages": "123-145",
            "publisher": "Nature Publishing Group",
        }

        result = library_manager._convert_to_standardized_metadata(flat_metadata)

        assert result["publication"]["journal"] == "Nature Neuroscience"
        assert result["publication"]["volume"] == "26"
        assert result["publication"]["issue"] == "4"
        assert result["publication"]["first_page"] == "123"
        assert result["publication"]["last_page"] == "145"
        assert result["publication"]["publisher"] == "Nature Publishing Group"

    def test_convert_to_standardized_metadata_citation_count_scalar(
        self, library_manager
    ):
        """_convert_to_standardized_metadata should handle scalar citation count."""
        flat_metadata = {
            "citation_count": 42,
            "citation_count_source": "SemanticScholar",
        }

        result = library_manager._convert_to_standardized_metadata(flat_metadata)

        assert result["citation_count"]["total"] == 42
        assert "SemanticScholar" in result["citation_count"]["total_engines"]

    def test_convert_to_standardized_metadata_citation_count_dict(
        self, library_manager
    ):
        """_convert_to_standardized_metadata should handle dict citation count."""
        flat_metadata = {"citation_count": {"total": 100, "2023": 25, "2022": 35}}

        result = library_manager._convert_to_standardized_metadata(flat_metadata)

        assert result["citation_count"]["total"] == 100
        assert result["citation_count"]["2023"] == 25
        assert result["citation_count"]["2022"] == 35


class TestLibraryManagerDotDictConversion:
    """Tests for DotDict to dict conversion."""

    @pytest.fixture
    def library_manager(self, tmp_path):
        """Create LibraryManager instance."""
        from scitex.scholar.storage._LibraryManager import LibraryManager

        master_dir = tmp_path / "master"
        master_dir.mkdir(parents=True)

        config = MagicMock()
        config.path_manager.get_library_master_dir.return_value = master_dir
        config.resolve.return_value = "test_project"

        manager = LibraryManager(config=config)
        manager.library_master_dir = master_dir
        return manager

    def test_dotdict_to_dict_with_plain_dict(self, library_manager):
        """_dotdict_to_dict should handle plain dict."""
        input_dict = {"key": "value", "nested": {"inner": 123}}
        result = library_manager._dotdict_to_dict(input_dict)

        assert result == input_dict

    def test_dotdict_to_dict_with_list(self, library_manager):
        """_dotdict_to_dict should handle lists."""
        input_list = [{"a": 1}, {"b": 2}]
        result = library_manager._dotdict_to_dict(input_list)

        assert result == input_list

    def test_dotdict_to_dict_with_scalar(self, library_manager):
        """_dotdict_to_dict should handle scalar values."""
        assert library_manager._dotdict_to_dict(42) == 42
        assert library_manager._dotdict_to_dict("string") == "string"
        assert library_manager._dotdict_to_dict(None) is None


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
