#!/usr/bin/env python3
"""Tests for LibraryCacheManager class."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.storage._LibraryCacheManager import LibraryCacheManager


class TestLibraryCacheManagerInit:
    """Tests for LibraryCacheManager initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default config when none provided."""
        with patch(
            "scitex.scholar.storage._LibraryCacheManager.ScholarConfig"
        ) as MockConfig:
            mock_config = MagicMock()
            mock_config.resolve.return_value = "default_project"
            MockConfig.return_value = mock_config

            manager = LibraryCacheManager()
            assert manager.config == mock_config
            assert manager.project == "default_project"

    def test_init_with_project(self):
        """Should initialize with specified project."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"

        manager = LibraryCacheManager(project="test_project", config=mock_config)
        assert manager.project == "test_project"
        mock_config.resolve.assert_called_with("project", "test_project")

    def test_init_with_config(self):
        """Should use provided config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "project"

        manager = LibraryCacheManager(config=mock_config)
        assert manager.config == mock_config


class TestIsDOIStored:
    """Tests for is_doi_stored method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_returns_none_for_empty_title(self, manager):
        """Should return None for empty title."""
        result = manager.is_doi_stored("")
        assert result is None

    def test_returns_none_for_none_title(self, manager):
        """Should return None for None title."""
        result = manager.is_doi_stored(None)
        assert result is None

    def test_returns_none_when_master_dir_not_exists(self, manager):
        """Should return None when master directory doesn't exist."""
        manager.config.get_library_master_dir.return_value = Path("/nonexistent")
        result = manager.is_doi_stored("Test Paper Title")
        assert result is None

    def test_finds_doi_with_exact_title_match(self, manager):
        """Should find DOI when title matches exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            master_dir = Path(tmpdir)
            paper_dir = master_dir / "12345678"
            paper_dir.mkdir()

            metadata = {
                "title": "Test Paper Title",
                "doi": "10.1038/nature12373",
                "year": 2023,
            }
            with open(paper_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            manager.config.get_library_master_dir.return_value = master_dir
            result = manager.is_doi_stored("Test Paper Title")
            assert result == "10.1038/nature12373"

    def test_case_insensitive_title_match(self, manager):
        """Should match titles case-insensitively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            master_dir = Path(tmpdir)
            paper_dir = master_dir / "12345678"
            paper_dir.mkdir()

            metadata = {
                "title": "TEST PAPER TITLE",
                "doi": "10.1038/nature12373",
                "year": 2023,
            }
            with open(paper_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            manager.config.get_library_master_dir.return_value = master_dir
            result = manager.is_doi_stored("test paper title")
            assert result == "10.1038/nature12373"

    def test_respects_year_filter(self, manager):
        """Should respect year filter when provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            master_dir = Path(tmpdir)
            paper_dir = master_dir / "12345678"
            paper_dir.mkdir()

            metadata = {
                "title": "Test Paper",
                "doi": "10.1038/nature12373",
                "year": 2023,
            }
            with open(paper_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            manager.config.get_library_master_dir.return_value = master_dir

            # Should match with correct year
            result = manager.is_doi_stored("Test Paper", year=2023)
            assert result == "10.1038/nature12373"

            # Should match with None year (no filter)
            result = manager.is_doi_stored("Test Paper", year=None)
            assert result == "10.1038/nature12373"

            # Should not match with wrong year
            result = manager.is_doi_stored("Test Paper", year=2020)
            assert result is None

    def test_skips_non_8_digit_directories(self, manager):
        """Should skip directories that don't have 8-character names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            master_dir = Path(tmpdir)
            # Create directory with wrong name length
            paper_dir = master_dir / "short"
            paper_dir.mkdir()

            metadata = {"title": "Test Paper", "doi": "10.1038/test"}
            with open(paper_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            manager.config.get_library_master_dir.return_value = master_dir
            result = manager.is_doi_stored("Test Paper")
            assert result is None

    def test_handles_json_decode_error(self, manager):
        """Should handle invalid JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            master_dir = Path(tmpdir)
            paper_dir = master_dir / "12345678"
            paper_dir.mkdir()

            with open(paper_dir / "metadata.json", "w") as f:
                f.write("invalid json{")

            manager.config.get_library_master_dir.return_value = master_dir
            result = manager.is_doi_stored("Test Paper")
            assert result is None


class TestSaveEntry:
    """Tests for save_entry method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_routes_to_resolved_when_doi_provided(self, manager):
        """Should call _save_resolved_entry when DOI is provided."""
        with patch.object(
            manager, "_save_resolved_entry", return_value=True
        ) as mock_resolved:
            result = manager.save_entry(
                title="Test Paper",
                doi="10.1038/test",
                year=2023,
                authors=["Smith, John"],
            )
            assert result is True
            mock_resolved.assert_called_once()

    def test_routes_to_unresolved_when_no_doi(self, manager):
        """Should call _save_unresolved_entry when no DOI provided."""
        with patch.object(
            manager, "_save_unresolved_entry", return_value=True
        ) as mock_unresolved:
            result = manager.save_entry(
                title="Test Paper",
                doi=None,
                year=2023,
                authors=["Smith, John"],
            )
            assert result is True
            mock_unresolved.assert_called_once()


class TestSaveResolvedEntry:
    """Tests for _save_resolved_entry method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_creates_metadata_file(self, manager):
        """Should create metadata.json with correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "MASTER" / "12345678"

            manager.config.path_manager.get_paper_storage_paths.return_value = (
                storage_path,
                "Smith-2023-Nature",
                "12345678",
            )

            with patch.object(manager, "_ensure_project_symlink", return_value=True):
                result = manager._save_resolved_entry(
                    title="Test Paper",
                    doi="10.1038/test",
                    year=2023,
                    authors=["Smith, John"],
                    source="crossref",
                    metadata={"journal": "Nature"},
                )

            assert result is True
            assert storage_path.exists()
            metadata_file = storage_path / "metadata.json"
            assert metadata_file.exists()

            with open(metadata_file) as f:
                saved_metadata = json.load(f)

            assert saved_metadata["title"] == "Test Paper"
            assert saved_metadata["doi"] == "10.1038/test"
            assert saved_metadata["year"] == 2023
            assert saved_metadata["authors"] == ["Smith, John"]
            assert saved_metadata["journal"] == "Nature"

    def test_preserves_existing_metadata(self, manager):
        """Should preserve existing metadata fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "MASTER" / "12345678"
            storage_path.mkdir(parents=True)

            existing_metadata = {
                "created_at": "2023-01-01T00:00:00",
                "extra_field": "should_be_preserved",
            }
            with open(storage_path / "metadata.json", "w") as f:
                json.dump(existing_metadata, f)

            manager.config.path_manager.get_paper_storage_paths.return_value = (
                storage_path,
                "Smith-2023-Nature",
                "12345678",
            )

            with patch.object(manager, "_ensure_project_symlink", return_value=True):
                result = manager._save_resolved_entry(
                    title="Test Paper",
                    doi="10.1038/test",
                )

            assert result is True
            with open(storage_path / "metadata.json") as f:
                saved_metadata = json.load(f)

            assert saved_metadata["created_at"] == "2023-01-01T00:00:00"
            assert saved_metadata["extra_field"] == "should_be_preserved"


class TestSaveUnresolvedEntry:
    """Tests for _save_unresolved_entry method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_creates_unresolved_metadata(self, manager):
        """Should create metadata with unresolved status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "MASTER" / "12345678"

            manager.config.path_manager.get_paper_storage_paths.return_value = (
                storage_path,
                "Unknown-2023-Unknown",
                "12345678",
            )

            with patch.object(manager, "_ensure_project_symlink", return_value=True):
                result = manager._save_unresolved_entry(
                    title="Test Paper",
                    year=2023,
                    authors=["Smith, John"],
                )

            assert result is True
            metadata_file = storage_path / "metadata.json"
            assert metadata_file.exists()

            with open(metadata_file) as f:
                saved_metadata = json.load(f)

            assert saved_metadata["title"] == "Test Paper"
            assert saved_metadata["doi"] is None
            assert saved_metadata["doi_resolution_failed"] is True
            assert saved_metadata["resolution_status"] == "unresolved"

    def test_skips_existing_unresolved_entry(self, manager):
        """Should skip if unresolved entry already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "MASTER" / "12345678"
            storage_path.mkdir(parents=True)

            with open(storage_path / "metadata.json", "w") as f:
                json.dump({"title": "Existing"}, f)

            manager.config.path_manager.get_paper_storage_paths.return_value = (
                storage_path,
                "Unknown-2023-Unknown",
                "12345678",
            )

            result = manager._save_unresolved_entry(title="Test Paper")
            assert result is True


class TestGenerateReadableName:
    """Tests for _generate_readable_name method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_generates_name_with_all_parts(self, manager):
        """Should generate name with author, year, and journal."""
        result = manager._generate_readable_name(
            authors=["Smith, John"],
            year=2023,
            journal="Nature",
        )
        assert result == "John-2023-Nature"

    def test_handles_missing_authors(self, manager):
        """Should use Unknown when no authors provided."""
        result = manager._generate_readable_name(
            authors=None,
            year=2023,
            journal="Nature",
        )
        assert result == "Unknown-2023-Nature"

    def test_handles_empty_authors_list(self, manager):
        """Should use Unknown when authors list is empty."""
        result = manager._generate_readable_name(
            authors=[],
            year=2023,
            journal="Nature",
        )
        assert result == "Unknown-2023-Nature"

    def test_handles_missing_year(self, manager):
        """Should use Unknown when no year provided."""
        result = manager._generate_readable_name(
            authors=["Smith, John"],
            year=None,
            journal="Nature",
        )
        assert result == "John-Unknown-Nature"

    def test_handles_missing_journal(self, manager):
        """Should use Unknown when no journal provided."""
        result = manager._generate_readable_name(
            authors=["Smith, John"],
            year=2023,
            journal=None,
        )
        assert result == "John-2023-Unknown"

    def test_extracts_last_name_from_author(self, manager):
        """Should extract last name from 'First Last' format."""
        result = manager._generate_readable_name(
            authors=["John Smith"],
            year=2023,
            journal="Nature",
        )
        assert result == "Smith-2023-Nature"

    def test_handles_single_name_author(self, manager):
        """Should handle author with single name."""
        result = manager._generate_readable_name(
            authors=["Madonna"],
            year=2023,
            journal="Nature",
        )
        assert result == "Madonna-2023-Nature"

    def test_cleans_journal_name(self, manager):
        """Should remove special characters from journal name."""
        result = manager._generate_readable_name(
            authors=["Smith, John"],
            year=2023,
            journal="Nature & Science!",
        )
        assert result == "John-2023-NatureScience"


class TestGetUnresolvedEntries:
    """Tests for get_unresolved_entries method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_returns_empty_list_when_dir_not_exists(self, manager):
        """Should return empty list when project dir doesn't exist."""
        manager.config.path_manager.get_library_project_dir.return_value = Path(
            "/nonexistent"
        )
        result = manager.get_unresolved_entries()
        assert result == []

    def test_finds_unresolved_entries(self, manager):
        """Should find entries with doi_resolution_failed flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paper_dir = project_dir / "12345678"
            paper_dir.mkdir()

            metadata = {
                "title": "Unresolved Paper",
                "doi_resolution_failed": True,
                "year": 2023,
                "authors": ["Test, Author"],
            }
            with open(paper_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            manager.config.path_manager.get_library_project_dir.return_value = (
                project_dir
            )
            result = manager.get_unresolved_entries()

            assert len(result) == 1
            assert result[0]["title"] == "Unresolved Paper"
            assert result[0]["paper_id"] == "12345678"

    def test_finds_entries_without_doi(self, manager):
        """Should find entries without DOI field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paper_dir = project_dir / "12345678"
            paper_dir.mkdir()

            metadata = {
                "title": "No DOI Paper",
                "year": 2023,
            }
            with open(paper_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            manager.config.path_manager.get_library_project_dir.return_value = (
                project_dir
            )
            result = manager.get_unresolved_entries()

            assert len(result) == 1
            assert result[0]["title"] == "No DOI Paper"

    def test_excludes_resolved_entries(self, manager):
        """Should exclude entries with valid DOI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            paper_dir = project_dir / "12345678"
            paper_dir.mkdir()

            metadata = {
                "title": "Resolved Paper",
                "doi": "10.1038/test",
                "year": 2023,
            }
            with open(paper_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            manager.config.path_manager.get_library_project_dir.return_value = (
                project_dir
            )
            result = manager.get_unresolved_entries()

            assert len(result) == 0


class TestCopyBibtexToLibrary:
    """Tests for copy_bibtex_to_library method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_copies_bibtex_file(self, manager):
        """Should copy bibtex file to library directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir) / "source"
            source_dir.mkdir()
            bibtex_file = source_dir / "papers.bib"
            bibtex_file.write_text("@article{test, title={Test}}")

            dest_dir = Path(tmpdir) / "library"
            dest_dir.mkdir()

            manager.config.get_library_project_info_dir.return_value = dest_dir
            result = manager.copy_bibtex_to_library(str(bibtex_file))

            assert result == str(dest_dir / "papers.bib")
            assert (dest_dir / "papers.bib").exists()

    def test_returns_empty_string_on_error(self, manager):
        """Should return empty string when copy fails."""
        manager.config.get_library_project_info_dir.return_value = Path("/nonexistent")
        result = manager.copy_bibtex_to_library("/nonexistent/file.bib")
        assert result == ""


class TestGetCacheStatistics:
    """Tests for get_cache_statistics method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_returns_zeros_when_dir_not_exists(self, manager):
        """Should return zero stats when master dir doesn't exist."""
        manager.config.get_library_master_dir.return_value = Path("/nonexistent")
        result = manager.get_cache_statistics()

        assert result["total_papers"] == 0
        assert result["resolved_papers"] == 0
        assert result["unresolved_papers"] == 0

    def test_counts_resolved_and_unresolved(self, manager):
        """Should count resolved and unresolved papers correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            master_dir = Path(tmpdir)

            # Create resolved paper
            resolved_dir = master_dir / "12345678"
            resolved_dir.mkdir()
            with open(resolved_dir / "metadata.json", "w") as f:
                json.dump({"doi": "10.1038/test"}, f)

            # Create unresolved paper
            unresolved_dir = master_dir / "87654321"
            unresolved_dir.mkdir()
            with open(unresolved_dir / "metadata.json", "w") as f:
                json.dump({"doi": None}, f)

            manager.config.get_library_master_dir.return_value = master_dir
            result = manager.get_cache_statistics()

            assert result["total_papers"] == 2
            assert result["resolved_papers"] == 1
            assert result["unresolved_papers"] == 1
            assert result["resolution_rate"] == 0.5

    def test_handles_missing_metadata_file(self, manager):
        """Should count as unresolved when metadata file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            master_dir = Path(tmpdir)
            paper_dir = master_dir / "12345678"
            paper_dir.mkdir()
            # No metadata.json file

            manager.config.get_library_master_dir.return_value = master_dir
            result = manager.get_cache_statistics()

            assert result["total_papers"] == 1
            assert result["unresolved_papers"] == 1


class TestEnsureProjectSymlink:
    """Tests for _ensure_project_symlink method."""

    @pytest.fixture
    def manager(self):
        """Create LibraryCacheManager with mocked config."""
        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        return LibraryCacheManager(config=mock_config)

    def test_creates_symlink(self, manager):
        """Should create symlink to master directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()
            master_dir = Path(tmpdir) / "MASTER"
            master_dir.mkdir()
            paper_dir = master_dir / "12345678"
            paper_dir.mkdir()

            manager.config.path_manager.get_library_project_dir.return_value = (
                project_dir
            )

            result = manager._ensure_project_symlink(
                title="Test Paper",
                authors=["Smith, John"],
                year=2023,
                journal="Nature",
                paper_id="12345678",
            )

            assert result is True
            symlink = project_dir / "John-2023-Nature"
            assert symlink.is_symlink()

    def test_updates_existing_symlink(self, manager):
        """Should update symlink if pointing to wrong target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()

            # Create existing symlink pointing to wrong target
            existing_symlink = project_dir / "John-2023-Nature"
            existing_symlink.symlink_to("../MASTER/wrong_id")

            manager.config.path_manager.get_library_project_dir.return_value = (
                project_dir
            )

            result = manager._ensure_project_symlink(
                title="Test Paper",
                authors=["Smith, John"],
                year=2023,
                journal="Nature",
                paper_id="12345678",
            )

            assert result is True
            assert existing_symlink.readlink() == Path("../MASTER/12345678")

    def test_skips_if_symlink_correct(self, manager):
        """Should skip if symlink already points to correct target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir) / "project"
            project_dir.mkdir()

            # Create correct symlink
            existing_symlink = project_dir / "John-2023-Nature"
            existing_symlink.symlink_to("../MASTER/12345678")

            manager.config.path_manager.get_library_project_dir.return_value = (
                project_dir
            )

            result = manager._ensure_project_symlink(
                title="Test Paper",
                authors=["Smith, John"],
                year=2023,
                journal="Nature",
                paper_id="12345678",
            )

            assert result is True


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
