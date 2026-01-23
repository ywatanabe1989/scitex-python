#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: tests/scitex/scholar/core/test_Scholar.py

"""Tests for Scholar core class - the main interface for scientific literature management."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================
@pytest.fixture
def mock_config():
    """Create a mock ScholarConfig for testing."""
    config = MagicMock()
    config.path_manager = MagicMock()
    config.path_manager.library_dir = Path("/tmp/test_library")
    config.path_manager.get_workspace_dir.return_value = Path("/tmp/test_workspace")
    config.path_manager._generate_paper_id.return_value = "ABC12345"
    config.paths = MagicMock()
    config.paths.scholar_dir = Path("/tmp/test_scholar")
    config.get_library_project_dir.return_value = Path("/tmp/test_library/test_project")
    config.get_library_master_dir.return_value = Path("/tmp/test_library/MASTER")
    config.resolve.return_value = True
    return config


@pytest.fixture
def mock_paper():
    """Create a mock Paper object."""
    paper = MagicMock()
    paper.metadata = MagicMock()
    paper.metadata.basic = MagicMock()
    paper.metadata.basic.title = "Test Paper Title"
    paper.metadata.basic.authors = ["Author One", "Author Two"]
    paper.metadata.basic.year = 2024
    paper.metadata.basic.abstract = "Test abstract"
    paper.metadata.basic.keywords = ["test", "paper"]
    paper.metadata.id = MagicMock()
    paper.metadata.id.doi = "10.1234/test.doi"
    paper.metadata.publication = MagicMock()
    paper.metadata.publication.journal = "Test Journal"
    paper.metadata.url = MagicMock()
    paper.metadata.url.pdfs = []
    paper.metadata.citation_count = MagicMock()
    paper.metadata.citation_count.total = 100
    paper.container = MagicMock()
    paper.container.scitex_id = "ABC12345"
    paper.to_dict.return_value = {
        "title": "Test Paper Title",
        "doi": "10.1234/test.doi",
    }
    return paper


@pytest.fixture
def mock_papers(mock_paper):
    """Create a mock Papers collection."""
    papers = MagicMock()
    papers.__iter__ = lambda self: iter([mock_paper])
    papers.__len__ = lambda self: 1
    papers.__getitem__ = lambda self, idx: mock_paper
    return papers


# =============================================================================
# Test Scholar Initialization
# =============================================================================
class TestScholarInit:
    """Tests for Scholar initialization."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_init_default(self, mock_scholar_config):
        """Test initialization with default settings."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        assert scholar.config is mock_config
        assert scholar.browser_mode == "stealth"

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_init_with_project(self, mock_scholar_config):
        """Test initialization with project name."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path(
            "/tmp/library/test_project"
        )
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists") as mock_ensure:
            scholar = Scholar(project="test_project")
            mock_ensure.assert_called_once()

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_init_with_browser_mode(self, mock_scholar_config):
        """Test initialization with custom browser mode."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar(browser_mode="interactive")

        assert scholar.browser_mode == "interactive"

    def test_init_with_config_instance(self):
        """Test initialization with ScholarConfig instance."""
        from scitex.scholar.config import ScholarConfig
        from scitex.scholar.core.Scholar import Scholar

        # Create a real config instance to test the isinstance check
        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")

        # Patch the isinstance check to accept our mock
        with patch.object(Scholar, "_ensure_project_exists"):
            with patch(
                "scitex.scholar.core.Scholar.isinstance",
                side_effect=lambda obj, cls: True
                if cls is ScholarConfig
                else isinstance(obj, cls),
            ):
                scholar = Scholar(config=mock_config)

        assert scholar.config is mock_config

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_init_with_config_path(self, mock_scholar_config):
        """Test initialization with config file path."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")
        mock_scholar_config.from_yaml.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar(config="/path/to/config.yaml")

        mock_scholar_config.from_yaml.assert_called_once_with("/path/to/config.yaml")


# =============================================================================
# Test Scholar Name Property
# =============================================================================
class TestScholarNameProperty:
    """Tests for Scholar name property."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_name_property(self, mock_scholar_config):
        """Test that name property returns class name."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        assert scholar.name == "Scholar"


# =============================================================================
# Test Scholar._init_config
# =============================================================================
class TestScholarInitConfig:
    """Tests for Scholar._init_config method."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_init_config_none(self, mock_scholar_config):
        """Test _init_config with None uses ScholarConfig.load()."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_scholar_config.load.return_value = mock_config

        result = Scholar._init_config(None, None)
        mock_scholar_config.load.assert_called_once()
        assert result is mock_config

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_init_config_string_path(self, mock_scholar_config):
        """Test _init_config with string path."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_scholar_config.from_yaml.return_value = mock_config

        result = Scholar._init_config(None, "/path/to/config.yaml")
        mock_scholar_config.from_yaml.assert_called_once_with("/path/to/config.yaml")
        assert result is mock_config

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_init_config_path_object(self, mock_scholar_config):
        """Test _init_config with Path object."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_scholar_config.from_yaml.return_value = mock_config

        result = Scholar._init_config(None, Path("/path/to/config.yaml"))
        mock_scholar_config.from_yaml.assert_called_once()

    def test_init_config_scholar_config_instance(self):
        """Test _init_config with ScholarConfig instance."""
        from scitex.scholar.config import ScholarConfig
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()

        # Patch isinstance to accept our mock as ScholarConfig
        with patch(
            "scitex.scholar.core.Scholar.isinstance",
            side_effect=lambda obj, cls: True
            if cls is ScholarConfig
            else isinstance(obj, cls),
        ):
            result = Scholar._init_config(None, mock_config)

        assert result is mock_config

    def test_init_config_invalid_type(self):
        """Test _init_config with invalid type raises TypeError."""
        from scitex.scholar.core.Scholar import Scholar

        with pytest.raises(TypeError, match="Invalid config type"):
            Scholar._init_config(None, 12345)


# =============================================================================
# Test Scholar._deduplicate_pdf_urls
# =============================================================================
class TestScholarDeduplicatePdfUrls:
    """Tests for Scholar._deduplicate_pdf_urls method."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_deduplicate_empty_list(self, mock_scholar_config):
        """Test deduplication of empty list."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        result = scholar._deduplicate_pdf_urls([])
        assert result == []

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_deduplicate_dict_format(self, mock_scholar_config):
        """Test deduplication with dict format URLs."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        urls = [
            {"url": "http://example.com/pdf1.pdf", "source": "zotero"},
            {"url": "http://example.com/pdf2.pdf", "source": "publisher"},
            {"url": "http://example.com/pdf1.pdf", "source": "other"},  # Duplicate
        ]
        result = scholar._deduplicate_pdf_urls(urls)
        assert len(result) == 2

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_deduplicate_string_format(self, mock_scholar_config):
        """Test deduplication with string format URLs."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp/workspace")
        mock_config.get_library_project_dir.return_value = Path("/tmp/library/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        urls = [
            "http://example.com/pdf1.pdf",
            "http://example.com/pdf2.pdf",
            "http://example.com/pdf1.pdf",  # Duplicate
        ]
        result = scholar._deduplicate_pdf_urls(urls)
        assert len(result) == 2


# =============================================================================
# Test Scholar Project Methods
# =============================================================================
class TestScholarProjectMethods:
    """Tests for Scholar project management methods."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_ensure_project_exists_creates_directory(
        self, mock_scholar_config, tmp_path
    ):
        """Test _ensure_project_exists creates project directory."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.get_library_project_dir.return_value = tmp_path / "test_project"
        mock_scholar_config.load.return_value = mock_config

        # Don't patch _ensure_project_exists for this test
        scholar = Scholar.__new__(Scholar)
        scholar.config = mock_config
        scholar.project = "test_project"
        scholar.browser_mode = "stealth"
        scholar.workspace_dir = tmp_path
        scholar._Scholar__scholar_engine = None
        scholar._Scholar__auth_manager = None
        scholar._Scholar__browser_manager = None
        scholar._Scholar__library_manager = None
        scholar._Scholar__library = None

        result = scholar._ensure_project_exists("test_project", "Test description")

        assert (tmp_path / "test_project").exists()
        assert (tmp_path / "test_project" / "info").exists()

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_list_projects(self, mock_scholar_config, tmp_path):
        """Test list_projects returns project information."""
        from scitex.scholar.core.Scholar import Scholar

        # Create test projects
        (tmp_path / "project1").mkdir()
        (tmp_path / "project2").mkdir()
        (tmp_path / "MASTER").mkdir()  # Should be excluded

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.path_manager.library_dir = tmp_path
        mock_config.get_library_project_dir.return_value = tmp_path / "default"
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        projects = scholar.list_projects()
        project_names = [p["name"] for p in projects]

        assert "project1" in project_names
        assert "project2" in project_names
        assert "MASTER" not in project_names


# =============================================================================
# Test Scholar Library Methods
# =============================================================================
class TestScholarLibraryMethods:
    """Tests for Scholar library management methods."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_get_library_statistics(self, mock_scholar_config, tmp_path):
        """Test get_library_statistics returns stats."""
        from scitex.scholar.core.Scholar import Scholar

        # Create MASTER directory with some files
        master_dir = tmp_path / "MASTER"
        master_dir.mkdir()
        (master_dir / "paper1").mkdir()
        (master_dir / "paper1" / "metadata.json").write_text("{}")

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.path_manager.library_dir = tmp_path
        mock_config.get_library_project_dir.return_value = tmp_path / "default"
        mock_config.get_library_master_dir.return_value = master_dir
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        stats = scholar.get_library_statistics()

        assert "total_projects" in stats
        assert "total_papers" in stats
        assert "storage_mb" in stats
        assert "library_path" in stats

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_backup_library(self, mock_scholar_config, tmp_path):
        """Test backup_library creates backup."""
        from scitex.scholar.core.Scholar import Scholar

        # Create library structure
        library_dir = tmp_path / "library"
        library_dir.mkdir()
        (library_dir / "paper.json").write_text("{}")

        backup_dir = tmp_path / "backup"

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.path_manager.library_dir = library_dir
        mock_config.get_library_project_dir.return_value = library_dir / "default"
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        backup_info = scholar.backup_library(backup_dir)

        assert "timestamp" in backup_info
        assert "source" in backup_info
        assert "backup" in backup_info
        assert "size_mb" in backup_info
        assert backup_dir.exists()


# =============================================================================
# Test Scholar Load Methods
# =============================================================================
class TestScholarLoadMethods:
    """Tests for Scholar loading methods."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_load_project_empty(self, mock_scholar_config, tmp_path):
        """Test load_project with empty project."""
        from scitex.scholar.core.Scholar import Scholar

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        mock_config = MagicMock()
        mock_config.resolve.return_value = "test_project"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.path_manager.library_dir = tmp_path
        mock_config.get_library_project_dir.return_value = project_dir
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()
            scholar.project = "test_project"

        papers = scholar.load_project()
        assert len(papers) == 0

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_load_project_nonexistent(self, mock_scholar_config, tmp_path):
        """Test load_project with non-existent project returns empty."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "nonexistent"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.path_manager.library_dir = tmp_path
        mock_config.get_library_project_dir.return_value = tmp_path / "nonexistent"
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()
            scholar.project = "nonexistent"

        papers = scholar.load_project()
        assert len(papers) == 0

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_load_project_no_project_raises(self, mock_scholar_config, tmp_path):
        """Test load_project with no project raises ValueError."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = None
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.get_library_project_dir.return_value = tmp_path
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()
            scholar.project = None

        with pytest.raises(ValueError, match="No project specified"):
            scholar.load_project()


# =============================================================================
# Test Scholar Search Methods
# =============================================================================
class TestScholarSearchMethods:
    """Tests for Scholar search methods."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_search_library(self, mock_scholar_config, tmp_path):
        """Test search_library returns Papers collection."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.get_library_project_dir.return_value = tmp_path / "default"
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        papers = scholar.search_library("test query")
        assert len(papers) == 0  # Returns empty for now


# =============================================================================
# Test Scholar Enrichment Methods
# =============================================================================
class TestScholarEnrichmentMethods:
    """Tests for Scholar enrichment methods."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_merge_enrichment_data_empty_results(self, mock_scholar_config, mock_paper):
        """Test _merge_enrichment_data with empty results."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        result = scholar._merge_enrichment_data(mock_paper, {})
        # Should return a copy of the paper (deepcopy)
        assert result is not mock_paper

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_merge_enrichment_data_with_id(self, mock_scholar_config):
        """Test _merge_enrichment_data merges ID data."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        # Create a real-like paper with nested structure
        from scitex.scholar.core.Paper import Paper

        paper = Paper()
        paper.metadata.basic.title = "Original Title"

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        results = {
            "id": {"doi": "10.1234/new.doi", "pmid": "12345678"},
            "basic": {"abstract": "New abstract"},
        }

        result = scholar._merge_enrichment_data(paper, results)
        assert result.metadata.id.doi == "10.1234/new.doi"
        assert result.metadata.basic.abstract == "New abstract"

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_enrich_current_project_no_project(self, mock_scholar_config):
        """Test _enrich_current_project raises without project."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = None
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()
            scholar.project = None

        with pytest.raises(ValueError, match="No project specified"):
            scholar._enrich_current_project()


# =============================================================================
# Test Scholar Save Methods
# =============================================================================
class TestScholarSaveMethods:
    """Tests for Scholar save methods."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    @patch("scitex.scholar.core.Scholar.ScholarLibrary")
    def test_save_papers_to_library(
        self, mock_library_class, mock_scholar_config, mock_papers
    ):
        """Test save_papers_to_library saves papers."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        mock_library = MagicMock()
        mock_library.save_paper.return_value = "paper_id_123"
        mock_library_class.return_value = mock_library

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        saved_ids = scholar.save_papers_to_library(mock_papers)
        assert len(saved_ids) == 1
        assert saved_ids[0] == "paper_id_123"


# =============================================================================
# Test Scholar Service Properties
# =============================================================================
class TestScholarServiceProperties:
    """Tests for Scholar lazy-loaded service properties."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    @patch("scitex.scholar.core.Scholar.ScholarEngine")
    def test_scholar_engine_property(self, mock_engine_class, mock_scholar_config):
        """Test _scholar_engine property lazy loading."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        mock_engine = MagicMock()
        mock_engine_class.return_value = mock_engine

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        # Access property twice
        engine1 = scholar._scholar_engine
        engine2 = scholar._scholar_engine

        # Should return same instance
        assert engine1 is engine2

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    @patch("scitex.scholar.core.Scholar.ScholarAuthManager")
    def test_auth_manager_property(self, mock_auth_class, mock_scholar_config):
        """Test _auth_manager property lazy loading."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        mock_auth = MagicMock()
        mock_auth_class.return_value = mock_auth

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        auth = scholar._auth_manager
        assert auth is mock_auth

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    @patch("scitex.scholar.core.Scholar.LibraryManager")
    def test_library_manager_property(self, mock_lib_class, mock_scholar_config):
        """Test _library_manager property lazy loading."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        mock_lib = MagicMock()
        mock_lib_class.return_value = mock_lib

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        lib_manager = scholar._library_manager
        assert lib_manager is mock_lib

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    @patch("scitex.scholar.core.Scholar.ScholarLibrary")
    def test_library_property(self, mock_lib_class, mock_scholar_config):
        """Test _library property lazy loading."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        mock_lib = MagicMock()
        mock_lib_class.return_value = mock_lib

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        lib = scholar._library
        assert lib is mock_lib


# =============================================================================
# Test Scholar Download Methods
# =============================================================================
class TestScholarDownloadMethods:
    """Tests for Scholar PDF download methods."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_download_pdfs_from_dois_empty(self, mock_scholar_config):
        """Test download_pdfs_from_dois with empty list."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        result = scholar.download_pdfs_from_dois([])
        assert result == {"downloaded": 0, "failed": 0, "errors": 0}

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_download_pdfs_from_bibtex_no_dois(self, mock_scholar_config, mock_papers):
        """Test download_pdfs_from_bibtex with papers without DOIs."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        # Make papers have no DOIs
        mock_paper = MagicMock()
        mock_paper.metadata.id.doi = None
        mock_papers.__iter__ = lambda self: iter([mock_paper])

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        result = scholar.download_pdfs_from_bibtex(mock_papers)
        assert result == {"downloaded": 0, "failed": 0, "errors": 0}


# =============================================================================
# Test Scholar Async Methods (sync wrapper tests)
# =============================================================================
class TestScholarAsyncMethods:
    """Tests for Scholar async method wrappers."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    @patch("scitex.scholar.core.Scholar.asyncio")
    def test_process_paper_calls_async(self, mock_asyncio, mock_scholar_config):
        """Test process_paper calls async version."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        mock_result = MagicMock()
        mock_asyncio.run.return_value = mock_result

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        result = scholar.process_paper(doi="10.1234/test")
        mock_asyncio.run.assert_called_once()
        assert result is mock_result

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    @patch("scitex.scholar.core.Scholar.asyncio")
    def test_process_papers_calls_async(self, mock_asyncio, mock_scholar_config):
        """Test process_papers calls async version."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        mock_result = MagicMock()
        mock_asyncio.run.return_value = mock_result

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        result = scholar.process_papers(["10.1234/test1", "10.1234/test2"])
        mock_asyncio.run.assert_called_once()
        assert result is mock_result


# =============================================================================
# Test Edge Cases
# =============================================================================
class TestScholarEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_save_papers_handles_exceptions(self, mock_scholar_config, mock_papers):
        """Test save_papers_to_library handles exceptions gracefully."""
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = Path("/tmp")
        mock_config.get_library_project_dir.return_value = Path("/tmp/default")
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        mock_library = MagicMock()
        mock_library.save_paper.side_effect = Exception("Save failed")
        scholar._Scholar__library = mock_library

        saved_ids = scholar.save_papers_to_library(mock_papers)
        # Should return empty list when all saves fail
        assert saved_ids == []

    @patch("scitex.scholar.core.Scholar.ScholarConfig")
    def test_backup_library_nonexistent_raises(self, mock_scholar_config, tmp_path):
        """Test backup_library raises when library doesn't exist."""
        from scitex.logging import ScholarError
        from scitex.scholar.core.Scholar import Scholar

        mock_config = MagicMock()
        mock_config.resolve.return_value = "default"
        mock_config.path_manager.get_workspace_dir.return_value = tmp_path
        mock_config.path_manager.library_dir = tmp_path / "nonexistent"
        mock_config.get_library_project_dir.return_value = tmp_path / "default"
        mock_scholar_config.load.return_value = mock_config

        with patch.object(Scholar, "_ensure_project_exists"):
            scholar = Scholar()

        with pytest.raises(ScholarError, match="Library directory does not exist"):
            scholar.backup_library(tmp_path / "backup")


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
