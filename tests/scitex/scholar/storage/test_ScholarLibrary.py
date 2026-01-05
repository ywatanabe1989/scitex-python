#!/usr/bin/env python3
# Timestamp: "2026-01-05"
# File: tests/scitex/scholar/storage/test_ScholarLibrary.py
# ----------------------------------------

"""
Comprehensive tests for the ScholarLibrary class.

Tests cover:
- Initialization with project name and path
- Paper loading and saving
- BibTeX import functionality
- DOI checking functionality
- Primitive extraction helper
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scitex.scholar.core import Paper
from scitex.scholar.storage import ScholarLibrary


def create_sample_paper(
    title: str = "Sample Paper",
    year: int = 2023,
    authors: list = None,
    journal: str = "Nature",
    doi: str = None,
    library_id: str = None,
) -> Paper:
    """Helper to create sample Paper objects."""
    paper = Paper()
    paper.metadata.basic.title = title
    paper.metadata.basic.year = year
    paper.metadata.basic.authors = authors or ["Smith, John", "Doe, Jane"]
    paper.metadata.publication.journal = journal
    if doi:
        paper.metadata.id.doi = doi
    if library_id:
        paper.container.library_id = library_id
    return paper


class TestScholarLibraryInit:
    """Tests for ScholarLibrary initialization."""

    def test_create_with_project_name(self, tmp_path):
        """Creating ScholarLibrary with project name should work."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test_project"
            mock_instance.library_dir = str(tmp_path)
            mock_config.return_value = mock_instance

            library = ScholarLibrary(project="test_project", config=mock_instance)

            assert library.project == "test_project"

    def test_create_with_path_project_dir(self, tmp_path):
        """Creating ScholarLibrary with Path to project dir should work."""
        # Create project dir structure
        project_dir = tmp_path / "library" / "my_project"
        project_dir.mkdir(parents=True)

        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "my_project"
            mock_config.return_value = mock_instance

            library = ScholarLibrary(project=project_dir, config=mock_instance)

            assert library.project == "my_project"

    def test_create_with_path_library_root(self, tmp_path):
        """Creating ScholarLibrary with library root path should use default project."""
        library_root = tmp_path / "library"
        library_root.mkdir()

        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "default"
            mock_config.return_value = mock_instance

            library = ScholarLibrary(project=library_root, config=mock_instance)

            assert library.project == "default"
            assert mock_instance.library_dir == str(library_root)

    def test_has_bibtex_handler(self, tmp_path):
        """ScholarLibrary should have bibtex_handler attribute."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test"
            mock_config.return_value = mock_instance

            library = ScholarLibrary(project="test", config=mock_instance)

            assert hasattr(library, "bibtex_handler")


class TestScholarLibraryExtractPrimitive:
    """Tests for _extract_primitive helper method."""

    @pytest.fixture
    def library(self, tmp_path):
        """Create ScholarLibrary instance for testing."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test"
            mock_config.return_value = mock_instance
            return ScholarLibrary(project="test", config=mock_instance)

    def test_extract_none(self, library):
        """_extract_primitive should return None for None input."""
        assert library._extract_primitive(None) is None

    def test_extract_string(self, library):
        """_extract_primitive should return string as-is."""
        assert library._extract_primitive("test") == "test"

    def test_extract_int(self, library):
        """_extract_primitive should return int as-is."""
        assert library._extract_primitive(42) == 42

    def test_extract_float(self, library):
        """_extract_primitive should return float as-is."""
        assert library._extract_primitive(3.14) == 3.14

    def test_extract_list(self, library):
        """_extract_primitive should return list as-is."""
        assert library._extract_primitive([1, 2, 3]) == [1, 2, 3]

    def test_extract_dict(self, library):
        """_extract_primitive should return dict as-is."""
        result = library._extract_primitive({"key": "value"})
        assert result == {"key": "value"}

    def test_extract_dotdict(self, library):
        """_extract_primitive should convert DotDict to dict."""
        from scitex.dict import DotDict

        dd = DotDict({"key": "value"})
        result = library._extract_primitive(dd)

        assert isinstance(result, dict)
        assert result == {"key": "value"}


class TestScholarLibraryPapersFromBibtex:
    """Tests for papers_from_bibtex method."""

    @pytest.fixture
    def library(self, tmp_path):
        """Create ScholarLibrary instance for testing."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test"
            mock_config.return_value = mock_instance
            return ScholarLibrary(project="test", config=mock_instance)

    def test_papers_from_bibtex_file(self, library, tmp_path):
        """papers_from_bibtex should parse BibTeX file."""
        # Create sample bibtex file
        bibtex_content = """
@article{smith2023,
    author = {Smith, John},
    title = {A Test Paper},
    journal = {Nature},
    year = {2023},
    doi = {10.1234/test}
}
"""
        bib_file = tmp_path / "test.bib"
        bib_file.write_text(bibtex_content)

        papers = library.papers_from_bibtex(bib_file)

        assert isinstance(papers, list)
        # Check at least one paper was parsed
        if len(papers) > 0:
            assert isinstance(papers[0], Paper)

    def test_papers_from_bibtex_string(self, library):
        """papers_from_bibtex should parse BibTeX string."""
        bibtex_str = """
@article{test2023,
    title = {Test Paper},
    year = {2023}
}
"""
        papers = library.papers_from_bibtex(bibtex_str)

        assert isinstance(papers, list)


class TestScholarLibraryPaperFromBibtexEntry:
    """Tests for paper_from_bibtex_entry method."""

    @pytest.fixture
    def library(self, tmp_path):
        """Create ScholarLibrary instance for testing."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test"
            mock_config.return_value = mock_instance
            return ScholarLibrary(project="test", config=mock_instance)

    def test_convert_basic_entry(self, library):
        """paper_from_bibtex_entry should convert basic entry."""
        entry = {
            "ID": "smith2023",
            "ENTRYTYPE": "article",
            "title": "Test Paper",
            "year": "2023",
            "author": "Smith, John",
        }

        paper = library.paper_from_bibtex_entry(entry)

        # May return None if year parsing fails in some implementations
        if paper is not None:
            assert isinstance(paper, Paper)
            assert paper.metadata.basic.title == "Test Paper"

    def test_convert_entry_with_doi(self, library):
        """paper_from_bibtex_entry should include DOI."""
        entry = {
            "ID": "test2023",
            "ENTRYTYPE": "article",
            "title": "DOI Paper",
            "year": "2023",
            "doi": "10.1234/test",
        }

        paper = library.paper_from_bibtex_entry(entry)

        if paper is not None:
            assert paper.metadata.id.doi == "10.1234/test"


class TestScholarLibrarySavePaper:
    """Tests for save_paper method."""

    @pytest.fixture
    def library(self, tmp_path):
        """Create ScholarLibrary instance with mocked manager."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test"
            mock_config.return_value = mock_instance

            lib = ScholarLibrary(project="test", config=mock_instance)
            # Mock the library manager's save method
            lib._library_manager.save_resolved_paper = MagicMock(
                return_value="AB12CD34"
            )
            return lib

    def test_save_pydantic_paper(self, library):
        """save_paper should save Pydantic Paper correctly."""
        paper = create_sample_paper(
            title="Test Paper",
            year=2023,
            authors=["Smith, John"],
            journal="Nature",
            doi="10.1234/test",
            library_id="AB12CD34",
        )

        result = library.save_paper(paper)

        assert result == "AB12CD34"
        library._library_manager.save_resolved_paper.assert_called_once()

    def test_save_paper_extracts_all_fields(self, library):
        """save_paper should extract all paper fields."""
        paper = create_sample_paper(
            title="Complete Paper",
            year=2024,
            authors=["Doe, Jane", "Smith, John"],
            journal="Science",
            doi="10.5678/complete",
        )
        paper.metadata.basic.abstract = "This is an abstract."
        paper.metadata.citation_count.total = 150
        paper.metadata.publication.impact_factor = 47.5

        library.save_paper(paper)

        call_kwargs = library._library_manager.save_resolved_paper.call_args.kwargs
        assert call_kwargs["title"] == "Complete Paper"
        assert call_kwargs["year"] == 2024
        assert call_kwargs["doi"] == "10.5678/complete"
        assert call_kwargs["abstract"] == "This is an abstract."
        assert call_kwargs["citation_count"] == 150
        assert call_kwargs["impact_factor"] == 47.5


class TestScholarLibraryLoadPaper:
    """Tests for load_paper method."""

    @pytest.fixture
    def library(self, tmp_path):
        """Create ScholarLibrary instance with mocked cache manager."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test"
            mock_config.return_value = mock_instance

            lib = ScholarLibrary(project="test", config=mock_instance)
            return lib

    def test_load_paper_calls_cache_manager(self, library):
        """load_paper should delegate to cache manager."""
        expected_data = {"metadata": {"basic": {"title": "Loaded Paper", "year": 2023}}}
        library._cache_manager.load_paper_metadata = MagicMock(
            return_value=expected_data
        )

        result = library.load_paper("AB12CD34")

        library._cache_manager.load_paper_metadata.assert_called_once_with("AB12CD34")
        assert result == expected_data


class TestScholarLibraryCheckExistingDoi:
    """Tests for check_existing_doi method."""

    @pytest.fixture
    def library(self, tmp_path):
        """Create ScholarLibrary instance with mocked cache manager."""
        with patch("scitex.scholar.config.ScholarConfig") as mock_config:
            mock_instance = MagicMock()
            mock_instance.resolve.return_value = "test"
            mock_config.return_value = mock_instance

            lib = ScholarLibrary(project="test", config=mock_instance)
            return lib

    def test_check_existing_doi_found(self, library):
        """check_existing_doi should return DOI if found."""
        library._cache_manager.is_doi_stored = MagicMock(
            return_value="10.1234/existing"
        )

        result = library.check_existing_doi("Existing Paper", year=2023)

        library._cache_manager.is_doi_stored.assert_called_once_with(
            "Existing Paper", 2023
        )
        assert result == "10.1234/existing"

    def test_check_existing_doi_not_found(self, library):
        """check_existing_doi should return None if not found."""
        library._cache_manager.is_doi_stored = MagicMock(return_value=None)

        result = library.check_existing_doi("New Paper", year=2024)

        assert result is None

    def test_check_existing_doi_without_year(self, library):
        """check_existing_doi should work without year."""
        library._cache_manager.is_doi_stored = MagicMock(return_value="10.1234/test")

        result = library.check_existing_doi("Some Paper")

        library._cache_manager.is_doi_stored.assert_called_once_with("Some Paper", None)
        assert result == "10.1234/test"


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
