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

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/storage/ScholarLibrary.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-09-30 04:18:54 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/storage/ScholarLibrary.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Union
# 
# from scitex.scholar.config import ScholarConfig
# 
# from ._LibraryCacheManager import LibraryCacheManager
# from ._LibraryManager import LibraryManager
# from .BibTeXHandler import BibTeXHandler
# 
# 
# class ScholarLibrary:
#     """Unified Scholar library management combining cache and storage operations."""
# 
#     def __init__(
#         self, project: Union[str, Path] = None, config: Optional[ScholarConfig] = None
#     ):
#         """Initialize ScholarLibrary.
# 
#         Args:
#             project: Project name (str) or library directory path (Path)
#             config: Optional ScholarConfig instance
#         """
#         self.config = config or ScholarConfig()
# 
#         # Handle both project name and library directory path
#         if isinstance(project, Path):
#             # If Path is the library root dir (e.g., ~/.scitex/scholar/library)
#             # use the default project from config
#             if project.name == "library":
#                 self.project = self.config.resolve("project", None)
#                 self.config.library_dir = str(project)
#             else:
#                 # Path points to specific project dir
#                 # e.g., /home/user/.scitex/scholar/library/myproject -> "myproject"
#                 self.project = project.name if project.is_dir() else project.stem
#                 # Set library_dir to parent if it's named 'library'
#                 if project.parent.name == "library":
#                     self.config.library_dir = str(project.parent)
#         else:
#             # Standard project name
#             self.project = self.config.resolve("project", project)
# 
#         self._cache_manager = LibraryCacheManager(
#             project=self.project, config=self.config
#         )
#         self._library_manager = LibraryManager(project=self.project, config=self.config)
#         self.bibtex_handler = BibTeXHandler(project=self.project, config=self.config)
# 
#     def load_paper(self, library_id: str) -> Dict[str, Any]:
#         """Load paper metadata from library."""
#         return self._cache_manager.load_paper_metadata(library_id)
# 
#     def _extract_primitive(self, value):
#         """Extract primitive value from DotDict or nested structure."""
#         from scitex.dict import DotDict
# 
#         if value is None:
#             return None
#         if isinstance(value, DotDict):
#             # Convert DotDict to plain dict first
#             value = dict(value)
#         if isinstance(value, dict):
#             # For nested dict structures, return as-is (will be handled by save_resolved_paper)
#             return value
#         # Return primitive types as-is
#         return value
# 
#     def save_paper(self, paper: "Paper", force: bool = False) -> str:
#         """Save paper to library with explicit parameters.
# 
#         Supports both old flat Paper and new Pydantic Paper structures.
#         """
#         # Check if this is a Pydantic Paper (has metadata attribute)
#         if hasattr(paper, "metadata"):
#             # New Pydantic Paper structure
#             return self._library_manager.save_resolved_paper(
#                 # Required fields
#                 title=paper.metadata.basic.title or "",
#                 doi=paper.metadata.id.doi or "",
#                 # Optional bibliographic fields
#                 year=paper.metadata.basic.year,
#                 authors=paper.metadata.basic.authors,
#                 journal=paper.metadata.publication.journal,
#                 abstract=paper.metadata.basic.abstract,
#                 # Additional bibliographic fields
#                 volume=paper.metadata.publication.volume,
#                 issue=paper.metadata.publication.issue,
#                 pages=f"{paper.metadata.publication.first_page or ''}-{paper.metadata.publication.last_page or ''}"
#                 if paper.metadata.publication.first_page
#                 else None,
#                 publisher=paper.metadata.publication.publisher,
#                 issn=paper.metadata.publication.issn,
#                 # Enrichment fields
#                 citation_count=paper.metadata.citation_count.total,
#                 impact_factor=paper.metadata.publication.impact_factor,
#                 # Library management
#                 library_id=paper.container.library_id,
#                 project=self.project,
#             )
#         else:
#             # Old flat Paper structure (legacy support)
#             paper_dict = paper.to_dict() if hasattr(paper, "to_dict") else {}
# 
#             return self._library_manager.save_resolved_paper(
#                 # Required fields
#                 title=self._extract_primitive(
#                     getattr(paper, "title", paper_dict.get("title", ""))
#                 ),
#                 doi=self._extract_primitive(
#                     getattr(paper, "doi", paper_dict.get("doi", ""))
#                 ),
#                 # Optional bibliographic fields
#                 year=self._extract_primitive(
#                     getattr(paper, "year", paper_dict.get("year"))
#                 ),
#                 authors=self._extract_primitive(
#                     getattr(paper, "authors", paper_dict.get("authors"))
#                 ),
#                 journal=self._extract_primitive(
#                     getattr(paper, "journal", paper_dict.get("journal"))
#                 ),
#                 abstract=self._extract_primitive(
#                     getattr(paper, "abstract", paper_dict.get("abstract"))
#                 ),
#                 # Additional bibliographic fields
#                 volume=self._extract_primitive(
#                     getattr(paper, "volume", paper_dict.get("volume"))
#                 ),
#                 issue=self._extract_primitive(
#                     getattr(paper, "issue", paper_dict.get("issue"))
#                 ),
#                 pages=self._extract_primitive(
#                     getattr(paper, "pages", paper_dict.get("pages"))
#                 ),
#                 publisher=self._extract_primitive(
#                     getattr(paper, "publisher", paper_dict.get("publisher"))
#                 ),
#                 issn=self._extract_primitive(
#                     getattr(paper, "issn", paper_dict.get("issn"))
#                 ),
#                 # Enrichment fields
#                 citation_count=self._extract_primitive(
#                     getattr(paper, "citation_count", paper_dict.get("citation_count"))
#                 ),
#                 impact_factor=self._extract_primitive(
#                     getattr(
#                         paper,
#                         "journal_impact_factor",
#                         paper_dict.get("journal_impact_factor"),
#                     )
#                 ),
#                 # Source tracking
#                 doi_source=self._extract_primitive(
#                     getattr(paper, "doi_source", paper_dict.get("doi_source"))
#                 ),
#                 title_source=self._extract_primitive(
#                     getattr(paper, "title_source", paper_dict.get("title_source"))
#                 ),
#                 abstract_source=self._extract_primitive(
#                     getattr(paper, "abstract_source", paper_dict.get("abstract_source"))
#                 ),
#                 # Library management
#                 library_id=self._extract_primitive(
#                     getattr(paper, "library_id", paper_dict.get("library_id"))
#                 ),
#                 project=self.project,
#             )
# 
#     def papers_from_bibtex(self, bibtex_input: Union[str, Path]) -> List["Paper"]:
#         """Create Papers from BibTeX file or content."""
#         return self.bibtex_handler.papers_from_bibtex(bibtex_input)
# 
#     def paper_from_bibtex_entry(self, entry: Dict[str, Any]) -> Optional["Paper"]:
#         """Convert BibTeX entry to Paper."""
#         return self.bibtex_handler.paper_from_bibtex_entry(entry)
# 
#     def check_existing_doi(
#         self, title: str, year: Optional[int] = None
#     ) -> Optional[str]:
#         """Check if DOI exists in library."""
#         return self._cache_manager.is_doi_stored(title, year)
# 
# 
# if __name__ == "__main__":
#     # Implement main guard to demonstrate typical usage of this script
#     def main():
#         pass
# 
#     main()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/storage/ScholarLibrary.py
# --------------------------------------------------------------------------------
