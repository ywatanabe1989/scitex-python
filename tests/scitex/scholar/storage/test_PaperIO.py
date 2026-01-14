#!/usr/bin/env python3
# Timestamp: "2026-01-05"
# File: tests/scitex/scholar/storage/test_PaperIO.py
# ----------------------------------------

"""
Comprehensive tests for the PaperIO class.

Tests cover:
- Initialization with Paper objects
- Path getter methods
- Check methods for file existence
- Save methods for metadata, PDF, text, tables, images
- Load methods for metadata, text, tables
- Utility methods
"""

import json
from pathlib import Path

import pytest

from scitex.scholar.core import Paper
from scitex.scholar.storage import PaperIO


def create_sample_paper(
    library_id: str = "A1B2C3D4",
    title: str = "Sample Paper",
    year: int = 2023,
    authors: list = None,
    journal: str = "Nature",
    doi: str = None,
) -> Paper:
    """Helper to create sample Paper objects with required library_id."""
    paper = Paper()
    paper.container.library_id = library_id
    paper.metadata.basic.title = title
    paper.metadata.basic.year = year
    paper.metadata.basic.authors = authors or ["Smith, John", "Doe, Jane"]
    paper.metadata.publication.journal = journal
    if doi:
        paper.metadata.id.doi = doi
    return paper


class TestPaperIOInit:
    """Tests for PaperIO initialization."""

    def test_create_with_paper_and_base_dir(self, tmp_path):
        """Creating PaperIO with paper and base_dir should work."""
        paper = create_sample_paper()
        io = PaperIO(paper, base_dir=tmp_path)

        assert io.paper is paper
        assert io.paper_dir == tmp_path / "A1B2C3D4"
        assert io.paper_dir.exists()

    def test_paper_dir_created(self, tmp_path):
        """Paper directory should be created on init."""
        paper = create_sample_paper(library_id="X1Y2Z3W4")
        io = PaperIO(paper, base_dir=tmp_path)

        assert (tmp_path / "X1Y2Z3W4").exists()

    def test_paper_without_library_id_raises(self, tmp_path):
        """Paper without library_id should raise ValueError."""
        paper = Paper()  # No library_id set

        with pytest.raises(ValueError, match="library_id"):
            PaperIO(paper, base_dir=tmp_path)

    def test_name_attribute(self, tmp_path):
        """PaperIO should have name attribute."""
        paper = create_sample_paper()
        io = PaperIO(paper, base_dir=tmp_path)

        assert io.name == "PaperIO"


class TestPaperIOPathGetters:
    """Tests for path getter methods."""

    @pytest.fixture
    def paper_io(self, tmp_path):
        """Create sample PaperIO instance."""
        paper = create_sample_paper()
        return PaperIO(paper, base_dir=tmp_path)

    def test_get_metadata_path(self, paper_io):
        """get_metadata_path should return path to metadata.json."""
        path = paper_io.get_metadata_path()

        assert path.name == "metadata.json"
        assert path.parent == paper_io.paper_dir

    def test_get_text_path(self, paper_io):
        """get_text_path should return path to content.txt."""
        path = paper_io.get_text_path()

        assert path.name == "content.txt"
        assert path.parent == paper_io.paper_dir

    def test_get_tables_path(self, paper_io):
        """get_tables_path should return path to tables.json."""
        path = paper_io.get_tables_path()

        assert path.name == "tables.json"
        assert path.parent == paper_io.paper_dir

    def test_get_images_dir(self, paper_io):
        """get_images_dir should return path to images/ and create it."""
        path = paper_io.get_images_dir()

        assert path.name == "images"
        assert path.parent == paper_io.paper_dir
        assert path.exists()

    def test_get_screenshots_dir(self, paper_io):
        """get_screenshots_dir should return path to screenshots/ and create it."""
        path = paper_io.get_screenshots_dir()

        assert path.name == "screenshots"
        assert path.parent == paper_io.paper_dir
        assert path.exists()


class TestPaperIOCheckMethods:
    """Tests for file existence check methods."""

    @pytest.fixture
    def paper_io(self, tmp_path):
        """Create sample PaperIO instance."""
        paper = create_sample_paper()
        return PaperIO(paper, base_dir=tmp_path)

    def test_has_metadata_false(self, paper_io):
        """has_metadata should return False when file doesn't exist."""
        assert paper_io.has_metadata() is False

    def test_has_metadata_true(self, paper_io):
        """has_metadata should return True when file exists."""
        paper_io.get_metadata_path().write_text("{}")
        assert paper_io.has_metadata() is True

    def test_has_pdf_false(self, paper_io):
        """has_pdf should return False when no PDF exists."""
        assert paper_io.has_pdf() is False

    def test_has_pdf_true(self, paper_io):
        """has_pdf should return True when PDF exists."""
        (paper_io.paper_dir / "test.pdf").write_bytes(b"PDF content")
        assert paper_io.has_pdf() is True

    def test_has_content_false(self, paper_io):
        """has_content should return False when file doesn't exist."""
        assert paper_io.has_content() is False

    def test_has_content_true(self, paper_io):
        """has_content should return True when file exists."""
        paper_io.get_text_path().write_text("Content here")
        assert paper_io.has_content() is True

    def test_has_tables_false(self, paper_io):
        """has_tables should return False when file doesn't exist."""
        assert paper_io.has_tables() is False

    def test_has_tables_true(self, paper_io):
        """has_tables should return True when file exists."""
        paper_io.get_tables_path().write_text("[]")
        assert paper_io.has_tables() is True


class TestPaperIOSaveMethods:
    """Tests for save methods."""

    @pytest.fixture
    def paper_io(self, tmp_path):
        """Create sample PaperIO instance."""
        paper = create_sample_paper(
            title="Test Paper",
            year=2023,
            authors=["Smith, John"],
            journal="Nature",
            doi="10.1234/test",
        )
        return PaperIO(paper, base_dir=tmp_path)

    def test_save_metadata(self, paper_io):
        """save_metadata should write Paper to metadata.json."""
        path = paper_io.save_metadata()

        assert path.exists()
        assert path.name == "metadata.json"

        # Verify content
        with open(path) as f:
            data = json.load(f)
        assert "metadata" in data
        assert data["metadata"]["basic"]["title"] == "Test Paper"

    def test_save_text(self, paper_io):
        """save_text should write content to content.txt."""
        text = "This is extracted text from the paper."
        path = paper_io.save_text(text)

        assert path.exists()
        assert path.name == "content.txt"
        assert path.read_text() == text

    def test_save_tables(self, paper_io):
        """save_tables should write tables to tables.json."""
        tables = [
            {"columns": ["A", "B"], "data": [[1, 2], [3, 4]]},
            {"columns": ["X", "Y"], "data": [[5, 6]]},
        ]
        path = paper_io.save_tables(tables)

        assert path.exists()
        assert path.name == "tables.json"

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == tables

    def test_save_image(self, paper_io):
        """save_image should write image to images/ directory."""
        image_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG header
        filename = "fig1.png"

        path = paper_io.save_image(image_data, filename)

        assert path.exists()
        assert path.name == filename
        assert path.parent.name == "images"
        assert path.read_bytes() == image_data

    def test_save_pdf_from_file(self, paper_io, tmp_path):
        """save_pdf should copy PDF to paper directory."""
        # Create source PDF
        source_pdf = tmp_path / "source" / "paper.pdf"
        source_pdf.parent.mkdir()
        source_pdf.write_bytes(b"%PDF-1.4 fake pdf content")

        path = paper_io.save_pdf(source_pdf)

        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.read_bytes() == b"%PDF-1.4 fake pdf content"

    def test_save_pdf_updates_paper(self, paper_io, tmp_path):
        """save_pdf should update paper metadata."""
        source_pdf = tmp_path / "source" / "test.pdf"
        source_pdf.parent.mkdir()
        source_pdf.write_bytes(b"%PDF-1.4 content")

        paper_io.save_pdf(source_pdf)

        assert paper_io.paper.metadata.path.pdfs is not None
        assert paper_io.paper.container.pdf_size_bytes > 0

    def test_save_pdf_nonexistent_raises(self, paper_io):
        """save_pdf should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            paper_io.save_pdf(Path("/nonexistent/file.pdf"))


class TestPaperIOLoadMethods:
    """Tests for load methods."""

    @pytest.fixture
    def paper_io(self, tmp_path):
        """Create sample PaperIO instance."""
        paper = create_sample_paper()
        return PaperIO(paper, base_dir=tmp_path)

    def test_load_metadata(self, paper_io):
        """load_metadata should read Paper from metadata.json."""
        # First save
        paper_io.paper.metadata.basic.title = "Saved Paper"
        paper_io.save_metadata()

        # Then load
        loaded = paper_io.load_metadata()

        assert loaded.metadata.basic.title == "Saved Paper"
        assert paper_io.paper is loaded  # Internal reference updated

    def test_load_metadata_nonexistent_raises(self, paper_io):
        """load_metadata should raise FileNotFoundError when file missing."""
        with pytest.raises(FileNotFoundError, match="Metadata not found"):
            paper_io.load_metadata()

    def test_load_text(self, paper_io):
        """load_text should read content from content.txt."""
        text = "Extracted paper content here."
        paper_io.save_text(text)

        loaded = paper_io.load_text()

        assert loaded == text

    def test_load_text_nonexistent_raises(self, paper_io):
        """load_text should raise FileNotFoundError when file missing."""
        with pytest.raises(FileNotFoundError, match="Text not found"):
            paper_io.load_text()

    def test_load_tables(self, paper_io):
        """load_tables should read tables from tables.json."""
        tables = [{"id": 1, "data": [1, 2, 3]}]
        paper_io.save_tables(tables)

        loaded = paper_io.load_tables()

        assert loaded == tables

    def test_load_tables_nonexistent_raises(self, paper_io):
        """load_tables should raise FileNotFoundError when file missing."""
        with pytest.raises(FileNotFoundError, match="Tables not found"):
            paper_io.load_tables()


class TestPaperIOUtilityMethods:
    """Tests for utility methods."""

    @pytest.fixture
    def paper_io(self, tmp_path):
        """Create sample PaperIO instance."""
        paper = create_sample_paper()
        return PaperIO(paper, base_dir=tmp_path)

    def test_get_all_files_empty(self, paper_io):
        """get_all_files should return status of all expected files."""
        status = paper_io.get_all_files()

        assert isinstance(status, dict)
        assert "metadata.json" in status
        assert "content.txt" in status
        assert "tables.json" in status
        # Images and screenshots should be created by get_all_files
        assert any("images" in k for k in status)
        assert any("screenshots" in k for k in status)

    def test_get_all_files_with_content(self, paper_io):
        """get_all_files should correctly show existing files."""
        paper_io.save_metadata()
        paper_io.save_text("test")

        status = paper_io.get_all_files()

        assert status["metadata.json"] is True
        assert status["content.txt"] is True
        assert status["tables.json"] is False

    def test_repr(self, paper_io):
        """repr should include library_id and path."""
        repr_str = repr(paper_io)

        assert "PaperIO" in repr_str
        assert paper_io.paper.container.library_id in repr_str


class TestPaperIOTableExtraction:
    """Tests for save_tables_from_extraction method."""

    @pytest.fixture
    def paper_io(self, tmp_path):
        """Create sample PaperIO instance."""
        paper = create_sample_paper()
        return PaperIO(paper, base_dir=tmp_path)

    def test_save_tables_from_extraction_with_dataframes(self, paper_io):
        """save_tables_from_extraction should convert DataFrames to JSON."""
        import pandas as pd

        # Simulate tables_dict from PDF extraction: Dict[page_num, List[DataFrame]]
        tables_dict = {
            1: [pd.DataFrame({"A": [1, 2], "B": [3, 4]})],
            3: [
                pd.DataFrame({"X": ["a", "b"]}),
                pd.DataFrame({"Y": [10.5, 20.5], "Z": [30.5, 40.5]}),
            ],
        }

        path = paper_io.save_tables_from_extraction(tables_dict)

        assert path.exists()
        assert path.name == "tables.json"

        # Verify structure
        loaded = paper_io.load_tables()
        assert len(loaded) == 3  # 1 + 2 tables

        # Check first table
        t1 = loaded[0]
        assert t1["page"] == 1
        assert t1["index"] == 0
        assert t1["columns"] == ["A", "B"]
        assert t1["shape"] == [2, 2]
        assert len(t1["data"]) == 2

        # Check third table (second from page 3)
        t3 = loaded[2]
        assert t3["page"] == 3
        assert t3["index"] == 1
        assert "Y" in t3["columns"]

    def test_save_tables_from_extraction_empty(self, paper_io):
        """save_tables_from_extraction with empty dict should save empty list."""
        path = paper_io.save_tables_from_extraction({})

        assert path.exists()
        loaded = paper_io.load_tables()
        assert loaded == []

    def test_save_tables_from_extraction_roundtrip(self, paper_io):
        """Data should be preserved through save/load cycle."""
        import pandas as pd

        original_data = {"Col1": [1, 2, 3], "Col2": ["x", "y", "z"]}
        tables_dict = {5: [pd.DataFrame(original_data)]}

        paper_io.save_tables_from_extraction(tables_dict)
        loaded = paper_io.load_tables()

        # Verify data integrity
        assert loaded[0]["data"][0]["Col1"] == 1
        assert loaded[0]["data"][2]["Col2"] == "z"


class TestPaperIORoundtrip:
    """Tests for save/load roundtrip operations."""

    def test_metadata_roundtrip(self, tmp_path):
        """Paper saved and loaded should preserve all data."""
        # Create paper with comprehensive metadata
        paper = create_sample_paper(
            library_id="ROUNDTRP",
            title="Roundtrip Test Paper",
            year=2023,
            authors=["Smith, John", "Doe, Jane"],
            journal="Nature",
            doi="10.1234/roundtrip",
        )
        paper.metadata.basic.abstract = "Test abstract for roundtrip."
        paper.metadata.citation_count.total = 150

        # Save
        io = PaperIO(paper, base_dir=tmp_path)
        io.save_metadata()

        # Load into new PaperIO
        new_paper = Paper()
        new_paper.container.library_id = "ROUNDTRP"
        new_io = PaperIO(new_paper, base_dir=tmp_path)
        loaded = new_io.load_metadata()

        # Verify
        assert loaded.metadata.basic.title == "Roundtrip Test Paper"
        assert loaded.metadata.basic.year == 2023
        assert loaded.metadata.basic.authors == ["Smith, John", "Doe, Jane"]
        assert loaded.metadata.id.doi == "10.1234/roundtrip"
        assert loaded.metadata.basic.abstract == "Test abstract for roundtrip."
        assert loaded.metadata.citation_count.total == 150

    def test_text_roundtrip(self, tmp_path):
        """Text saved and loaded should be identical."""
        paper = create_sample_paper()
        io = PaperIO(paper, base_dir=tmp_path)

        text = "Line 1\nLine 2\nLine 3 with special chars: äöü"
        io.save_text(text)
        loaded = io.load_text()

        assert loaded == text

    def test_tables_roundtrip(self, tmp_path):
        """Tables saved and loaded should be identical."""
        paper = create_sample_paper()
        io = PaperIO(paper, base_dir=tmp_path)

        tables = [
            {"name": "Table 1", "columns": ["A", "B"], "data": [[1.5, 2.5]]},
            {"name": "Table 2", "columns": ["X"], "data": [["value"]]},
        ]
        io.save_tables(tables)
        loaded = io.load_tables()

        assert loaded == tables

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/storage/PaperIO.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-10-11 23:45:23 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/storage/PaperIO.py
# # ----------------------------------------
# from __future__ import annotations
# 
# import os
# 
# __FILE__ = "./src/scitex/scholar/storage/PaperIO.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Functionalities:
#   - Provides simple IO interface for Paper objects
#   - Handles save/load/check operations for Paper data in MASTER directory
#   - Each Paper gets structured directory: MASTER/{8-digit-ID}/
#   - Supports incremental data addition (check → process → save pattern)
#   - All operations work with Paper object fields
# 
# Dependencies:
#   - packages:
#     - pydantic
#     - scitex
# 
# IO:
#   - input-files:
#     - library/MASTER/{paper_id}/metadata.json
#     - library/MASTER/{paper_id}/main.pdf
#     - library/downloads/{UUID} (for PDF import)
# 
#   - output-files:
#     - library/MASTER/{paper_id}/metadata.json
#     - library/MASTER/{paper_id}/main.pdf
#     - library/MASTER/{paper_id}/content.txt
#     - library/MASTER/{paper_id}/tables.json
#     - library/MASTER/{paper_id}/images/
#     - library/MASTER/{paper_id}/screenshots/
# """
# 
# """Imports"""
# import argparse
# import json
# import shutil
# from pathlib import Path
# from typing import Any, Dict, List, Optional
# 
# from scitex import logging
# from scitex.scholar.core import Paper
# 
# logger = logging.getLogger(__name__)
# 
# 
# class PaperIO:
#     """Simple IO interface for Paper objects.
# 
#     Handles all file operations for a Paper in its MASTER directory.
#     """
# 
#     def __init__(self, paper: Paper, base_dir: Optional[Path] = None):
#         """Initialize PaperIO for a Paper.
# 
#         Args:
#             paper: Paper object to manage
#             base_dir: Base directory (default: ~/.scitex/scholar/library/MASTER)
#         """
#         self.paper = paper
#         self.name = self.__class__.__name__
# 
#         if base_dir is None:
#             from scitex.scholar import ScholarConfig
# 
#             config = ScholarConfig()
#             base_dir = config.get_library_master_dir()
# 
#         # Get paper ID from container
#         paper_id = paper.container.library_id
#         if not paper_id:
#             raise ValueError("Paper must have container.library_id set")
# 
#         self.paper_dir = Path(base_dir) / paper_id
#         self.paper_dir.mkdir(parents=True, exist_ok=True)
# 
#     # ========================================
#     # Path Getters
#     # ========================================
#     def get_metadata_path(self) -> Path:
#         """Get path to metadata.json"""
#         return self.paper_dir / "metadata.json"
# 
#     def get_pdf_path(self, suffix: str = None) -> Path:
#         """Get formatted PDF path using PathManager template.
# 
#         Returns path like: MASTER/{paper_id}/{FirstAuthor}-{year}-{Journal}.pdf
#         If suffix provided: {FirstAuthor}-{year}-{Journal}-{suffix}.pdf
#         """
#         from scitex.scholar import ScholarConfig
# 
#         config = ScholarConfig()
# 
#         # Extract metadata for formatting
#         first_author = (
#             self.paper.metadata.basic.authors[0].split()[-1]
#             if self.paper.metadata.basic.authors
#             else "Unknown"
#         )
#         year = self.paper.metadata.basic.year or 0
#         journal_name = (
#             self.paper.metadata.publication.short_journal
#             or self.paper.metadata.publication.journal
#             or "Unknown"
#         )
# 
#         # Normalize journal name using PathManager (single source of truth)
#         # This delegates to PathManager._sanitize_filename() which replaces spaces/dots with hyphens
#         pdf_name = config.path_manager.get_library_project_entry_pdf_fname(
#             first_author=first_author,
#             year=year,
#             journal_name=journal_name,  # Will be sanitized by PathManager
#         )
# 
#         # Add suffix if provided (for duplicate PDFs)
#         if suffix:
#             name, ext = pdf_name.rsplit(".", 1)
#             pdf_name = f"{name}-{suffix}.{ext}"
# 
#         return self.paper_dir / pdf_name
# 
#     def get_text_path(self) -> Path:
#         """Get path to content.txt"""
#         return self.paper_dir / "content.txt"
# 
#     def get_tables_path(self) -> Path:
#         """Get path to tables.json"""
#         return self.paper_dir / "tables.json"
# 
#     def get_images_dir(self) -> Path:
#         """Get path to images/ directory"""
#         images_dir = self.paper_dir / "images"
#         images_dir.mkdir(exist_ok=True)
#         return images_dir
# 
#     def get_screenshots_dir(self) -> Path:
#         """Get path to screenshots/ directory"""
#         screenshots_dir = self.paper_dir / "screenshots"
#         screenshots_dir.mkdir(exist_ok=True)
#         return screenshots_dir
# 
#     def get_entry_name_for_project(self) -> str:
#         """Generate entry/symlink name using PathManager format.
# 
#         Returns formatted name like:
#         PDF-01_CC-000113_IF-010_2017_Baldassano_Brain
#         """
#         from scitex.scholar import ScholarConfig
# 
#         config = ScholarConfig()
# 
#         # Count PDFs in directory
#         n_pdfs = len(list(self.paper_dir.glob("*.pdf")))
# 
#         # Extract metadata
#         citation_count = self.paper.metadata.citation_count.total or 0
#         impact_factor = int(self.paper.metadata.publication.impact_factor or 0)
#         year = self.paper.metadata.basic.year or 0
#         first_author = (
#             self.paper.metadata.basic.authors[0].split()[-1]
#             if self.paper.metadata.basic.authors
#             else "Unknown"
#         )
#         journal_name = (
#             self.paper.metadata.publication.short_journal
#             or self.paper.metadata.publication.journal
#             or "Unknown"
#         )
# 
#         # Use PathManager to format (single source of truth)
#         return config.path_manager.get_library_project_entry_dirname(
#             n_pdfs=n_pdfs,
#             citation_count=citation_count,
#             impact_factor=impact_factor,
#             year=year,
#             first_author=first_author,
#             journal_name=journal_name,
#         )
# 
#     # ========================================
#     # Check Methods
#     # ========================================
#     def has_metadata(self) -> bool:
#         """Check if metadata.json exists"""
#         return self.get_metadata_path().exists()
# 
#     def has_pdf(self) -> bool:
#         """Check if any PDF exists in paper directory"""
#         return len(list(self.paper_dir.glob("*.pdf"))) > 0
# 
#     def has_content(self) -> bool:
#         """Check if content.txt exists"""
#         return self.get_text_path().exists()
# 
#     def has_tables(self) -> bool:
#         """Check if tables.json exists"""
#         return self.get_tables_path().exists()
# 
#     # ========================================
#     # Save Methods
#     # ========================================
#     def save_metadata(self) -> Path:
#         """Save Paper metadata to metadata.json
# 
#         Returns
#         -------
#             Path to saved metadata.json
#         """
#         path = self.get_metadata_path()
#         with open(path, "w") as f:
#             json.dump(self.paper.to_dict(), f, indent=2)
#         logger.debug(f"{self.name}: Saved metadata: {path}")
#         return path
# 
#     def save_pdf(self, pdf_path: Path) -> Path:
#         """Copy PDF to paper directory as main.pdf
# 
#         Args:
#             pdf_path: Source PDF file path
# 
#         Returns
#         -------
#             Path to main.pdf in paper directory
#         """
#         pdf_path = Path(pdf_path)
#         if not pdf_path.exists():
#             raise FileNotFoundError(f"PDF not found: {pdf_path}")
# 
#         dest = self.get_pdf_path()
#         shutil.copy2(pdf_path, dest)
# 
#         # Update paper object
#         self.paper.metadata.path.pdfs = [str(dest)]
#         self.paper.container.pdf_size_bytes = dest.stat().st_size
# 
#         logger.debug(f"{self.name}: Saved PDF: {dest}")
#         return dest
# 
#     def save_text(self, text: str) -> Path:
#         """Save extracted text to content.txt
# 
#         Args:
#             text: Extracted text content
# 
#         Returns
#         -------
#             Path to content.txt
#         """
#         path = self.get_text_path()
#         with open(path, "w", encoding="utf-8") as f:
#             f.write(text)
#         logger.debug(f"{self.name}: Saved text: {path}")
#         return path
# 
#     def save_tables(self, tables: List[Any]) -> Path:
#         """Save extracted tables to tables.json
# 
#         Args:
#             tables: List of table data
# 
#         Returns
#         -------
#             Path to tables.json
#         """
#         path = self.get_tables_path()
#         with open(path, "w") as f:
#             json.dump(tables, f, indent=2)
#         logger.debug(f"{self.name}: Saved {len(tables)} tables: {path}")
#         return path
# 
#     def save_tables_from_extraction(self, tables_dict: Dict[int, List[Any]]) -> Path:
#         """Save tables from PDF extraction (converts DataFrames to JSON).
# 
#         Args:
#             tables_dict: Dict[page_num, List[DataFrame]] from PDF extraction
# 
#         Returns
#         -------
#             Path to tables.json
#         """
#         tables_data = []
#         for page_num, page_tables in tables_dict.items():
#             for idx, df in enumerate(page_tables):
#                 try:
#                     entry = {
#                         "page": page_num,
#                         "index": idx,
#                         "columns": list(df.columns) if hasattr(df, "columns") else [],
#                         "data": df.to_dict(orient="records") if hasattr(df, "to_dict") else [],
#                         "shape": list(df.shape) if hasattr(df, "shape") else [0, 0],
#                     }
#                     tables_data.append(entry)
#                 except Exception as e:
#                     logger.warning(f"{self.name}: Could not convert table {page_num}:{idx}: {e}")
#         return self.save_tables(tables_data)
# 
#     def save_image(self, image_data: bytes, filename: str) -> Path:
#         """Save extracted image to images/ directory
# 
#         Args:
#             image_data: Image bytes
#             filename: Image filename (e.g., "fig1.png")
# 
#         Returns
#         -------
#             Path to saved image
#         """
#         images_dir = self.get_images_dir()
#         path = images_dir / filename
#         with open(path, "wb") as f:
#             f.write(image_data)
#         logger.debug(f"{self.name}: Saved image: {path}")
#         return path
# 
#     # ========================================
#     # Load Methods
#     # ========================================
#     def load_metadata(self) -> Paper:
#         """Load Paper from metadata.json and update internal reference.
# 
#         Returns
#         -------
#             Paper object
#         """
#         path = self.get_metadata_path()
#         if not path.exists():
#             raise FileNotFoundError(f"Metadata not found: {path}")
# 
#         with open(path) as f:
#             data = json.load(f)
# 
#         paper = Paper.from_dict(data)
#         # Update internal reference so save_metadata() uses the loaded paper
#         self.paper = paper
#         logger.info(f"{self.name}: Loaded metadata: {path}")
#         return paper
# 
#     def load_text(self) -> str:
#         """Load extracted text from content.txt
# 
#         Returns
#         -------
#             Text content
#         """
#         path = self.get_text_path()
#         if not path.exists():
#             raise FileNotFoundError(f"Text not found: {path}")
# 
#         with open(path, encoding="utf-8") as f:
#             text = f.read()
#         logger.info(f"{self.name}: Loaded text: {path}")
#         return text
# 
#     def load_tables(self) -> List[Any]:
#         """Load extracted tables from tables.json
# 
#         Returns
#         -------
#             List of tables
#         """
#         path = self.get_tables_path()
#         if not path.exists():
#             raise FileNotFoundError(f"Tables not found: {path}")
# 
#         with open(path) as f:
#             tables = json.load(f)
#         logger.info(f"{self.name}: Loaded {len(tables)} tables: {path}")
#         return tables
# 
#     # ========================================
#     # Utility Methods
#     # ========================================
#     def get_all_files(self) -> Dict[str, bool]:
#         """Get status of all expected files
# 
#         Returns
#         -------
#             Dictionary of actual filename: exists (shows real PDF name if exists)
#         """
#         # Get actual PDF filename using PathManager getter
#         pdf_files = list(self.paper_dir.glob("*.pdf"))
#         if pdf_files:
#             # Show actual PDF filename
#             pdf_key = pdf_files[0].name
#         else:
#             # Show expected PDF filename from PathManager
#             expected_pdf = self.get_pdf_path()
#             pdf_key = expected_pdf.name
# 
#         return {
#             "metadata.json": self.has_metadata(),
#             pdf_key: self.has_pdf(),  # Shows actual PDF filename
#             "content.txt": self.has_content(),
#             "tables.json": self.has_tables(),
#             "images/": self.get_images_dir().exists(),
#             "screenshots/": self.get_screenshots_dir().exists(),
#         }
# 
#     def __repr__(self) -> str:
#         """String representation"""
#         return f"PaperIO({self.paper.container.library_id}, {self.paper_dir})"
# 
# 
# def main(args):
#     """Demo: PaperIO usage with worker pattern"""
#     from scitex.scholar.core import Paper
# 
#     logger.info("PaperIO Demo - Worker Pattern")
# 
#     # Create paper
#     paper = Paper()
#     paper.metadata.id.doi = "10.1212/WNL.0000000000200348"
#     paper.metadata.id.doi_engines = ["demo"]
#     paper.container.library_id = "A1B2C3D4"
# 
#     logger.info(f"Paper DOI: {paper.metadata.id.doi}")
#     logger.info(f"Library ID: {paper.container.library_id}")
# 
#     # Initialize IO
#     io = PaperIO(paper)
#     logger.info(f"Paper directory: {io.paper_dir}")
# 
#     # Worker 1: Metadata
#     if not io.has_metadata():
#         paper.metadata.basic.title = "Demo Paper"
#         paper.metadata.basic.title_engines = ["demo"]
#         io.save_metadata()
#         logger.success("Metadata saved")
#     else:
#         logger.info("Metadata exists, loading...")
#         paper = io.load_metadata()
# 
#     # Worker 2: PDF
#     if not io.has_pdf():
#         logger.info("No PDF found (would download here)")
#     else:
#         logger.success(f"PDF exists: {io.get_pdf_path().stat().st_size / 1e6:.2f} MB")
# 
#     # Status
#     status = io.get_all_files()
#     logger.info("File status:")
#     for filename, exists in status.items():
#         logger.info(f"  {'✓' if exists else '✗'} {filename}")
# 
#     return 0
# 
# 
# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments."""
#     parser = argparse.ArgumentParser(
#         description="PaperIO - Simple IO interface for Paper objects"
#     )
#     args = parser.parse_args()
#     return args
# 
# 
# def run_main() -> None:
#     """Initialize scitex framework, run main function, and cleanup."""
#     global CONFIG, CC, sys, plt, rng
# 
#     import sys
# 
#     import matplotlib.pyplot as plt
# 
#     import scitex as stx
# 
#     args = parse_args()
# 
#     CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
#         sys,
#         plt,
#         args=args,
#         file=__FILE__,
#         sdir_suffix=None,
#         verbose=False,
#         agg=True,
#     )
# 
#     exit_status = main(args)
# 
#     stx.session.close(
#         CONFIG,
#         verbose=False,
#         notify=False,
#         message="",
#         exit_status=exit_status,
#     )
# 
# 
# if __name__ == "__main__":
#     run_main()
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/scholar/storage/PaperIO.py
# --------------------------------------------------------------------------------
