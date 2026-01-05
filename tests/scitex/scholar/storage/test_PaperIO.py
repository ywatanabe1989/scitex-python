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

    pytest.main([os.path.abspath(__file__), "-v"])
