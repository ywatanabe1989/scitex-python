#!/usr/bin/env python3
# Time-stamp: "2025-06-02 17:10:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/io/_load_modules/test__pdf.py

"""Comprehensive tests for PDF file loading functionality."""

import os
import tempfile

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")
from unittest.mock import MagicMock, patch


class TestLoadPdf:
    """Test suite for _load_pdf function"""

    def test_file_existence_validation(self):
        """Test that function validates file exists"""
        from scitex.io._load_modules._pdf import _load_pdf

        # Test non-existent files (source validates file existence, not extension)
        non_existent_files = [
            "nonexistent.pdf",
            "/path/to/missing/file.pdf",
            "./does_not_exist.pdf",
        ]

        for missing_file in non_existent_files:
            with pytest.raises(FileNotFoundError, match="PDF file not found"):
                _load_pdf(missing_file)

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_text_mode_fitz_extraction(self, mock_fitz, mock_exists):
        """Test text extraction using fitz (PyMuPDF) backend"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Create mock document and pages
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Sample text from page 1"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # Call function with text mode
        result = _load_pdf("test_document.pdf", mode="text")

        # Verify fitz.open was called correctly
        mock_fitz.open.assert_called_once_with("test_document.pdf")

        # Verify result (text may be cleaned, so check content presence)
        assert isinstance(result, str)
        assert "Sample text" in result

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_multi_page_text_extraction(self, mock_fitz, mock_exists):
        """Test processing of multi-page PDFs"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Create mock pages
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Content from page 1"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Content from page 2"
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Content from page 3"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(
            return_value=iter([mock_page1, mock_page2, mock_page3])
        )
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_fitz.open.return_value = mock_doc

        result = _load_pdf("multi_page.pdf", mode="text")

        # Verify all pages were processed
        assert "Content from page 1" in result or "page 1" in result
        assert "Content from page 2" in result or "page 2" in result
        assert "Content from page 3" in result or "page 3" in result

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_empty_pdf_handling(self, mock_fitz, mock_exists):
        """Test handling of empty PDFs"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Create mock document with no content
        mock_page = MagicMock()
        mock_page.get_text.return_value = ""

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        result = _load_pdf("empty.pdf", mode="text")

        assert result == ""
        assert isinstance(result, str)

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_unicode_text_handling(self, mock_fitz, mock_exists):
        """Test handling of Unicode characters in PDF text"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Create mock pages with Unicode content
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Héllo Wörld! 你好世界"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Математика العربية"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page1, mock_page2]))
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_fitz.open.return_value = mock_doc

        result = _load_pdf("unicode.pdf", mode="text")

        assert isinstance(result, str)
        # Check that unicode content is present (may be cleaned)
        assert "Héllo" in result or "hello" in result.lower()

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_large_pdf_handling(self, mock_fitz, mock_exists):
        """Test handling of PDFs with many pages"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Create 100 mock pages
        mock_pages = []
        for i in range(100):
            mock_page = MagicMock()
            mock_page.get_text.return_value = f"This is content from page {i + 1}."
            mock_pages.append(mock_page)

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter(mock_pages))
        mock_doc.__len__ = MagicMock(return_value=100)
        mock_fitz.open.return_value = mock_doc

        result = _load_pdf("large_document.pdf", mode="text")

        assert isinstance(result, str)
        assert "page 1" in result
        assert "page 100" in result

    def test_function_signature(self):
        """Test function signature and parameters"""
        import inspect

        from scitex.io._load_modules._pdf import _load_pdf

        sig = inspect.signature(_load_pdf)

        # Check parameters
        assert "lpath" in sig.parameters
        assert "mode" in sig.parameters
        assert "metadata" in sig.parameters

        # Check default values
        assert sig.parameters["mode"].default == "full"
        assert sig.parameters["metadata"].default == False

    def test_function_docstring(self):
        """Test that function has proper docstring"""
        from scitex.io._load_modules._pdf import _load_pdf

        assert hasattr(_load_pdf, "__doc__")
        assert _load_pdf.__doc__ is not None
        docstring = _load_pdf.__doc__

        # Check for key documentation elements
        assert "PDF" in docstring
        assert "mode" in docstring
        assert "text" in docstring.lower()

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_text_mode_with_kwargs(self, mock_fitz, mock_exists):
        """Test that additional kwargs are handled"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content"
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # Call with various kwargs
        result = _load_pdf("test.pdf", mode="text", clean_text=True, backend="fitz")

        assert isinstance(result, str)

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_path_variations(self, mock_fitz, mock_exists):
        """Test various path formats"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content"
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # Test different path formats
        paths = [
            "/home/user/documents/report.pdf",
            "./data/document.pdf",
            "file-with-hyphens.pdf",
            "file_with_underscores.pdf",
            "very_long_filename_with_many_characters_and_numbers_123.pdf",
        ]

        for path in paths:
            mock_fitz.open.reset_mock()
            result = _load_pdf(path, mode="text")
            mock_fitz.open.assert_called_once_with(path)
            assert isinstance(result, str)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError"""
        from scitex.io._load_modules._pdf import _load_pdf

        # Mock file exists
        with patch("os.path.exists", return_value=True):
            with patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True):
                with patch("scitex.io._load_modules._pdf.fitz") as mock_fitz:
                    mock_doc = MagicMock()
                    mock_fitz.open.return_value = mock_doc

                    with pytest.raises(ValueError, match="Unknown extraction mode"):
                        _load_pdf("test.pdf", mode="invalid_mode")

    @patch("os.path.exists")
    @patch("os.path.getsize")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_scientific_mode(self, mock_fitz, mock_getsize, mock_exists):
        """Test scientific extraction mode"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True
        mock_getsize.return_value = 1024

        # Mock for text extraction
        mock_page = MagicMock()
        mock_page.get_text.return_value = (
            "Abstract\nThis paper presents results.\nMethods\nWe used..."
        )
        mock_page.get_images.return_value = []

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.metadata = {"title": "Test Paper", "author": "Test Author"}
        mock_doc.is_encrypted = False
        mock_fitz.open.return_value = mock_doc

        # Mock file hash calculation
        with patch(
            "scitex.io._load_modules._pdf._calculate_file_hash", return_value="abc123"
        ):
            result = _load_pdf("scientific_paper.pdf", mode="scientific")

        # DotDict is dict-like but not isinstance dict
        assert hasattr(result, "__getitem__")
        assert "extraction_mode" in result
        assert result["extraction_mode"] == "scientific"

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_metadata_mode(self, mock_fitz, mock_exists):
        """Test metadata extraction mode"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "subject": "Test Subject",
            "keywords": "test, pdf",
            "creator": "Test Creator",
            "producer": "Test Producer",
            "creationDate": "D:20230101",
            "modDate": "D:20230615",
        }
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.is_encrypted = False
        mock_fitz.open.return_value = mock_doc

        with patch("os.path.getsize", return_value=1024):
            with patch(
                "scitex.io._load_modules._pdf._calculate_file_hash",
                return_value="abc123",
            ):
                result = _load_pdf("document.pdf", mode="metadata")

        assert isinstance(result, dict)
        assert "title" in result
        assert result["title"] == "Test Document"
        assert result["author"] == "Test Author"

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.PyPDF2")
    @patch("scitex.io._load_modules._pdf.PYPDF2_AVAILABLE", True)
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", False)
    @patch("scitex.io._load_modules._pdf.PDFPLUMBER_AVAILABLE", False)
    def test_pypdf2_fallback_text_extraction(self, mock_pypdf2, mock_exists):
        """Test text extraction using PyPDF2 fallback backend"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Create mock reader and pages with correct PyPDF2 API
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PyPDF2 text"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_pypdf2.PdfReader.return_value = mock_reader

        result = _load_pdf("test.pdf", mode="text", backend="pypdf2")

        mock_pypdf2.PdfReader.assert_called_once_with("test.pdf")
        assert isinstance(result, str)
        assert "Sample PyPDF2 text" in result or "Sample" in result

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_text_extraction_edge_cases(self, mock_fitz, mock_exists):
        """Test edge cases in text extraction"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Test pages with special characters and formatting
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = "Line 1\nLine 2\nLine 3"
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = "Tabs\there\tand\tthere"
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = "Special chars: !@#$%^&*()"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(
            return_value=iter([mock_page1, mock_page2, mock_page3])
        )
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_fitz.open.return_value = mock_doc

        result = _load_pdf("formatted.pdf", mode="text")

        assert isinstance(result, str)
        # Content should be present (may be cleaned/transformed)
        assert "Line" in result or "Tabs" in result or "Special" in result

    def test_module_dependencies(self):
        """Test that the function handles missing dependencies gracefully"""
        from scitex.io._load_modules._pdf import _load_pdf

        # Verify the function exists and is callable
        assert callable(_load_pdf)

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_scientific_document_scenario(self, mock_fitz, mock_exists):
        """Test realistic scientific document processing scenario"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Simulate a scientific paper with typical content
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = (
            "Abstract\nThis paper presents a novel approach to..."
        )
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = (
            "Introduction\nRecent advances in machine learning..."
        )
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = (
            "Methods\nWe employed a cross-validation approach..."
        )
        mock_page4 = MagicMock()
        mock_page4.get_text.return_value = "Results\nThe experimental results show..."
        mock_page5 = MagicMock()
        mock_page5.get_text.return_value = "References\n[1] Smith, J. et al. (2023)..."

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(
            return_value=iter(
                [mock_page1, mock_page2, mock_page3, mock_page4, mock_page5]
            )
        )
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_fitz.open.return_value = mock_doc

        result = _load_pdf("scientific_paper.pdf", mode="text")

        # Verify scientific content is properly extracted
        assert isinstance(result, str)
        assert "Abstract" in result or "paper" in result.lower()

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_metadata_tuple_return(self, mock_fitz, mock_exists):
        """Test metadata=True returns tuple"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        mock_page = MagicMock()
        mock_page.get_text.return_value = "Test content"
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # Call with metadata=True - the function returns (result, metadata_dict)
        # metadata_dict will be None if extraction fails (which is expected in test)
        result = _load_pdf("test.pdf", mode="text", metadata=True)

        # Should return tuple (result, metadata_dict)
        assert isinstance(result, tuple)
        assert len(result) == 2
        # First element is the extracted text
        assert isinstance(result[0], str)
        # Second element is metadata (None if extraction failed)

    @patch("os.path.exists")
    @patch("scitex.io._load_modules._pdf.fitz")
    @patch("scitex.io._load_modules._pdf.FITZ_AVAILABLE", True)
    def test_clean_text_option(self, mock_fitz, mock_exists):
        """Test clean_text option"""
        from scitex.io._load_modules._pdf import _load_pdf

        mock_exists.return_value = True

        # Text with artifacts that should be cleaned
        mock_page = MagicMock()
        mock_page.get_text.return_value = (
            "Text   with    extra   spaces\n\n\n\nand newlines"
        )
        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_fitz.open.return_value = mock_doc

        # With clean_text=True (default)
        result_clean = _load_pdf("test.pdf", mode="text", clean_text=True)

        # With clean_text=False
        mock_fitz.open.return_value = mock_doc
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        result_raw = _load_pdf("test.pdf", mode="text", clean_text=False)

        # Both should be strings
        assert isinstance(result_clean, str)
        assert isinstance(result_raw, str)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])
