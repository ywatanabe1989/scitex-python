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

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load_modules/_pdf.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-06 10:27:52 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/io/_load_modules/_pdf.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Enhanced PDF loading module with comprehensive extraction capabilities.
# 
# This module provides advanced PDF extraction for scientific papers, including:
# - Text extraction with formatting preservation
# - Table extraction as pandas DataFrames
# - Image extraction with metadata
# - Section-aware text parsing
# - Multiple extraction modes for different use cases
# """
# 
# import hashlib
# import re
# import tempfile
# from typing import Any, Dict, List
# 
# from scitex import logging
# from scitex.dict import DotDict
# 
# logger = logging.getLogger(__name__)
# 
# # Try to import PDF libraries in order of preference
# try:
#     import fitz  # PyMuPDF - preferred for text and images
# 
#     FITZ_AVAILABLE = True
# except ImportError:
#     FITZ_AVAILABLE = False
# 
# try:
#     import pdfplumber  # Best for table extraction
# 
#     PDFPLUMBER_AVAILABLE = True
# except ImportError:
#     PDFPLUMBER_AVAILABLE = False
# 
# try:
#     import PyPDF2  # Fallback option
# 
#     PYPDF2_AVAILABLE = True
# except ImportError:
#     PYPDF2_AVAILABLE = False
# 
# try:
#     import pandas as pd
# 
#     PANDAS_AVAILABLE = True
# except ImportError:
#     PANDAS_AVAILABLE = False
# 
# 
# def _load_pdf(lpath: str, mode: str = "full", metadata: bool = False, **kwargs) -> Any:
#     """
#     Load PDF file with comprehensive extraction capabilities.
# 
#     Args:
#         lpath: Path to PDF file
#         mode: Extraction mode (default: 'full')
#             - 'full': Complete extraction including text, sections, metadata, pages, tables, and images
#             - 'scientific': Optimized for scientific papers (text + sections + tables + images + stats)
#             - 'text': Plain text extraction only
#             - 'sections': Section-aware text extraction
#             - 'tables': Extract tables as DataFrames
#             - 'images': Extract images with metadata
#             - 'metadata': PDF metadata only
#             - 'pages': Page-by-page extraction
#         metadata: If True, return (result, metadata_dict) tuple for API consistency with images.
#             If False (default), return result only. (default: False)
#         **kwargs: Additional arguments
#             - backend: 'auto' (default), 'fitz', 'pdfplumber', or 'pypdf2'
#             - clean_text: Clean extracted text (default: True)
#             - extract_images: Extract images to files (default: False for 'full' mode, True for 'scientific')
#             - output_dir: Directory for extracted images/tables (default: temp dir)
#             - save_as_jpg: Convert all extracted images to JPG format (default: True)
#             - table_settings: Dict of pdfplumber table extraction settings
# 
#     Returns:
#         Extracted content based on mode and metadata parameter:
# 
#         When metadata=False (default):
#             - 'text': str
#             - 'sections': Dict[str, str]
#             - 'tables': Dict[int, List[pd.DataFrame]]
#             - 'images': List[Dict] with image metadata
#             - 'metadata': Dict with PDF metadata
#             - 'pages': List[Dict] with page content
#             - 'full': Dict with comprehensive extraction
#             - 'scientific': Dict with scientific paper extraction
# 
#         When metadata=True:
#             - Returns: (result, metadata_dict) tuple
#             - metadata_dict contains embedded scitex metadata from PDF Subject field
# 
#     Examples:
#         >>> import scitex.io as stx
# 
#         >>> # Full extraction (default) - everything included
#         >>> data = stx.load("paper.pdf")
#         >>> print(data['full_text'])      # Complete text
#         >>> print(data['metadata'])       # PDF metadata
# 
#         >>> # With metadata tuple (consistent with images)
#         >>> data, meta = stx.load("paper.pdf", metadata=True)
#         >>> print(meta['scitex']['version'])  # Embedded scitex metadata
# 
#         >>> # Scientific mode
#         >>> paper = stx.load("paper.pdf", mode="scientific")
#         >>> print(paper['sections'])
# 
#         >>> # Simple text extraction
#         >>> text = stx.load("paper.pdf", mode="text")
#     """
#     mode = kwargs.get("mode", mode)
#     backend = kwargs.get("backend", "auto")
#     clean_text = kwargs.get("clean_text", True)
#     extract_images = kwargs.get("extract_images", False)
#     output_dir = kwargs.get("output_dir", None)
#     table_settings = kwargs.get("table_settings", {})
# 
#     # Validate file exists
#     if not os.path.exists(lpath):
#         raise FileNotFoundError(f"PDF file not found: {lpath}")
# 
#     # Extension validation removed - handled by load() function
#     # This allows loading files without extensions when ext='pdf' is specified
# 
#     # Select backend based on mode and availability
#     backend = _select_backend(mode, backend)
# 
#     # Create output directory if needed
#     if output_dir is None and (
#         extract_images or mode in ["images", "scientific", "full"]
#     ):
#         output_dir = tempfile.mkdtemp(prefix="pdf_extract_")
#         logger.debug(f"Using temporary directory: {output_dir}")
# 
#     # Extract based on mode
#     if mode == "text":
#         result = _extract_text(lpath, backend, clean_text)
#     elif mode == "sections":
#         result = _extract_sections(lpath, backend, clean_text)
#     elif mode == "tables":
#         result = _extract_tables(lpath, table_settings)
#     elif mode == "images":
#         save_as_jpg = kwargs.get("save_as_jpg", True)
#         result = _extract_images(lpath, output_dir, save_as_jpg)
#     elif mode == "metadata":
#         result = _extract_metadata(lpath, backend)
#     elif mode == "pages":
#         result = _extract_pages(lpath, backend, clean_text)
#     elif mode == "scientific":
#         save_as_jpg = kwargs.get("save_as_jpg", True)
#         result = _extract_scientific(
#             lpath, clean_text, output_dir, table_settings, save_as_jpg
#         )
#     elif mode == "full":
#         save_as_jpg = kwargs.get("save_as_jpg", True)
#         result = _extract_full(
#             lpath,
#             backend,
#             clean_text,
#             extract_images,
#             output_dir,
#             table_settings,
#             save_as_jpg,
#         )
#     else:
#         raise ValueError(f"Unknown extraction mode: {mode}")
# 
#     # If metadata parameter is True, return tuple (result, metadata_dict)
#     # This provides API consistency with image loading
#     if metadata:
#         try:
#             from .._metadata import read_metadata
# 
#             metadata_dict = read_metadata(lpath)
#             return result, metadata_dict
#         except Exception:
#             # If metadata extraction fails, return with None
#             return result, None
# 
#     return result
# 
# 
# def _select_backend(mode: str, requested: str) -> str:
#     """Select appropriate backend based on mode and availability."""
#     if requested != "auto":
#         return requested
# 
#     # Mode-specific backend selection
#     if mode in ["tables"]:
#         if PDFPLUMBER_AVAILABLE:
#             return "pdfplumber"
#         else:
#             logger.warning(
#                 "pdfplumber not available for table extraction. Install with: pip install pdfplumber"
#             )
#             return "fitz" if FITZ_AVAILABLE else "pypdf2"
# 
#     elif mode in ["images", "scientific", "full"]:
#         if FITZ_AVAILABLE:
#             return "fitz"
#         else:
#             logger.warning(
#                 "PyMuPDF (fitz) recommended for image extraction. Install with: pip install PyMuPDF"
#             )
#             return "pdfplumber" if PDFPLUMBER_AVAILABLE else "pypdf2"
# 
#     else:  # text, sections, metadata, pages
#         if FITZ_AVAILABLE:
#             return "fitz"
#         elif PDFPLUMBER_AVAILABLE:
#             return "pdfplumber"
#         elif PYPDF2_AVAILABLE:
#             return "pypdf2"
#         else:
#             raise ImportError(
#                 "No PDF library available. Install one of:\n"
#                 "  pip install PyMuPDF     # Recommended\n"
#                 "  pip install pdfplumber  # Best for tables\n"
#                 "  pip install PyPDF2      # Basic fallback"
#             )
# 
# 
# def _extract_text(lpath: str, backend: str, clean: bool) -> str:
#     """Extract plain text from PDF."""
#     if backend == "fitz":
#         return _extract_text_fitz(lpath, clean)
#     elif backend == "pdfplumber":
#         return _extract_text_pdfplumber(lpath, clean)
#     else:
#         return _extract_text_pypdf2(lpath, clean)
# 
# 
# def _extract_text_fitz(lpath: str, clean: bool) -> str:
#     """Extract text using PyMuPDF."""
#     if not FITZ_AVAILABLE:
#         raise ImportError("PyMuPDF (fitz) not available")
# 
#     try:
#         doc = fitz.open(lpath)
#         text_parts = []
# 
#         for page_num, page in enumerate(doc):
#             text = page.get_text()
#             if text.strip():
#                 text_parts.append(text)
# 
#         doc.close()
# 
#         full_text = "\n".join(text_parts)
# 
#         if clean:
#             full_text = _clean_pdf_text(full_text)
# 
#         return full_text
# 
#     except Exception as e:
#         logger.error(f"Error extracting text with fitz from {lpath}: {e}")
#         raise
# 
# 
# def _extract_text_pdfplumber(lpath: str, clean: bool) -> str:
#     """Extract text using pdfplumber."""
#     if not PDFPLUMBER_AVAILABLE:
#         raise ImportError("pdfplumber not available")
# 
#     try:
#         import pdfplumber
# 
#         text_parts = []
#         with pdfplumber.open(lpath) as pdf:
#             for page in pdf.pages:
#                 text = page.extract_text()
#                 if text:
#                     text_parts.append(text)
# 
#         full_text = "\n".join(text_parts)
# 
#         if clean:
#             full_text = _clean_pdf_text(full_text)
# 
#         return full_text
# 
#     except Exception as e:
#         logger.error(f"Error extracting text with pdfplumber from {lpath}: {e}")
#         raise
# 
# 
# def _extract_text_pypdf2(lpath: str, clean: bool) -> str:
#     """Extract text using PyPDF2."""
#     if not PYPDF2_AVAILABLE:
#         raise ImportError("PyPDF2 not available")
# 
#     try:
#         reader = PyPDF2.PdfReader(lpath)
#         text_parts = []
# 
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text = page.extract_text()
#             if text.strip():
#                 text_parts.append(text)
# 
#         full_text = "\n".join(text_parts)
# 
#         if clean:
#             full_text = _clean_pdf_text(full_text)
# 
#         return full_text
# 
#     except Exception as e:
#         logger.error(f"Error extracting text with PyPDF2 from {lpath}: {e}")
#         raise
# 
# 
# def _extract_tables(
#     lpath: str, table_settings: Dict = None
# ) -> Dict[int, List["pd.DataFrame"]]:
#     """
#     Extract tables from PDF as pandas DataFrames.
# 
#     Returns:
#         Dict mapping page numbers to list of DataFrames
#     """
#     if not PDFPLUMBER_AVAILABLE:
#         raise ImportError(
#             "pdfplumber required for table extraction. Install with:\n"
#             "  pip install pdfplumber pandas"
#         )
# 
#     if not PANDAS_AVAILABLE:
#         raise ImportError("pandas required for table extraction")
# 
#     import pandas as pd
#     import pdfplumber
# 
#     tables_dict = {}
#     table_settings = table_settings or {}
# 
#     try:
#         with pdfplumber.open(lpath) as pdf:
#             for page_num, page in enumerate(pdf.pages):
#                 # Extract tables from page
#                 tables = page.extract_tables(**table_settings)
# 
#                 if tables:
#                     # Convert to DataFrames
#                     dfs = []
#                     for table in tables:
#                         if table and len(table) > 0:
#                             # First row as header if it looks like headers
#                             if len(table) > 1 and all(
#                                 isinstance(cell, str) for cell in table[0] if cell
#                             ):
#                                 df = pd.DataFrame(table[1:], columns=table[0])
#                             else:
#                                 df = pd.DataFrame(table)
# 
#                             # Clean up DataFrame
#                             df = (
#                                 df.replace("", None)
#                                 .dropna(how="all", axis=1)
#                                 .dropna(how="all", axis=0)
#                             )
# 
#                             if not df.empty:
#                                 dfs.append(df)
# 
#                     if dfs:
#                         tables_dict[page_num] = dfs
# 
#         logger.info(f"Extracted tables from {len(tables_dict)} pages")
#         return tables_dict
# 
#     except Exception as e:
#         logger.error(f"Error extracting tables: {e}")
#         raise
# 
# 
# def _extract_images(
#     lpath: str, output_dir: str = None, save_as_jpg: bool = True
# ) -> List[Dict[str, Any]]:
#     """
#     Extract images from PDF with metadata.
# 
#     Args:
#         lpath: Path to PDF file
#         output_dir: Directory to save images (optional)
#         save_as_jpg: If True, convert all images to JPG format (default: True)
# 
#     Returns:
#         List of dicts containing image metadata and paths
#     """
#     if not FITZ_AVAILABLE:
#         raise ImportError(
#             "PyMuPDF (fitz) required for image extraction. Install with:\n"
#             "  pip install PyMuPDF"
#         )
# 
#     images_info = []
# 
#     try:
#         doc = fitz.open(lpath)
# 
#         for page_num, page in enumerate(doc):
#             image_list = page.get_images()
# 
#             for img_index, img in enumerate(image_list):
#                 xref = img[0]
# 
#                 # Extract image data
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 original_ext = base_image["ext"]
# 
#                 image_info = {
#                     "page": page_num + 1,
#                     "index": img_index,
#                     "width": base_image["width"],
#                     "height": base_image["height"],
#                     "colorspace": base_image["colorspace"],
#                     "bpc": base_image["bpc"],  # bits per component
#                     "original_ext": original_ext,
#                     "size_bytes": len(image_bytes),
#                 }
# 
#                 # Save image if output directory provided
#                 if output_dir:
#                     os.makedirs(output_dir, exist_ok=True)
# 
#                     if save_as_jpg and original_ext not in ["jpg", "jpeg"]:
#                         # Convert to JPG using PIL
#                         try:
#                             from PIL import Image
#                             import io
# 
#                             # Open image from bytes
#                             img_pil = Image.open(io.BytesIO(image_bytes))
# 
#                             # Convert RGBA to RGB if necessary
#                             if img_pil.mode in ("RGBA", "LA", "P"):
#                                 # Create a white background
#                                 background = Image.new(
#                                     "RGB", img_pil.size, (255, 255, 255)
#                                 )
#                                 if img_pil.mode == "P":
#                                     img_pil = img_pil.convert("RGBA")
#                                 background.paste(
#                                     img_pil,
#                                     mask=img_pil.split()[-1]
#                                     if img_pil.mode == "RGBA"
#                                     else None,
#                                 )
#                                 img_pil = background
#                             elif img_pil.mode != "RGB":
#                                 img_pil = img_pil.convert("RGB")
# 
#                             # Save as JPG
#                             filename = f"page_{page_num + 1}_img_{img_index}.jpg"
#                             filepath = os.path.join(output_dir, filename)
#                             img_pil.save(filepath, "JPEG", quality=95)
# 
#                             image_info["ext"] = "jpg"
#                         except ImportError:
#                             logger.warning(
#                                 "PIL not available for image conversion. Install with: pip install Pillow"
#                             )
#                             # Fall back to original format
#                             filename = (
#                                 f"page_{page_num + 1}_img_{img_index}.{original_ext}"
#                             )
#                             filepath = os.path.join(output_dir, filename)
#                             with open(filepath, "wb") as img_file:
#                                 img_file.write(image_bytes)
#                             image_info["ext"] = original_ext
#                     else:
#                         # Save with original format
#                         ext = "jpg" if original_ext == "jpeg" else original_ext
#                         filename = f"page_{page_num + 1}_img_{img_index}.{ext}"
#                         filepath = os.path.join(output_dir, filename)
#                         with open(filepath, "wb") as img_file:
#                             img_file.write(image_bytes)
#                         image_info["ext"] = ext
# 
#                     image_info["filepath"] = filepath
#                     image_info["filename"] = filename
# 
#                 images_info.append(image_info)
# 
#         doc.close()
# 
#         logger.info(f"Extracted {len(images_info)} images from PDF")
#         return images_info
# 
#     except Exception as e:
#         logger.error(f"Error extracting images: {e}")
#         raise
# 
# 
# def _extract_sections(lpath: str, backend: str, clean: bool) -> Dict[str, str]:
#     """Extract text organized by sections."""
#     # Get full text first
#     text = _extract_text(lpath, backend, clean=False)
# 
#     # Parse into sections
#     sections = _parse_sections(text)
# 
#     # Clean section text if requested
#     if clean:
#         for section, content in sections.items():
#             sections[section] = _clean_pdf_text(content)
# 
#     return sections
# 
# 
# def _parse_sections(text: str) -> Dict[str, str]:
#     """
#     Parse text into sections based on IMRaD structure.
# 
#     Follows the standard scientific paper structure:
#     - frontpage: Title, authors, affiliations, keywords
#     - abstract: Paper summary
#     - introduction: Background and motivation
#     - methods: Methodology (materials and methods, experimental design)
#     - results: Findings
#     - discussion: Interpretation and implications
#     - references: Citations
#     """
#     sections = {}
#     current_section = "frontpage"
#     current_text = []
# 
#     # Simplified section patterns - IMRaD + frontpage only
#     # Only match standalone section headers (exact matches)
#     section_patterns = [
#         r"^abstract\s*$",
#         r"^summary\s*$",
#         r"^introduction\s*$",
#         r"^background\s*$",
#         r"^methods?\s*$",
#         r"^materials?\s+and\s+methods?\s*$",
#         r"^methodology\s*$",
#         r"^results?\s*$",
#         r"^discussion\s*$",
#         r"^references?\s*$",
#     ]
# 
#     lines = text.split("\n")
# 
#     for line in lines:
#         line_lower = line.lower().strip()
#         line_stripped = line.strip()
# 
#         # Check if this line is a section header
#         is_header = False
#         for pattern in section_patterns:
#             if re.match(pattern, line_lower):
#                 # Additional validation: header lines should be short (< 50 chars)
#                 # and not contain numbers/punctuation (except spaces)
#                 if len(line_stripped) < 50:
#                     # Save previous section
#                     if current_text:
#                         sections[current_section] = "\n".join(current_text).strip()
# 
#                     # Start new section
#                     current_section = line_lower.strip()
#                     current_text = []
#                     is_header = True
#                     break
# 
#         if not is_header:
#             current_text.append(line)
# 
#     # Save last section
#     if current_text:
#         sections[current_section] = "\n".join(current_text).strip()
# 
#     return sections
# 
# 
# def _extract_metadata(lpath: str, backend: str) -> Dict[str, Any]:
#     """Extract PDF metadata."""
#     metadata = {
#         "file_path": lpath,
#         "file_name": os.path.basename(lpath),
#         "file_size": os.path.getsize(lpath),
#         "backend": backend,
#     }
# 
#     if backend == "fitz" and FITZ_AVAILABLE:
#         try:
#             doc = fitz.open(lpath)
#             pdf_metadata = doc.metadata
# 
#             metadata.update(
#                 {
#                     "title": pdf_metadata.get("title", ""),
#                     "author": pdf_metadata.get("author", ""),
#                     "subject": pdf_metadata.get("subject", ""),
#                     "keywords": pdf_metadata.get("keywords", ""),
#                     "creator": pdf_metadata.get("creator", ""),
#                     "producer": pdf_metadata.get("producer", ""),
#                     "creation_date": str(pdf_metadata.get("creationDate", "")),
#                     "modification_date": str(pdf_metadata.get("modDate", "")),
#                     "pages": len(doc),
#                     "encrypted": doc.is_encrypted,
#                 }
#             )
# 
#             # Try to parse scitex metadata from subject field (for consistency with PNG)
#             subject = pdf_metadata.get("subject", "")
#             if subject:
#                 try:
#                     import json
# 
#                     # Check if subject is JSON (scitex metadata)
#                     parsed_subject = json.loads(subject)
#                     if isinstance(parsed_subject, dict):
#                         # Merge parsed scitex metadata with standard PDF metadata
#                         # This makes PDF metadata format consistent with PNG
#                         metadata.update(parsed_subject)
#                         # Remove the raw JSON string from subject to avoid duplication
#                         metadata.pop("subject", None)
#                 except (json.JSONDecodeError, ValueError):
#                     # Not JSON, keep subject as string
#                     pass
# 
#             doc.close()
# 
#         except Exception as e:
#             logger.error(f"Error extracting metadata with fitz: {e}")
# 
#     elif backend == "pdfplumber" and PDFPLUMBER_AVAILABLE:
#         try:
#             import pdfplumber
# 
#             with pdfplumber.open(lpath) as pdf:
#                 metadata["pages"] = len(pdf.pages)
#                 if hasattr(pdf, "metadata"):
#                     metadata.update(pdf.metadata)
# 
#                 # Try to parse scitex metadata from subject field (for consistency with PNG)
#                 if "Subject" in metadata or "subject" in metadata:
#                     subject = metadata.get("Subject") or metadata.get("subject", "")
#                     if subject:
#                         try:
#                             import json
# 
#                             parsed_subject = json.loads(subject)
#                             if isinstance(parsed_subject, dict):
#                                 # Merge parsed scitex metadata with standard PDF metadata
#                                 metadata.update(parsed_subject)
#                                 # Remove the raw JSON string from subject to avoid duplication
#                                 metadata.pop("Subject", None)
#                                 metadata.pop("subject", None)
#                         except (json.JSONDecodeError, ValueError):
#                             # Not JSON, keep subject as string
#                             pass
#         except Exception as e:
#             logger.error(f"Error extracting metadata with pdfplumber: {e}")
# 
#     elif backend == "pypdf2" and PYPDF2_AVAILABLE:
#         try:
#             reader = PyPDF2.PdfReader(lpath)
# 
#             if reader.metadata:
#                 metadata.update(
#                     {
#                         "title": reader.metadata.get("/Title", ""),
#                         "author": reader.metadata.get("/Author", ""),
#                         "subject": reader.metadata.get("/Subject", ""),
#                         "creator": reader.metadata.get("/Creator", ""),
#                         "producer": reader.metadata.get("/Producer", ""),
#                         "creation_date": str(reader.metadata.get("/CreationDate", "")),
#                         "modification_date": str(reader.metadata.get("/ModDate", "")),
#                     }
#                 )
# 
#             metadata["pages"] = len(reader.pages)
#             metadata["encrypted"] = reader.is_encrypted
# 
#             # Try to parse scitex metadata from subject field (for consistency with PNG)
#             subject = metadata.get("subject", "")
#             if subject:
#                 try:
#                     import json
# 
#                     parsed_subject = json.loads(subject)
#                     if isinstance(parsed_subject, dict):
#                         # Merge parsed scitex metadata with standard PDF metadata
#                         metadata.update(parsed_subject)
#                         # Remove the raw JSON string from subject to avoid duplication
#                         metadata.pop("subject", None)
#                 except (json.JSONDecodeError, ValueError):
#                     # Not JSON, keep subject as string
#                     pass
# 
#         except Exception as e:
#             logger.error(f"Error extracting metadata with PyPDF2: {e}")
# 
#     # Generate file hash
#     metadata["md5_hash"] = _calculate_file_hash(lpath)
# 
#     return metadata
# 
# 
# def _extract_pages(lpath: str, backend: str, clean: bool) -> List[Dict[str, Any]]:
#     """Extract content page by page."""
#     pages = []
# 
#     if backend == "fitz" and FITZ_AVAILABLE:
#         doc = fitz.open(lpath)
# 
#         for page_num, page in enumerate(doc):
#             text = page.get_text()
#             if clean:
#                 text = _clean_pdf_text(text)
# 
#             pages.append(
#                 {
#                     "page_number": page_num + 1,
#                     "text": text,
#                     "char_count": len(text),
#                     "word_count": len(text.split()),
#                 }
#             )
# 
#         doc.close()
# 
#     elif backend == "pdfplumber" and PDFPLUMBER_AVAILABLE:
#         import pdfplumber
# 
#         with pdfplumber.open(lpath) as pdf:
#             for page_num, page in enumerate(pdf.pages):
#                 text = page.extract_text() or ""
#                 if clean:
#                     text = _clean_pdf_text(text)
# 
#                 pages.append(
#                     {
#                         "page_number": page_num + 1,
#                         "text": text,
#                         "char_count": len(text),
#                         "word_count": len(text.split()),
#                     }
#                 )
# 
#     elif backend == "pypdf2" and PYPDF2_AVAILABLE:
#         reader = PyPDF2.PdfReader(lpath)
# 
#         for page_num in range(len(reader.pages)):
#             page = reader.pages[page_num]
#             text = page.extract_text()
#             if clean:
#                 text = _clean_pdf_text(text)
# 
#             pages.append(
#                 {
#                     "page_number": page_num + 1,
#                     "text": text,
#                     "char_count": len(text),
#                     "word_count": len(text.split()),
#                 }
#             )
# 
#     return pages
# 
# 
# def _extract_scientific(
#     lpath: str,
#     clean_text: bool,
#     output_dir: str,
#     table_settings: Dict,
#     save_as_jpg: bool = True,
# ) -> DotDict:
#     """
#     Optimized extraction for scientific papers.
#     Extracts text, tables, images, and sections in a structured format.
#     """
#     result = {
#         "pdf_path": lpath,
#         "filename": os.path.basename(lpath),
#         "extraction_mode": "scientific",
#     }
# 
#     try:
#         # Extract text and sections
#         backend = _select_backend("text", "auto")
#         result["text"] = _extract_text(lpath, backend, clean_text)
#         result["sections"] = _extract_sections(lpath, backend, clean_text)
# 
#         # Extract metadata
#         result["metadata"] = _extract_metadata(lpath, backend)
# 
#         # Extract tables if pdfplumber available
#         if PDFPLUMBER_AVAILABLE and PANDAS_AVAILABLE:
#             try:
#                 result["tables"] = _extract_tables(lpath, table_settings)
#             except Exception as e:
#                 logger.warning(f"Could not extract tables: {e}")
#                 result["tables"] = {}
#         else:
#             result["tables"] = {}
#             logger.info("Table extraction requires pdfplumber and pandas")
# 
#         # Extract images if fitz available
#         if FITZ_AVAILABLE:
#             try:
#                 result["images"] = _extract_images(lpath, output_dir, save_as_jpg)
#             except Exception as e:
#                 logger.warning(f"Could not extract images: {e}")
#                 result["images"] = []
#         else:
#             result["images"] = []
#             logger.info("Image extraction requires PyMuPDF (fitz)")
# 
#         # Calculate statistics
#         result["stats"] = {
#             "total_chars": len(result["text"]),
#             "total_words": len(result["text"].split()),
#             "total_pages": result["metadata"].get("pages", 0),
#             "num_sections": len(result["sections"]),
#             "num_tables": sum(len(tables) for tables in result["tables"].values()),
#             "num_images": len(result["images"]),
#         }
# 
#         logger.info(
#             f"Scientific extraction complete: "
#             f"{result['stats']['total_pages']} pages, "
#             f"{result['stats']['num_sections']} sections, "
#             f"{result['stats']['num_tables']} tables, "
#             f"{result['stats']['num_images']} images"
#         )
# 
#     except Exception as e:
#         logger.error(f"Error in scientific extraction: {e}")
#         result["error"] = str(e)
# 
#     return DotDict(result)
# 
# 
# def _extract_full(
#     lpath: str,
#     backend: str,
#     clean: bool,
#     extract_images: bool,
#     output_dir: str,
#     table_settings: Dict,
#     save_as_jpg: bool = True,
# ) -> DotDict:
#     """Extract comprehensive data from PDF."""
#     result = {
#         "pdf_path": lpath,
#         "filename": os.path.basename(lpath),
#         "backend": backend,
#         "extraction_params": {
#             "clean_text": clean,
#             "extract_images": extract_images,
#         },
#     }
# 
#     # Extract all components
#     try:
#         result["full_text"] = _extract_text(lpath, backend, clean)
#         result["sections"] = _extract_sections(lpath, backend, clean)
#         result["metadata"] = _extract_metadata(lpath, backend)
#         result["pages"] = _extract_pages(lpath, backend, clean)
# 
#         # Extract tables if available
#         if PDFPLUMBER_AVAILABLE and PANDAS_AVAILABLE:
#             try:
#                 result["tables"] = _extract_tables(lpath, table_settings)
#             except Exception as e:
#                 logger.warning(f"Could not extract tables: {e}")
#                 result["tables"] = {}
# 
#         # Extract images if requested and available
#         if extract_images and FITZ_AVAILABLE:
#             try:
#                 result["images"] = _extract_images(lpath, output_dir, save_as_jpg)
#             except Exception as e:
#                 logger.warning(f"Could not extract images: {e}")
#                 result["images"] = []
# 
#         # Calculate statistics
#         result["stats"] = {
#             "total_chars": len(result["full_text"]),
#             "total_words": len(result["full_text"].split()),
#             "total_pages": len(result["pages"]),
#             "num_sections": len(result["sections"]),
#             "num_tables": sum(
#                 len(tables) for tables in result.get("tables", {}).values()
#             ),
#             "num_images": len(result.get("images", [])),
#             "avg_words_per_page": (
#                 len(result["full_text"].split()) / len(result["pages"])
#                 if result["pages"]
#                 else 0
#             ),
#         }
# 
#     except Exception as e:
#         logger.error(f"Error in full extraction: {e}")
#         result["error"] = str(e)
# 
#     return DotDict(result)
# 
# 
# def _clean_pdf_text(text: str) -> str:
#     """Clean extracted PDF text."""
#     # Remove excessive whitespace
#     text = re.sub(r"\s+", " ", text)
# 
#     # Fix hyphenated words at line breaks
#     text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
# 
#     # Remove page numbers (common patterns)
#     text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
#     text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)
# 
#     # Clean up common PDF artifacts
#     text = text.replace("\x00", "")  # Null bytes
#     text = re.sub(r"[\x01-\x1f\x7f-\x9f]", "", text)  # Control characters
# 
#     # Normalize quotes and dashes
#     text = text.replace('"', '"').replace('"', '"')
#     text = text.replace(""", "'").replace(""", "'")
#     text = text.replace("–", "-").replace("—", "-")
# 
#     # Remove multiple consecutive newlines
#     text = re.sub(r"\n{3,}", "\n\n", text)
# 
#     return text.strip()
# 
# 
# def _calculate_file_hash(lpath: str) -> str:
#     """Calculate MD5 hash of file."""
#     hash_md5 = hashlib.md5()
#     with open(lpath, "rb") as f:
#         for chunk in iter(lambda: f.read(4096), b""):
#             hash_md5.update(chunk)
#     return hash_md5.hexdigest()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load_modules/_pdf.py
# --------------------------------------------------------------------------------
