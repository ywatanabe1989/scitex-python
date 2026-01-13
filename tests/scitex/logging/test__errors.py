#!/usr/bin/env python3
"""Tests for scitex.logging._errors module."""

import os
import tempfile

import pytest


class TestSciTeXError:
    """Test base SciTeXError class."""

    def test_scitex_error_basic_message(self):
        """Test SciTeXError with just a message."""
        from scitex.logging._errors import SciTeXError

        error = SciTeXError("Test error")
        assert "Test error" in str(error)
        assert error.message == "Test error"

    def test_scitex_error_with_context(self):
        """Test SciTeXError with context dictionary."""
        from scitex.logging._errors import SciTeXError

        error = SciTeXError("Test error", context={"key1": "value1", "key2": "value2"})
        assert "Context:" in str(error)
        assert "key1: value1" in str(error)
        assert "key2: value2" in str(error)
        assert error.context == {"key1": "value1", "key2": "value2"}

    def test_scitex_error_with_suggestion(self):
        """Test SciTeXError with suggestion."""
        from scitex.logging._errors import SciTeXError

        error = SciTeXError("Test error", suggestion="Try this instead")
        assert "Suggestion: Try this instead" in str(error)
        assert error.suggestion == "Try this instead"

    def test_scitex_error_full_format(self):
        """Test SciTeXError with all parameters."""
        from scitex.logging._errors import SciTeXError

        error = SciTeXError(
            "Test error",
            context={"file": "test.py"},
            suggestion="Check the file",
        )
        assert "SciTeX Error:" in str(error)
        assert "Test error" in str(error)
        assert "Context:" in str(error)
        assert "file: test.py" in str(error)
        assert "Suggestion: Check the file" in str(error)

    def test_scitex_error_empty_context(self):
        """Test SciTeXError with empty context."""
        from scitex.logging._errors import SciTeXError

        error = SciTeXError("Test error", context={})
        assert error.context == {}

    def test_scitex_error_is_exception(self):
        """Test that SciTeXError is an Exception."""
        from scitex.logging._errors import SciTeXError

        assert issubclass(SciTeXError, Exception)

    def test_scitex_error_can_be_raised(self):
        """Test that SciTeXError can be raised and caught."""
        from scitex.logging._errors import SciTeXError

        with pytest.raises(SciTeXError) as exc_info:
            raise SciTeXError("Test raise")

        assert "Test raise" in str(exc_info.value)


class TestConfigurationErrors:
    """Test configuration-related errors."""

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from SciTeXError."""
        from scitex.logging._errors import ConfigurationError, SciTeXError

        assert issubclass(ConfigurationError, SciTeXError)

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError."""
        from scitex.logging._errors import ConfigFileNotFoundError

        error = ConfigFileNotFoundError("/path/to/config.yaml")
        assert "/path/to/config.yaml" in str(error)
        assert "filepath" in error.context
        assert error.context["filepath"] == "/path/to/config.yaml"
        assert error.suggestion is not None

    def test_config_key_error(self):
        """Test ConfigKeyError."""
        from scitex.logging._errors import ConfigKeyError

        error = ConfigKeyError("missing_key")
        assert "missing_key" in str(error)
        assert error.context["missing_key"] == "missing_key"

    def test_config_key_error_with_available_keys(self):
        """Test ConfigKeyError with available keys list."""
        from scitex.logging._errors import ConfigKeyError

        error = ConfigKeyError("missing", available_keys=["key1", "key2"])
        assert "missing" in str(error)
        assert "available_keys" in error.context


class TestIOErrors:
    """Test IO-related errors."""

    def test_io_error_inheritance(self):
        """Test IOError inherits from SciTeXError."""
        from scitex.logging._errors import IOError, SciTeXError

        assert issubclass(IOError, SciTeXError)

    def test_file_format_error(self):
        """Test FileFormatError."""
        from scitex.logging._errors import FileFormatError

        error = FileFormatError("/path/file.txt")
        assert "/path/file.txt" in str(error)
        assert error.context["filepath"] == "/path/file.txt"

    def test_file_format_error_with_formats(self):
        """Test FileFormatError with expected and actual formats."""
        from scitex.logging._errors import FileFormatError

        error = FileFormatError(
            "/path/file.txt", expected_format="json", actual_format="csv"
        )
        assert "expected: json" in str(error)
        assert "got: csv" in str(error)

    def test_save_error(self):
        """Test SaveError."""
        from scitex.logging._errors import SaveError

        error = SaveError("/path/file.txt", "Permission denied")
        assert "/path/file.txt" in str(error)
        assert "Permission denied" in str(error)
        assert error.context["filepath"] == "/path/file.txt"
        assert error.context["reason"] == "Permission denied"

    def test_load_error(self):
        """Test LoadError."""
        from scitex.logging._errors import LoadError

        error = LoadError("/path/file.txt", "File not found")
        assert "/path/file.txt" in str(error)
        assert "File not found" in str(error)


class TestScholarErrors:
    """Test scholar module errors."""

    def test_scholar_error_inheritance(self):
        """Test ScholarError inherits from SciTeXError."""
        from scitex.logging._errors import ScholarError, SciTeXError

        assert issubclass(ScholarError, SciTeXError)

    def test_search_error(self):
        """Test SearchError."""
        from scitex.logging._errors import SearchError

        error = SearchError("machine learning", "PubMed", "API timeout")
        assert "machine learning" in str(error)
        assert "PubMed" in str(error)
        assert error.context["query"] == "machine learning"

    def test_enrichment_error(self):
        """Test EnrichmentError."""
        from scitex.logging._errors import EnrichmentError

        error = EnrichmentError("Paper Title", "Missing journal info")
        assert "Paper Title" in str(error)
        assert error.context["paper_title"] == "Paper Title"

    def test_pdf_download_error(self):
        """Test PDFDownloadError."""
        from scitex.logging._errors import PDFDownloadError

        error = PDFDownloadError("https://example.com/paper.pdf", "404 Not Found")
        assert "example.com" in str(error)
        assert error.context["url"] == "https://example.com/paper.pdf"

    def test_doi_resolution_error(self):
        """Test DOIResolutionError."""
        from scitex.logging._errors import DOIResolutionError

        error = DOIResolutionError("10.1234/example", "Invalid DOI format")
        assert "10.1234/example" in str(error)
        assert error.context["doi"] == "10.1234/example"

    def test_pdf_extraction_error(self):
        """Test PDFExtractionError."""
        from scitex.logging._errors import PDFExtractionError

        error = PDFExtractionError("/path/paper.pdf", "Encrypted PDF")
        assert "/path/paper.pdf" in str(error)

    def test_bibtex_enrichment_error(self):
        """Test BibTeXEnrichmentError."""
        from scitex.logging._errors import BibTeXEnrichmentError

        error = BibTeXEnrichmentError("/path/refs.bib", "Parse error")
        assert "/path/refs.bib" in str(error)

    def test_translator_error(self):
        """Test TranslatorError."""
        from scitex.logging._errors import TranslatorError

        error = TranslatorError("PubMedTranslator", "JavaScript error")
        assert "PubMedTranslator" in str(error)

    def test_authentication_error(self):
        """Test AuthenticationError."""
        from scitex.logging._errors import AuthenticationError

        error = AuthenticationError("API", "Invalid token")
        assert "API" in str(error)
        assert error.context["provider"] == "API"


class TestPlottingErrors:
    """Test plotting-related errors."""

    def test_plotting_error_inheritance(self):
        """Test PlottingError inherits from SciTeXError."""
        from scitex.logging._errors import PlottingError, SciTeXError

        assert issubclass(PlottingError, SciTeXError)

    def test_figure_not_found_error_with_int(self):
        """Test FigureNotFoundError with integer ID."""
        from scitex.logging._errors import FigureNotFoundError

        error = FigureNotFoundError(1)
        assert "1" in str(error)
        assert error.context["figure_id"] == 1

    def test_figure_not_found_error_with_string(self):
        """Test FigureNotFoundError with string ID."""
        from scitex.logging._errors import FigureNotFoundError

        error = FigureNotFoundError("my_figure")
        assert "my_figure" in str(error)

    def test_axis_error(self):
        """Test AxisError."""
        from scitex.logging._errors import AxisError

        error = AxisError("Invalid axis index")
        assert "Invalid axis index" in str(error)

    def test_axis_error_with_info(self):
        """Test AxisError with axis info."""
        from scitex.logging._errors import AxisError

        error = AxisError("Bad axis", axis_info={"row": 0, "col": 1})
        assert error.context is not None


class TestDataErrors:
    """Test data processing errors."""

    def test_data_error_inheritance(self):
        """Test DataError inherits from SciTeXError."""
        from scitex.logging._errors import DataError, SciTeXError

        assert issubclass(DataError, SciTeXError)

    def test_shape_error(self):
        """Test ShapeError."""
        from scitex.logging._errors import ShapeError

        error = ShapeError((10, 20), (10, 30), "matrix multiply")
        assert "matrix multiply" in str(error)
        assert error.context["expected_shape"] == (10, 20)
        assert error.context["actual_shape"] == (10, 30)

    def test_dtype_error(self):
        """Test DTypeError."""
        from scitex.logging._errors import DTypeError

        error = DTypeError("float32", "int64", "tensor operation")
        assert "tensor operation" in str(error)
        assert error.context["expected_dtype"] == "float32"
        assert error.context["actual_dtype"] == "int64"


class TestPathErrors:
    """Test path-related errors."""

    def test_path_error_inheritance(self):
        """Test PathError inherits from SciTeXError."""
        from scitex.logging._errors import PathError, SciTeXError

        assert issubclass(PathError, SciTeXError)

    def test_invalid_path_error(self):
        """Test InvalidPathError."""
        from scitex.logging._errors import InvalidPathError

        error = InvalidPathError("/absolute/path", "Must be relative")
        assert "/absolute/path" in str(error)
        assert error.context["path"] == "/absolute/path"

    def test_path_not_found_error(self):
        """Test PathNotFoundError."""
        from scitex.logging._errors import PathNotFoundError

        error = PathNotFoundError("./missing/file.txt")
        assert "./missing/file.txt" in str(error)


class TestTemplateErrors:
    """Test template-related errors."""

    def test_template_error_inheritance(self):
        """Test TemplateError inherits from SciTeXError."""
        from scitex.logging._errors import SciTeXError, TemplateError

        assert issubclass(TemplateError, SciTeXError)

    def test_template_violation_error(self):
        """Test TemplateViolationError."""
        from scitex.logging._errors import TemplateViolationError

        error = TemplateViolationError("script.py", "Missing header")
        assert "script.py" in str(error)
        assert "Missing header" in str(error)


class TestNNErrors:
    """Test neural network errors."""

    def test_nn_error_inheritance(self):
        """Test NNError inherits from SciTeXError."""
        from scitex.logging._errors import NNError, SciTeXError

        assert issubclass(NNError, SciTeXError)

    def test_model_error(self):
        """Test ModelError."""
        from scitex.logging._errors import ModelError

        error = ModelError("ResNet50", "Weight loading failed")
        assert "ResNet50" in str(error)
        assert error.context["model_name"] == "ResNet50"


class TestStatsErrors:
    """Test statistics errors."""

    def test_stats_error_inheritance(self):
        """Test StatsError inherits from SciTeXError."""
        from scitex.logging._errors import SciTeXError, StatsError

        assert issubclass(StatsError, SciTeXError)

    def test_test_error(self):
        """Test TestError."""
        from scitex.logging._errors import TestError

        error = TestError("t-test", "Sample size too small")
        assert "t-test" in str(error)
        assert error.context["test_name"] == "t-test"


class TestValidationHelpers:
    """Test validation helper functions."""

    def test_check_path_valid_relative(self):
        """Test check_path with valid relative paths."""
        from scitex.logging._errors import check_path

        # Should not raise
        check_path("./valid/path")
        check_path("../parent/path")

    def test_check_path_invalid_absolute(self):
        """Test check_path raises for absolute paths."""
        from scitex.logging._errors import InvalidPathError, check_path

        with pytest.raises(InvalidPathError):
            check_path("/absolute/path")

    def test_check_path_invalid_type(self):
        """Test check_path raises for non-string paths."""
        from scitex.logging._errors import InvalidPathError, check_path

        with pytest.raises(InvalidPathError):
            check_path(123)

    def test_check_file_exists_success(self):
        """Test check_file_exists with existing file."""
        from scitex.logging._errors import check_file_exists

        with tempfile.NamedTemporaryFile() as f:
            # Should not raise
            check_file_exists(f.name)

    def test_check_file_exists_failure(self):
        """Test check_file_exists with non-existing file."""
        from scitex.logging._errors import PathNotFoundError, check_file_exists

        with pytest.raises(PathNotFoundError):
            check_file_exists("/nonexistent/file.txt")

    def test_check_shape_compatibility_success(self):
        """Test check_shape_compatibility with matching shapes."""
        from scitex.logging._errors import check_shape_compatibility

        # Should not raise
        check_shape_compatibility((10, 20), (10, 20), "test op")

    def test_check_shape_compatibility_failure(self):
        """Test check_shape_compatibility with mismatched shapes."""
        from scitex.logging._errors import ShapeError, check_shape_compatibility

        with pytest.raises(ShapeError):
            check_shape_compatibility((10, 20), (10, 30), "test op")

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_errors.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-21"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_errors.py
# 
# """Error classes for SciTeX.
# 
# All SciTeX exceptions are defined here for unified error handling.
# Re-exported from scitex.logging for backwards compatibility.
# 
# Usage:
#     from scitex.logging import SciTeXError, SaveError, LoadError
#     # or (backwards compatible)
#     from scitex.logging import SciTeXError, SaveError, LoadError
# """
# 
# from typing import Optional, Union
# 
# 
# # =============================================================================
# # Base Errors
# # =============================================================================
# 
# 
# class SciTeXError(Exception):
#     """Base Exception class for all SciTeX errors."""
# 
#     def __init__(
#         self,
#         message: str,
#         context: Optional[dict] = None,
#         suggestion: Optional[str] = None,
#     ):
#         """Initialize SciTeX error with detailed information.
# 
#         Parameters
#         ----------
#         message : str
#             The error message
#         context : dict, optional
#             Additional context information (e.g., file paths, variable values)
#         suggestion : str, optional
#             Suggested fix or action
#         """
#         self.message = message
#         self.context = context or {}
#         self.suggestion = suggestion
# 
#         # Build the full error message
#         error_parts = [f"SciTeX Error: {message}"]
# 
#         if context:
#             error_parts.append("\nContext:")
#             for key, value in context.items():
#                 error_parts.append(f"  {key}: {value}")
# 
#         if suggestion:
#             error_parts.append(f"\nSuggestion: {suggestion}")
# 
#         super().__init__("\n".join(error_parts))
# 
# 
# # =============================================================================
# # Configuration Errors
# # =============================================================================
# 
# 
# class ConfigurationError(SciTeXError):
#     """Raised when there are issues with SciTeX configuration."""
# 
#     pass
# 
# 
# class ConfigFileNotFoundError(ConfigurationError):
#     """Raised when a required configuration file is not found."""
# 
#     def __init__(self, filepath: str):
#         super().__init__(
#             f"Configuration file not found: {filepath}",
#             context={"filepath": filepath},
#             suggestion="Ensure the configuration file exists in ./config/ directory",
#         )
# 
# 
# class ConfigKeyError(ConfigurationError):
#     """Raised when a required configuration key is missing."""
# 
#     def __init__(self, key: str, available_keys: Optional[list] = None):
#         context = {"missing_key": key}
#         if available_keys:
#             context["available_keys"] = available_keys
# 
#         super().__init__(
#             f"Configuration key '{key}' not found",
#             context=context,
#             suggestion=f"Add '{key}' to your configuration file or check for typos",
#         )
# 
# 
# # =============================================================================
# # IO Errors
# # =============================================================================
# 
# 
# class IOError(SciTeXError):
#     """Base class for input/output related errors."""
# 
#     pass
# 
# 
# class FileFormatError(IOError):
#     """Raised when file format is not supported or incorrect."""
# 
#     def __init__(
#         self,
#         filepath: str,
#         expected_format: Optional[str] = None,
#         actual_format: Optional[str] = None,
#     ):
#         context = {"filepath": filepath}
#         if expected_format:
#             context["expected_format"] = expected_format
#         if actual_format:
#             context["actual_format"] = actual_format
# 
#         message = f"File format error for: {filepath}"
#         if expected_format and actual_format:
#             message += f" (expected: {expected_format}, got: {actual_format})"
# 
#         super().__init__(
#             message,
#             context=context,
#             suggestion="Check the file extension and content format",
#         )
# 
# 
# class SaveError(IOError):
#     """Raised when saving data fails."""
# 
#     def __init__(self, filepath: str, reason: str):
#         super().__init__(
#             f"Failed to save to {filepath}: {reason}",
#             context={"filepath": filepath, "reason": reason},
#             suggestion="Check file permissions and disk space",
#         )
# 
# 
# class LoadError(IOError):
#     """Raised when loading data fails."""
# 
#     def __init__(self, filepath: str, reason: str):
#         super().__init__(
#             f"Failed to load from {filepath}: {reason}",
#             context={"filepath": filepath, "reason": reason},
#             suggestion="Verify the file exists and is not corrupted",
#         )
# 
# 
# # =============================================================================
# # Scholar Module Errors
# # =============================================================================
# 
# 
# class ScholarError(SciTeXError):
#     """Base class for scholar module errors."""
# 
#     pass
# 
# 
# class SearchError(ScholarError):
#     """Raised when paper search fails."""
# 
#     def __init__(self, query: str, source: str, reason: str):
#         super().__init__(
#             f"Search failed for query '{query}' on {source}",
#             context={"query": query, "source": source, "reason": reason},
#             suggestion="Check your internet connection and API keys",
#         )
# 
# 
# class EnrichmentError(ScholarError):
#     """Raised when paper enrichment fails."""
# 
#     def __init__(self, paper_title: str, reason: str):
#         super().__init__(
#             f"Failed to enrich paper: {paper_title}",
#             context={"paper_title": paper_title, "reason": reason},
#             suggestion="Verify journal information is available",
#         )
# 
# 
# class PDFDownloadError(ScholarError):
#     """Raised when PDF download fails."""
# 
#     def __init__(self, url: str, reason: str):
#         super().__init__(
#             f"Failed to download PDF from {url}",
#             context={"url": url, "reason": reason},
#             suggestion="Check if the paper is open access",
#         )
# 
# 
# class DOIResolutionError(ScholarError):
#     """Raised when DOI resolution fails."""
# 
#     def __init__(self, doi: str, reason: str):
#         super().__init__(
#             f"Failed to resolve DOI: {doi}",
#             context={"doi": doi, "reason": reason},
#             suggestion="Verify the DOI is correct and try again",
#         )
# 
# 
# class PDFExtractionError(ScholarError):
#     """Raised when PDF text extraction fails."""
# 
#     def __init__(self, filepath: str, reason: str):
#         super().__init__(
#             f"Failed to extract text from PDF: {filepath}",
#             context={"filepath": filepath, "reason": reason},
#             suggestion="Ensure the PDF is not corrupted or encrypted",
#         )
# 
# 
# class BibTeXEnrichmentError(ScholarError):
#     """Raised when BibTeX enrichment fails."""
# 
#     def __init__(self, bibtex_file: str, reason: str):
#         super().__init__(
#             f"Failed to enrich BibTeX file: {bibtex_file}",
#             context={"bibtex_file": bibtex_file, "reason": reason},
#             suggestion="Check the BibTeX format and ensure all entries are valid",
#         )
# 
# 
# class TranslatorError(ScholarError):
#     """Raised when Zotero translator operations fail."""
# 
#     def __init__(self, translator_name: str, reason: str):
#         super().__init__(
#             f"Translator error in {translator_name}: {reason}",
#             context={"translator": translator_name, "reason": reason},
#             suggestion="Check translator compatibility and JavaScript environment",
#         )
# 
# 
# class AuthenticationError(ScholarError):
#     """Raised when authentication fails."""
# 
#     def __init__(self, provider: str, reason: str = ""):
#         super().__init__(
#             f"Authentication failed for {provider}: {reason}",
#             context={"provider": provider, "reason": reason},
#             suggestion="Check your credentials and authentication settings",
#         )
# 
# 
# # =============================================================================
# # Plotting Errors
# # =============================================================================
# 
# 
# class PlottingError(SciTeXError):
#     """Base class for plotting-related errors."""
# 
#     pass
# 
# 
# class FigureNotFoundError(PlottingError):
#     """Raised when attempting to operate on a non-existent figure."""
# 
#     def __init__(self, fig_id: Union[int, str]):
#         super().__init__(
#             f"Figure {fig_id} not found",
#             context={"figure_id": fig_id},
#             suggestion="Ensure the figure was created before attempting to save/modify it",
#         )
# 
# 
# class AxisError(PlottingError):
#     """Raised when there are issues with plot axes."""
# 
#     def __init__(self, message: str, axis_info: Optional[dict] = None):
#         super().__init__(
#             message,
#             context={"axis_info": axis_info} if axis_info else None,
#             suggestion="Check axis indices and subplot configuration",
#         )
# 
# 
# # =============================================================================
# # Data Processing Errors
# # =============================================================================
# 
# 
# class DataError(SciTeXError):
#     """Base class for data processing errors."""
# 
#     pass
# 
# 
# class ShapeError(DataError):
#     """Raised when data shapes are incompatible."""
# 
#     def __init__(self, expected_shape: tuple, actual_shape: tuple, operation: str):
#         super().__init__(
#             f"Shape mismatch in {operation}",
#             context={
#                 "expected_shape": expected_shape,
#                 "actual_shape": actual_shape,
#                 "operation": operation,
#             },
#             suggestion="Reshape or transpose your data to match expected dimensions",
#         )
# 
# 
# class DTypeError(DataError):
#     """Raised when data types are incompatible."""
# 
#     def __init__(self, expected_dtype: str, actual_dtype: str, operation: str):
#         super().__init__(
#             f"Data type mismatch in {operation}",
#             context={
#                 "expected_dtype": expected_dtype,
#                 "actual_dtype": actual_dtype,
#                 "operation": operation,
#             },
#             suggestion=f"Convert data to {expected_dtype} using appropriate casting",
#         )
# 
# 
# # =============================================================================
# # Path Errors
# # =============================================================================
# 
# 
# class PathError(SciTeXError):
#     """Base class for path-related errors."""
# 
#     pass
# 
# 
# class InvalidPathError(PathError):
#     """Raised when a path is invalid or doesn't follow SciTeX conventions."""
# 
#     def __init__(self, path: str, reason: str):
#         super().__init__(
#             f"Invalid path: {path}",
#             context={"path": path, "reason": reason},
#             suggestion="Use relative paths starting with './' or '../'",
#         )
# 
# 
# class PathNotFoundError(PathError):
#     """Raised when a required path doesn't exist."""
# 
#     def __init__(self, path: str):
#         super().__init__(
#             f"Path not found: {path}",
#             context={"path": path},
#             suggestion="Check if the path exists and is accessible",
#         )
# 
# 
# # =============================================================================
# # Template Errors
# # =============================================================================
# 
# 
# class TemplateError(SciTeXError):
#     """Base class for template-related errors."""
# 
#     pass
# 
# 
# class TemplateViolationError(TemplateError):
#     """Raised when SciTeX template is not followed."""
# 
#     def __init__(self, filepath: str, violation: str):
#         super().__init__(
#             f"Template violation in {filepath}: {violation}",
#             context={"filepath": filepath, "violation": violation},
#             suggestion="Follow the SciTeX template structure as defined in IMPORTANT-SCITEX-02-file-template.md",
#         )
# 
# 
# # =============================================================================
# # Neural Network Errors
# # =============================================================================
# 
# 
# class NNError(SciTeXError):
#     """Base class for neural network module errors."""
# 
#     pass
# 
# 
# class ModelError(NNError):
#     """Raised when there are issues with neural network models."""
# 
#     def __init__(self, model_name: str, reason: str):
#         super().__init__(
#             f"Model error in {model_name}: {reason}",
#             context={"model_name": model_name, "reason": reason},
#         )
# 
# 
# # =============================================================================
# # Statistics Errors
# # =============================================================================
# 
# 
# class StatsError(SciTeXError):
#     """Base class for statistics module errors."""
# 
#     pass
# 
# 
# class TestError(StatsError):
#     """Raised when statistical tests fail."""
# 
#     def __init__(self, test_name: str, reason: str):
#         super().__init__(
#             f"Statistical test '{test_name}' failed: {reason}",
#             context={"test_name": test_name, "reason": reason},
#         )
# 
# 
# # =============================================================================
# # Validation Helpers
# # =============================================================================
# 
# 
# def check_path(path: str) -> None:
#     """Validate a path according to SciTeX conventions."""
#     if not isinstance(path, str):
#         raise InvalidPathError(str(path), "Path must be a string")
# 
#     if not (path.startswith("./") or path.startswith("../")):
#         raise InvalidPathError(
#             path, "Path must be relative and start with './' or '../'"
#         )
# 
# 
# def check_file_exists(filepath: str) -> None:
#     """Check if a file exists."""
#     import os
# 
#     if not os.path.exists(filepath):
#         raise PathNotFoundError(filepath)
# 
# 
# def check_shape_compatibility(shape1: tuple, shape2: tuple, operation: str) -> None:
#     """Check if two shapes are compatible for an operation."""
#     if shape1 != shape2:
#         raise ShapeError(shape1, shape2, operation)
# 
# 
# __all__ = [
#     # Base errors
#     "SciTeXError",
#     # Configuration
#     "ConfigurationError",
#     "ConfigFileNotFoundError",
#     "ConfigKeyError",
#     # IO
#     "IOError",
#     "FileFormatError",
#     "SaveError",
#     "LoadError",
#     # Scholar
#     "ScholarError",
#     "SearchError",
#     "EnrichmentError",
#     "PDFDownloadError",
#     "DOIResolutionError",
#     "PDFExtractionError",
#     "BibTeXEnrichmentError",
#     "TranslatorError",
#     "AuthenticationError",
#     # Plotting
#     "PlottingError",
#     "FigureNotFoundError",
#     "AxisError",
#     # Data
#     "DataError",
#     "ShapeError",
#     "DTypeError",
#     # Path
#     "PathError",
#     "InvalidPathError",
#     "PathNotFoundError",
#     # Template
#     "TemplateError",
#     "TemplateViolationError",
#     # Neural Network
#     "NNError",
#     "ModelError",
#     # Statistics
#     "StatsError",
#     "TestError",
#     # Validation helpers
#     "check_path",
#     "check_file_exists",
#     "check_shape_compatibility",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_errors.py
# --------------------------------------------------------------------------------
