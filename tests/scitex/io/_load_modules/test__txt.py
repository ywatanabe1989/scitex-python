#!/usr/bin/env python3
# Time-stamp: "2025-06-02 14:30:00 (ywatanabe)"
# File: ./scitex_repo/tests/scitex/io/_load_modules/test__txt.py

"""Tests for text file loading functionality.

This module tests the _load_txt function from scitex.io._load_modules._txt,
which handles loading text files with various encodings and formats.
"""

import os
import tempfile
import warnings

import pytest

# Required for scitex.io module
pytest.importorskip("h5py")
pytest.importorskip("zarr")


def test_load_txt_basic():
    """Test loading a basic text file."""
    from scitex.io._load_modules._txt import _load_txt

    content = "Hello World\nThis is a test\nThird line"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        # Default behavior: as_lines=True, strip=True - returns list of stripped non-empty lines
        loaded_lines = _load_txt(temp_path)
        assert loaded_lines == ["Hello World", "This is a test", "Third line"]

        # as_lines=False returns full content
        loaded_content = _load_txt(temp_path, as_lines=False)
        assert loaded_content == content

        # as_lines=False, strip=True returns stripped content
        loaded_stripped = _load_txt(temp_path, as_lines=False, strip=True)
        assert loaded_stripped == content.strip()
    finally:
        os.unlink(temp_path)


def test_load_txt_empty_lines():
    """Test handling of empty lines."""
    from scitex.io._load_modules._txt import _load_txt

    content = "Line 1\n\n\nLine 2\n   \nLine 3\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        # Default: as_lines=True, strip=True - filters out empty lines
        loaded_lines = _load_txt(temp_path)
        assert loaded_lines == ["Line 1", "Line 2", "Line 3"]

        # as_lines=False preserves everything
        loaded_content = _load_txt(temp_path, as_lines=False, strip=False)
        assert loaded_content == content
    finally:
        os.unlink(temp_path)


def test_load_txt_different_extensions():
    """Test loading files with different extensions."""
    from scitex.io._load_modules._txt import _load_txt

    content = "Test content"
    extensions = [".txt", ".log", ".event", ".py", ".sh"]

    for ext in extensions:
        with tempfile.NamedTemporaryFile(mode="w", suffix=ext, delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            # Default returns list of lines
            loaded = _load_txt(temp_path)
            assert loaded == [content]

            # as_lines=False returns string
            loaded_str = _load_txt(temp_path, as_lines=False)
            assert loaded_str == content
        finally:
            os.unlink(temp_path)


def test_load_txt_unexpected_extension_warning(caplog):
    """Test warning for unexpected file extensions."""
    from scitex.io._load_modules._txt import _load_txt

    content = "Test content"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        # Should still load the file
        loaded = _load_txt(temp_path, as_lines=False)
        assert loaded == content

        # Should have logged warning about unexpected extension
        assert "Unexpected extension" in caplog.text
    finally:
        os.unlink(temp_path)


def test_load_txt_unicode_content():
    """Test loading files with Unicode content."""
    from scitex.io._load_modules._txt import _load_txt

    unicode_content = "Hello 世界\n日本語テスト\n🎉 Emoji test"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(unicode_content)
        temp_path = f.name

    try:
        # as_lines=False returns full content
        loaded = _load_txt(temp_path, as_lines=False)
        assert loaded == unicode_content

        # Default: as_lines=True returns list
        loaded_lines = _load_txt(temp_path)
        assert loaded_lines == ["Hello 世界", "日本語テスト", "🎉 Emoji test"]
    finally:
        os.unlink(temp_path)


def test_load_txt_different_encodings():
    """Test loading files with different encodings."""
    from scitex.io._load_modules._txt import _load_txt

    # Test Latin-1 encoded content
    latin1_content = "Café résumé naïve"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="latin1"
    ) as f:
        f.write(latin1_content)
        temp_path = f.name

    try:
        # Should handle encoding fallback
        loaded = _load_txt(temp_path, as_lines=False)
        assert loaded == latin1_content
    finally:
        os.unlink(temp_path)


def test_load_txt_with_strip():
    """Test strip functionality."""
    from scitex.io._load_modules._txt import _load_txt

    content_with_whitespace = "  \n  Hello World  \n  \n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content_with_whitespace)
        temp_path = f.name

    try:
        # as_lines=False, strip=False preserves whitespace
        loaded = _load_txt(temp_path, as_lines=False, strip=False)
        assert loaded == content_with_whitespace

        # as_lines=False, strip=True strips content
        loaded_stripped = _load_txt(temp_path, as_lines=False, strip=True)
        assert loaded_stripped == "Hello World"

        # Default: as_lines=True, strip=True returns stripped non-empty lines
        loaded_lines = _load_txt(temp_path)
        assert loaded_lines == ["Hello World"]
    finally:
        os.unlink(temp_path)


def test_load_txt_nonexistent_file():
    """Test loading a nonexistent file."""
    from scitex.io._load_modules._txt import _load_txt

    # The current implementation raises FileNotFoundError directly
    with pytest.raises(FileNotFoundError):
        _load_txt("/nonexistent/path/file.txt")


def test_load_txt_empty_file():
    """Test loading an empty file."""
    from scitex.io._load_modules._txt import _load_txt

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        temp_path = f.name

    try:
        # Default: as_lines=True returns empty list
        loaded_lines = _load_txt(temp_path)
        assert loaded_lines == []

        # as_lines=False returns empty string
        loaded = _load_txt(temp_path, as_lines=False)
        assert loaded == ""
    finally:
        os.unlink(temp_path)


def test_load_txt_large_file():
    """Test loading a large text file."""
    from scitex.io._load_modules._txt import _load_txt

    # Create a file with many lines
    lines = [f"Line {i}: " + "x" * 100 for i in range(1000)]
    content = "\n".join(lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        temp_path = f.name

    try:
        # as_lines=False returns full content
        loaded = _load_txt(temp_path, as_lines=False)
        assert loaded == content

        # Default: as_lines=True returns list of lines
        loaded_lines = _load_txt(temp_path)
        assert len(loaded_lines) == 1000
        assert loaded_lines[0] == lines[0]
        assert loaded_lines[-1] == lines[-1]
    finally:
        os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load_modules/_txt.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-07-15 09:27:36 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/io/_load_modules/_txt.py
# # ----------------------------------------
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# from scitex import logging
# 
# logger = logging.getLogger(__name__)
# 
# # # UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8a in position 30173: invalid start byte
# # def _load_txt(lpath, **kwargs):
# #     """Load text file and return non-empty lines."""
# #     SUPPORTED_EXTENSIONS = (".txt", ".log", ".event", ".py", ".sh", "")
# #     try:
# #         if not lpath.endswith(SUPPORTED_EXTENSIONS):
# #             warnings.warn(
# #                 f"File must have supported extensions: {SUPPORTED_EXTENSIONS}"
# #             )
# 
# #         # Try UTF-8 first (most common)
# #         try:
# #             with open(lpath, "r", encoding="utf-8") as f:
# #                 return [
# #                     line.strip()
# #                     for line in f.read().splitlines()
# #                     if line.strip()
# #                 ]
# #         except UnicodeDecodeError:
# #             # Fallback to system default encoding
# #             with open(lpath, "r") as f:
# #                 return [
# #                     line.strip()
# #                     for line in f.read().splitlines()
# #                     if line.strip()
# #                 ]
# 
# 
# #     except (ValueError, FileNotFoundError) as e:
# #         raise ValueError(f"Error loading file {lpath}: {str(e)}")
# # Removed duplicate function - see main _load_txt below
# 
# 
# # def _load_txt(lpath, strip=False, as_lines=False):
# #     """
# #     Load text file and return its content.
# #     - Warn if extension is unexpected.
# #     - Try UTF-8 first, then default encoding.
# #     - If strip=True, strip whitespace.
# #     - If as_lines=True, return list of lines (backward compatibility).
# #     """
# #     if not lpath.endswith((".txt", ".log", ".event", ".py", ".sh")):
# #         warnings.warn(f"Unexpected extension for file: {lpath}")
# 
# #     try:
# #         with open(lpath, "r", encoding="utf-8") as file:
# #             content = file.read()
# #     except UnicodeDecodeError:
# #         # Fallback: try to detect correct encoding
# #         encoding = _check_encoding(lpath)
# #         with open(lpath, "r", encoding=encoding) as file:
# #             content = file.read()
# 
# #     # For backward compatibility, check if as_lines parameter or legacy behavior needed
# #     if as_lines:
# #         raw_lines = content.splitlines()
# #         if strip:
# #             return [line.strip() for line in raw_lines if line.strip()]
# #         return [line for line in raw_lines if line.strip()]
# 
# #     # Default: return full content (possibly stripped)
# #     if strip:
# #         return content.strip()
# 
# #     return content
# 
# 
# def _load_txt(lpath, strip=True, as_lines=True):
#     """
#     Load text file and return its content.
#     - Warn if extension is unexpected.
#     - Try UTF-8 first, then default encoding.
#     - If strip=True, strip whitespace from each line.
#     - If as_lines=True, return list of lines.
#     """
#     # Convert Path object to string if needed
#     from pathlib import Path
# 
#     if isinstance(lpath, Path):
#         lpath = str(lpath)
# 
#     if not lpath.endswith((".txt", ".log", ".event", ".py", ".sh", ".tex", ".bib")):
#         logger.warning(f"Unexpected extension for file: {lpath}")
# 
#     try:
#         with open(lpath, "r", encoding="utf-8") as file:
#             content = file.read()
#     except UnicodeDecodeError:
#         encoding = _check_encoding(lpath)
#         with open(lpath, "r", encoding=encoding) as file:
#             content = file.read()
# 
#     if as_lines:
#         raw_lines = content.splitlines()
#         if strip:
#             return [line.strip() for line in raw_lines if line.strip()]
#         return [line for line in raw_lines if line]
# 
#     if strip:
#         return content.strip()
#     return content
# 
# 
# def _check_encoding(file_path):
#     """Check file encoding by trying common encodings."""
#     encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1", "ascii"]
# 
#     for encoding in encodings:
#         try:
#             with open(file_path, "r", encoding=encoding) as f:
#                 f.read()
#             return encoding
#         except UnicodeDecodeError:
#             continue
# 
#     raise ValueError(f"Unable to determine encoding for {file_path}")
# 
# 
# # def _check_encoding(file_path):
# #     """
# #     Check the encoding of a given file.
# 
# #     This function attempts to read the file with different encodings
# #     to determine the correct one.
# 
# #     Parameters:
# #     -----------
# #     file_path : str
# #         The path to the file to check.
# 
# #     Returns:
# #     --------
# #     str
# #         The detected encoding of the file.
# 
# #     Raises:
# #     -------
# #     IOError
# #         If the file cannot be read or the encoding cannot be determined.
# #     """
# #     import chardet
# 
# #     with open(file_path, "rb") as file:
# #         raw_data = file.read()
# 
# #     result = chardet.detect(raw_data)
# #     return result["encoding"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_load_modules/_txt.py
# --------------------------------------------------------------------------------
