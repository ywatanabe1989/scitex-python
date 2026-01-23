#!/usr/bin/env python3
"""Tests for scitex.logging._context module."""

import logging
import os
import tempfile
from pathlib import Path

import pytest


class TestLogToFile:
    """Test log_to_file context manager."""

    def test_log_to_file_creates_file(self):
        """Test that log_to_file creates the log file."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"

            with log_to_file(log_path):
                pass

            assert log_path.exists()

    def test_log_to_file_creates_parent_directories(self):
        """Test that log_to_file creates parent directories if needed."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nested" / "dirs" / "test.log"

            with log_to_file(log_path):
                pass

            assert log_path.exists()
            assert log_path.parent.exists()

    def test_log_to_file_writes_log_messages(self):
        """Test that log messages are written to the file."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = logging.getLogger("test_context")
            logger.setLevel(logging.DEBUG)

            with log_to_file(log_path, level=logging.DEBUG):
                logger.info("Test message")

            content = log_path.read_text()
            assert "Test message" in content

    def test_log_to_file_respects_level(self):
        """Test that log_to_file respects the specified level."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = logging.getLogger("test_context_level")
            logger.setLevel(logging.DEBUG)

            with log_to_file(log_path, level=logging.WARNING):
                logger.debug("Debug message")
                logger.warning("Warning message")

            content = log_path.read_text()
            assert "Debug message" not in content
            assert "Warning message" in content

    def test_log_to_file_append_mode(self):
        """Test that log_to_file with mode='a' appends to existing file."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = logging.getLogger("test_context_append")
            logger.setLevel(logging.DEBUG)

            # Write first message
            with log_to_file(log_path, mode="w"):
                logger.info("First message")

            # Append second message
            with log_to_file(log_path, mode="a"):
                logger.info("Second message")

            content = log_path.read_text()
            assert "First message" in content
            assert "Second message" in content

    def test_log_to_file_overwrite_mode(self):
        """Test that log_to_file with mode='w' overwrites existing file."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = logging.getLogger("test_context_overwrite")
            logger.setLevel(logging.DEBUG)

            # Write first message
            with log_to_file(log_path, mode="w"):
                logger.info("First message")

            # Overwrite with second message
            with log_to_file(log_path, mode="w"):
                logger.info("Second message")

            content = log_path.read_text()
            assert "First message" not in content
            assert "Second message" in content

    def test_log_to_file_removes_handler_after_context(self):
        """Test that handler is removed after context manager exits."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            root_logger = logging.getLogger()

            handlers_before = len(root_logger.handlers)

            with log_to_file(log_path):
                handlers_during = len(root_logger.handlers)
                assert handlers_during == handlers_before + 1

            handlers_after = len(root_logger.handlers)
            assert handlers_after == handlers_before

    def test_log_to_file_yields_handler(self):
        """Test that log_to_file yields the file handler."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"

            with log_to_file(log_path) as handler:
                assert isinstance(handler, logging.FileHandler)

    def test_log_to_file_accepts_string_path(self):
        """Test that log_to_file accepts string path."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")

            with log_to_file(log_path):
                pass

            assert os.path.exists(log_path)

    def test_log_to_file_accepts_path_object(self):
        """Test that log_to_file accepts Path object."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"

            with log_to_file(log_path):
                pass

            assert log_path.exists()

    def test_log_to_file_uses_scitex_formatter_by_default(self):
        """Test that log_to_file uses SciTeXFileFormatter by default."""
        from scitex.logging._context import log_to_file
        from scitex.logging._formatters import SciTeXFileFormatter

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"

            with log_to_file(log_path) as handler:
                assert isinstance(handler.formatter, SciTeXFileFormatter)

    def test_log_to_file_accepts_custom_formatter(self):
        """Test that log_to_file accepts custom formatter."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            custom_formatter = logging.Formatter("CUSTOM: %(message)s")

            with log_to_file(log_path, formatter=custom_formatter) as handler:
                assert handler.formatter == custom_formatter

    def test_log_to_file_cleanup_on_exception(self):
        """Test that handler is cleaned up even on exception."""
        from scitex.logging._context import log_to_file

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            root_logger = logging.getLogger()
            handlers_before = len(root_logger.handlers)

            with pytest.raises(ValueError):
                with log_to_file(log_path):
                    raise ValueError("Test exception")

            handlers_after = len(root_logger.handlers)
            assert handlers_after == handlers_before

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_context.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-10-11 22:30:00 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/logging/_context.py
# # ----------------------------------------
# from __future__ import annotations
# 
# import os
# 
# __FILE__ = "./src/scitex/logging/_context.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# Context manager for temporary logging to specific files.
# 
# Provides clean API for adding/removing file handlers during execution.
# """
# 
# import logging as _logging
# from contextlib import contextmanager
# from pathlib import Path
# from typing import Optional, Union
# 
# from ._formatters import SciTeXFileFormatter
# 
# 
# @contextmanager
# def log_to_file(
#     file_path: Union[str, Path],
#     level: int = _logging.DEBUG,
#     mode: str = "w",
#     formatter: Optional[_logging.Formatter] = None,
# ):
#     """Context manager to temporarily log all output to a specific file.
# 
#     Usage:
#         from scitex import logging
#         logger = logging.getLogger(__name__)
# 
#         with logging.log_to_file("/path/to/log.txt"):
#             logger.info("This goes to both console and /path/to/log.txt")
#             logger.success("This too!")
# 
#     Args:
#         file_path: Path to log file
#         level: Logging level for this handler (default: DEBUG)
#         mode: File mode ('w' for overwrite, 'a' for append)
#         formatter: Custom formatter (default: SciTeXFileFormatter)
# 
#     Yields
#     ------
#         The file handler (can be ignored)
#     """
#     # Ensure directory exists
#     file_path = Path(file_path)
#     file_path.parent.mkdir(parents=True, exist_ok=True)
# 
#     # Create handler
#     handler = _logging.FileHandler(str(file_path), mode=mode)
#     handler.setLevel(level)
# 
#     # Set formatter
#     if formatter is None:
#         formatter = SciTeXFileFormatter()
#     handler.setFormatter(formatter)
# 
#     # Add to root logger
#     root_logger = _logging.getLogger()
#     root_logger.addHandler(handler)
# 
#     # Log where output is going (lazy import to avoid circular dependency)
#     def _log_info():
#         try:
#             from scitex import logging
# 
#             logger = logging.getLogger(__name__)
#             logger.debug(f"Logging to: {file_path}")
#         except:
#             pass  # Silently fail if logging not ready
# 
#     _log_info()
# 
#     try:
#         yield handler
#     finally:
#         # Clean up handler
#         root_logger.removeHandler(handler)
#         handler.close()
# 
#         # Log completion (lazy import)
#         def _log_saved():
#             try:
#                 from scitex import logging
# 
#                 logger = logging.getLogger(__name__)
#                 logger.debug(f"Log saved: {file_path}")
#             except:
#                 pass
# 
#         _log_saved()
# 
# 
# __all__ = ["log_to_file"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_context.py
# --------------------------------------------------------------------------------
