#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for SciTeXLogger class."""

import logging
import tempfile
from pathlib import Path
import pytest

from scitex.logging._logger import SciTeXLogger, setup_logger_class
from scitex.logging._levels import SUCCESS, FAIL, DEBUG, INFO


class TestSciTeXLogger:
    """Test SciTeXLogger class functionality."""

    @pytest.fixture(autouse=True)
    def setup_logger(self):
        """Setup a fresh logger for each test."""
        # Setup the logger class
        setup_logger_class()

        # Create a test logger
        logger = logging.getLogger(f"test_logger_{id(self)}")
        logger.handlers.clear()
        logger.setLevel(DEBUG)

        # Add a handler to capture output
        self.log_records = []

        class ListHandler(logging.Handler):
            def __init__(self, records_list):
                super().__init__()
                self.records = records_list

            def emit(self, record):
                self.records.append(record)

        handler = ListHandler(self.log_records)
        handler.setLevel(DEBUG)
        logger.addHandler(handler)

        self.logger = logger
        yield

        # Cleanup
        logger.handlers.clear()

    def test_logger_is_scitex_logger(self):
        """Test that logger is an instance of SciTeXLogger."""
        assert isinstance(self.logger, SciTeXLogger)

    def test_debug_method(self):
        """Test debug logging method."""
        self.logger.debug("Debug message")
        assert len(self.log_records) == 1
        assert self.log_records[0].levelno == DEBUG
        assert self.log_records[0].getMessage() == "Debug message"

    def test_info_method(self):
        """Test info logging method."""
        self.logger.info("Info message")
        assert len(self.log_records) == 1
        assert self.log_records[0].levelno == INFO
        assert self.log_records[0].getMessage() == "Info message"

    def test_warning_method(self):
        """Test warning logging method."""
        self.logger.warning("Warning message")
        assert len(self.log_records) == 1
        assert self.log_records[0].levelno == logging.WARNING

    def test_error_method(self):
        """Test error logging method."""
        self.logger.error("Error message")
        assert len(self.log_records) == 1
        assert self.log_records[0].levelno == logging.ERROR

    def test_critical_method(self):
        """Test critical logging method."""
        self.logger.critical("Critical message")
        assert len(self.log_records) == 1
        assert self.log_records[0].levelno == logging.CRITICAL

    def test_success_method(self):
        """Test success logging method."""
        self.logger.success("Success message")
        assert len(self.log_records) == 1
        assert self.log_records[0].levelno == SUCCESS
        assert self.log_records[0].getMessage() == "Success message"

    def test_fail_method(self):
        """Test fail logging method."""
        self.logger.fail("Fail message")
        assert len(self.log_records) == 1
        assert self.log_records[0].levelno == FAIL
        assert self.log_records[0].getMessage() == "Fail message"

    def test_indent_parameter(self):
        """Test logging with indent parameter."""
        self.logger.info("Indented message", indent=2)
        assert len(self.log_records) == 1
        record = self.log_records[0]
        assert hasattr(record, 'indent')
        assert record.indent == 2

    def test_separator_parameter(self):
        """Test logging with separator parameter."""
        self.logger.info("Message with separator", sep="=", n_sep=20)
        assert len(self.log_records) == 1
        record = self.log_records[0]
        message = record.getMessage()
        assert "=" * 20 in message
        assert "Message with separator" in message

    def test_color_parameter(self):
        """Test logging with color parameter."""
        self.logger.info("Colored message", c="red")
        assert len(self.log_records) == 1
        record = self.log_records[0]
        assert hasattr(record, 'color')
        assert record.color == "red"

    def test_to_context_manager(self):
        """Test the to() context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"

            with self.logger.to(str(log_file)):
                self.logger.info("Message to file")

            # Check file was created and contains the message
            assert log_file.exists()
            content = log_file.read_text()
            assert "Message to file" in content


class TestSetupLoggerClass:
    """Test setup_logger_class function."""

    def test_setup_logger_class_sets_custom_class(self):
        """Test that setup_logger_class sets SciTeXLogger as the logger class."""
        # Reset to default
        logging.setLoggerClass(logging.Logger)

        # Setup custom class
        setup_logger_class()

        # Create a new logger
        logger = logging.getLogger("test_setup_logger")

        # Verify it's a SciTeXLogger
        assert isinstance(logger, SciTeXLogger)

        # Cleanup
        logger.handlers.clear()

    def test_setup_logger_class_updates_root_logger(self):
        """Test that setup_logger_class updates the root logger."""
        setup_logger_class()

        root_logger = logging.getLogger()
        assert root_logger.__class__ == SciTeXLogger

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_logger.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-17 15:03:33 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/logging/_logger.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/logging/_logger.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# __FILE__ = __file__
# 
# """Enhanced logger class for SciTeX."""
# 
# import logging
# import pprint as _pprint
# 
# from ._levels import FAIL, SUCCESS
# 
# 
# class SciTeXLogger(logging.Logger):
#     """Enhanced logger with success/fail methods, indent, separator, and color support."""
# 
#     def _log_with_indent(
#         self,
#         level,
#         message,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         *args,
#         **kwargs,
#     ):
#         """Internal method to log with indent, separator, color, and pprint support.
# 
#         Args:
#             level: Logging level
#             message: Message to log
#             indent: Number of spaces to indent
#             sep: Separator character (e.g., '=', '-')
#             n_sep: Number of separator characters
#             c: Color code
#             pprint: If True, format message using pprint for better readability
#             *args: Additional arguments
#             **kwargs: Additional keyword arguments
#         """
#         # Apply pprint formatting if requested
#         if pprint:
#             # If message is a string, keep it as is
#             # Otherwise, format it with pprint
#             if not isinstance(message, str):
#                 # Convert DotDict to regular dict for better pprint support
#                 # Exclude private keys (starting with _) by default
#                 if hasattr(message, "to_dict"):
#                     message = message.to_dict(include_private=False)
#                 message = _pprint.pformat(message, indent=2, width=80, compact=False)
# 
#             # For multi-line messages, indent all lines after the first
#             # to align with the log level prefix (e.g., "INFO: ") plus any indent
#             if "\n" in message:
#                 lines = message.split("\n")
#                 # Calculate the indent needed:
#                 # - 6 chars for "INFO: " or "ERRO: " prefix
#                 # - Plus the indent parameter for additional spacing
#                 total_indent = 6 + indent
#                 prefix_indent = " " * total_indent
#                 # Join lines with proper indentation
#                 message = (
#                     lines[0]
#                     + "\n"
#                     + "\n".join(prefix_indent + line for line in lines[1:])
#                 )
# 
#         # Add separator lines if requested
#         if sep is not None:
#             separator = sep * n_sep
#             message = f"\n{separator}\n{message}\n{separator}"
# 
#         # Add indent and color info to extra
#         if indent > 0 or sep is not None or c is not None:
#             extra = kwargs.get("extra", {})
#             extra["indent"] = indent
#             if c is not None:
#                 extra["color"] = c
#             kwargs["extra"] = extra
# 
#         self._log(level, message, args, **kwargs)
# 
#     def debug(
#         self,
#         message,
#         *args,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         **kwargs,
#     ):
#         """Log a debug message with optional indent, separator, color, and pprint."""
#         if self.isEnabledFor(logging.DEBUG):
#             self._log_with_indent(
#                 logging.DEBUG, message, indent, sep, n_sep, c, pprint, *args, **kwargs
#             )
# 
#     def info(
#         self,
#         message,
#         *args,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         **kwargs,
#     ):
#         """Log an info message with optional indent, separator, color, and pprint."""
#         if self.isEnabledFor(logging.INFO):
#             self._log_with_indent(
#                 logging.INFO, message, indent, sep, n_sep, c, pprint, *args, **kwargs
#             )
# 
#     def warning(
#         self,
#         message,
#         *args,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         **kwargs,
#     ):
#         """Log a warning message with optional indent, separator, color, and pprint."""
#         if self.isEnabledFor(logging.WARNING):
#             self._log_with_indent(
#                 logging.WARNING,
#                 message,
#                 indent,
#                 sep,
#                 n_sep,
#                 c,
#                 pprint,
#                 *args,
#                 **kwargs,
#             )
# 
#     def error(
#         self,
#         message,
#         *args,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         **kwargs,
#     ):
#         """Log an error message with optional indent, separator, color, and pprint."""
#         if self.isEnabledFor(logging.ERROR):
#             self._log_with_indent(
#                 logging.ERROR, message, indent, sep, n_sep, c, pprint, *args, **kwargs
#             )
# 
#     def critical(
#         self,
#         message,
#         *args,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         **kwargs,
#     ):
#         """Log a critical message with optional indent, separator, color, and pprint."""
#         if self.isEnabledFor(logging.CRITICAL):
#             self._log_with_indent(
#                 logging.CRITICAL,
#                 message,
#                 indent,
#                 sep,
#                 n_sep,
#                 c,
#                 pprint,
#                 *args,
#                 **kwargs,
#             )
# 
#     def success(
#         self,
#         message,
#         *args,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         **kwargs,
#     ):
#         """Log a success message with optional indent, separator, color, and pprint."""
#         if self.isEnabledFor(SUCCESS):
#             self._log_with_indent(
#                 SUCCESS, message, indent, sep, n_sep, c, pprint, *args, **kwargs
#             )
# 
#     def fail(
#         self,
#         message,
#         *args,
#         indent=0,
#         sep=None,
#         n_sep=40,
#         c=None,
#         pprint=False,
#         **kwargs,
#     ):
#         """Log a failure message with optional indent, separator, color, and pprint."""
#         if self.isEnabledFor(FAIL):
#             self._log_with_indent(
#                 FAIL, message, indent, sep, n_sep, c, pprint, *args, **kwargs
#             )
# 
#     def to(self, file_path, level=None, mode="w"):
#         """Context manager to temporarily log to a specific file.
# 
#         Usage:
#             logger = logging.getLogger(__name__)
#             with logger.to("/path/to/file.log"):
#                 logger.info("This goes to both console and file.log")
# 
#         Args:
#             file_path: Path to log file
#             level: Logging level (default: DEBUG)
#             mode: File mode ('w' for overwrite, 'a' for append)
# 
#         Returns:
#             Context manager
#         """
#         from ._context import log_to_file
# 
#         return log_to_file(file_path, level=level or logging.DEBUG, mode=mode)
# 
# 
# def setup_logger_class():
#     """Setup the custom logger class."""
#     # Set custom logger class before any logger creation
#     logging.setLoggerClass(SciTeXLogger)
# 
#     # Force existing root logger to use custom class
#     root = logging.getLogger()
#     root.__class__ = SciTeXLogger
# 
# 
# __all__ = ["SciTeXLogger", "setup_logger_class"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/logging/_logger.py
# --------------------------------------------------------------------------------
