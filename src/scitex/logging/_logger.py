#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-17 15:03:33 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/logging/_logger.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/logging/_logger.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

__FILE__ = __file__

"""Enhanced logger class for SciTeX."""

import logging
import pprint as _pprint

from ._levels import FAIL, SUCCESS


class SciTeXLogger(logging.Logger):
    """Enhanced logger with success/fail methods, indent, separator, and color support."""

    def _log_with_indent(
        self,
        level,
        message,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        *args,
        **kwargs,
    ):
        """Internal method to log with indent, separator, color, and pprint support.

        Args:
            level: Logging level
            message: Message to log
            indent: Number of spaces to indent
            sep: Separator character (e.g., '=', '-')
            n_sep: Number of separator characters
            c: Color code
            pprint: If True, format message using pprint for better readability
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        # Apply pprint formatting if requested
        if pprint:
            # If message is a string, keep it as is
            # Otherwise, format it with pprint
            if not isinstance(message, str):
                # Convert DotDict to regular dict for better pprint support
                # Exclude private keys (starting with _) by default
                if hasattr(message, "to_dict"):
                    message = message.to_dict(include_private=False)
                message = _pprint.pformat(message, indent=2, width=80, compact=False)

            # For multi-line messages, indent all lines after the first
            # to align with the log level prefix (e.g., "INFO: ") plus any indent
            if "\n" in message:
                lines = message.split("\n")
                # Calculate the indent needed:
                # - 6 chars for "INFO: " or "ERRO: " prefix
                # - Plus the indent parameter for additional spacing
                total_indent = 6 + indent
                prefix_indent = " " * total_indent
                # Join lines with proper indentation
                message = (
                    lines[0]
                    + "\n"
                    + "\n".join(prefix_indent + line for line in lines[1:])
                )

        # Add separator lines if requested
        if sep is not None:
            separator = sep * n_sep
            message = f"\n{separator}\n{message}\n{separator}"

        # Add indent and color info to extra
        if indent > 0 or sep is not None or c is not None:
            extra = kwargs.get("extra", {})
            extra["indent"] = indent
            if c is not None:
                extra["color"] = c
            kwargs["extra"] = extra

        self._log(level, message, args, **kwargs)

    def debug(
        self,
        message,
        *args,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        **kwargs,
    ):
        """Log a debug message with optional indent, separator, color, and pprint."""
        if self.isEnabledFor(logging.DEBUG):
            self._log_with_indent(
                logging.DEBUG, message, indent, sep, n_sep, c, pprint, *args, **kwargs
            )

    def info(
        self,
        message,
        *args,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        **kwargs,
    ):
        """Log an info message with optional indent, separator, color, and pprint."""
        if self.isEnabledFor(logging.INFO):
            self._log_with_indent(
                logging.INFO, message, indent, sep, n_sep, c, pprint, *args, **kwargs
            )

    def warning(
        self,
        message,
        *args,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        **kwargs,
    ):
        """Log a warning message with optional indent, separator, color, and pprint."""
        if self.isEnabledFor(logging.WARNING):
            self._log_with_indent(
                logging.WARNING,
                message,
                indent,
                sep,
                n_sep,
                c,
                pprint,
                *args,
                **kwargs,
            )

    def error(
        self,
        message,
        *args,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        **kwargs,
    ):
        """Log an error message with optional indent, separator, color, and pprint."""
        if self.isEnabledFor(logging.ERROR):
            self._log_with_indent(
                logging.ERROR, message, indent, sep, n_sep, c, pprint, *args, **kwargs
            )

    def critical(
        self,
        message,
        *args,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        **kwargs,
    ):
        """Log a critical message with optional indent, separator, color, and pprint."""
        if self.isEnabledFor(logging.CRITICAL):
            self._log_with_indent(
                logging.CRITICAL,
                message,
                indent,
                sep,
                n_sep,
                c,
                pprint,
                *args,
                **kwargs,
            )

    def success(
        self,
        message,
        *args,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        **kwargs,
    ):
        """Log a success message with optional indent, separator, color, and pprint."""
        if self.isEnabledFor(SUCCESS):
            self._log_with_indent(
                SUCCESS, message, indent, sep, n_sep, c, pprint, *args, **kwargs
            )

    def fail(
        self,
        message,
        *args,
        indent=0,
        sep=None,
        n_sep=40,
        c=None,
        pprint=False,
        **kwargs,
    ):
        """Log a failure message with optional indent, separator, color, and pprint."""
        if self.isEnabledFor(FAIL):
            self._log_with_indent(
                FAIL, message, indent, sep, n_sep, c, pprint, *args, **kwargs
            )

    def to(self, file_path, level=None, mode="w"):
        """Context manager to temporarily log to a specific file.

        Usage:
            logger = logging.getLogger(__name__)
            with logger.to("/path/to/file.log"):
                logger.info("This goes to both console and file.log")

        Args:
            file_path: Path to log file
            level: Logging level (default: DEBUG)
            mode: File mode ('w' for overwrite, 'a' for append)

        Returns:
            Context manager
        """
        from ._context import log_to_file

        return log_to_file(file_path, level=level or logging.DEBUG, mode=mode)


def setup_logger_class():
    """Setup the custom logger class."""
    # Set custom logger class before any logger creation
    logging.setLoggerClass(SciTeXLogger)

    # Force existing root logger to use custom class
    root = logging.getLogger()
    root.__class__ = SciTeXLogger


__all__ = ["SciTeXLogger", "setup_logger_class"]

# EOF
