#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-08-21 20:09:30 (ywatanabe)"
# File: /home/ywatanabe/proj/SciTeX-Code/src/scitex/logging/__init__.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Modular logging utilities for SciTeX.

This module provides enhanced logging capabilities with both console and file output,
ensuring consistent logging across the SciTeX package.

Migration:
    # OLD (deprecated)
    from scitex import logging
    logger = logging.getLogger(__name__)

    # NEW (recommended)
    from scitex import logging
    logger = logging.getLogger(__name__)

Usage:
    from scitex import logging  # DEPRECATED
    logger = logging.getLogger(__name__)
    logger.success("Operation completed successfully")
    logger.fail("Operation failed")

    # Configure logging with file output
    logging.configure(level='info', enable_file=True)

    # Get current log file location
    log_file = logging.get_log_path()
"""

import logging as _logging

from ._Tee import Tee, tee

# Import modular components
from ._levels import SUCCESS, FAIL, DEBUG, INFO, WARNING, ERROR, CRITICAL
from ._logger import SciTeXLogger, setup_logger_class
from ._formatters import SciTeXConsoleFormatter, SciTeXFileFormatter
from ._handlers import create_console_handler, create_file_handler, get_default_log_path
from ._config import (
    set_level,
    get_level,
    enable_file_logging,
    is_file_logging_enabled,
    configure,
    get_log_path,
)
from ._print_capture import (
    PrintCapture,
    enable_print_capture,
    disable_print_capture,
    is_print_capture_enabled,
)
from ._context import log_to_file

# Warnings (like Python's warnings module)
from ._warnings import (
    SciTeXWarning,
    UnitWarning,
    StyleWarning,
    SciTeXDeprecationWarning,
    PerformanceWarning,
    DataLossWarning,
    warn,
    filterwarnings,
    resetwarnings,
    warn_deprecated,
    warn_performance,
    warn_data_loss,
)

# Errors (exceptions)
from ._errors import (
    SciTeXError,
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigKeyError,
    IOError,
    FileFormatError,
    SaveError,
    LoadError,
    ScholarError,
    SearchError,
    EnrichmentError,
    PDFDownloadError,
    DOIResolutionError,
    PDFExtractionError,
    BibTeXEnrichmentError,
    TranslatorError,
    AuthenticationError,
    PlottingError,
    FigureNotFoundError,
    AxisError,
    DataError,
    ShapeError,
    DTypeError,
    PathError,
    InvalidPathError,
    PathNotFoundError,
    TemplateError,
    TemplateViolationError,
    NNError,
    ModelError,
    StatsError,
    TestError,
    check_path,
    check_file_exists,
    check_shape_compatibility,
)

# Re-export standard logging functions for compatibility
getLogger = _logging.getLogger
basicConfig = _logging.basicConfig
disable = _logging.disable

level_by_env = os.getenv("SCITEX_LOGGING_LEVEL", "INFO").upper()
level_map = {
    "DEBU": DEBUG,
    "DEBUG": DEBUG,
    "INFO": INFO,
    "WARN": WARNING,
    "WARNING": WARNING,
    "ERRO": ERROR,
    "ERROR": ERROR,
    "CRIT": CRITICAL,
    "CRITICAL": CRITICAL,
    "SUCC": SUCCESS,
    "SUCCESS": SUCCESS,
    "FAIL": FAIL,
}
level = level_map.get(level_by_env, INFO)

# Auto-configure logging on import with file logging enabled, print capture disabled by default
configure(level=level, enable_file=True, enable_console=True, capture_prints=False)

# Export public API
__all__ = [
    # Core logging functions
    "getLogger",
    # Log levels
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "SUCCESS",
    "FAIL",
    # Configuration
    "configure",
    "get_log_path",
    "Tee",
    "tee",
    # Context managers
    "log_to_file",
    # Warnings (like Python's warnings module)
    "SciTeXWarning",
    "UnitWarning",
    "StyleWarning",
    "SciTeXDeprecationWarning",
    "PerformanceWarning",
    "DataLossWarning",
    "warn",
    "filterwarnings",
    "resetwarnings",
    "warn_deprecated",
    "warn_performance",
    "warn_data_loss",
    # Errors (exceptions)
    "SciTeXError",
    "ConfigurationError",
    "ConfigFileNotFoundError",
    "ConfigKeyError",
    "IOError",
    "FileFormatError",
    "SaveError",
    "LoadError",
    "ScholarError",
    "SearchError",
    "EnrichmentError",
    "PDFDownloadError",
    "DOIResolutionError",
    "PDFExtractionError",
    "BibTeXEnrichmentError",
    "TranslatorError",
    "AuthenticationError",
    "PlottingError",
    "FigureNotFoundError",
    "AxisError",
    "DataError",
    "ShapeError",
    "DTypeError",
    "PathError",
    "InvalidPathError",
    "PathNotFoundError",
    "TemplateError",
    "TemplateViolationError",
    "NNError",
    "ModelError",
    "StatsError",
    "TestError",
    # Validation helpers
    "check_path",
    "check_file_exists",
    "check_shape_compatibility",
]

# EOF
