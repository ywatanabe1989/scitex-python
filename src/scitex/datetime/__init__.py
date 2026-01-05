#!/usr/bin/env python3
# Timestamp: "2026-01-05 14:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/datetime/__init__.py

"""
Scitex datetime module.

Provides utilities for datetime operations including:
- linspace: Create linearly spaced datetime arrays
- normalize_timestamp: Standardize timestamps to consistent format
- to_datetime: Convert various formats to datetime objects
- validate_timestamp_format: Validate timestamp string format
- format_for_filename: Format timestamps for filenames
- format_for_display: Format timestamps for display
- get_time_delta_seconds: Calculate time differences
"""

from ._linspace import linspace
from ._normalize_timestamp import (
    ALTERNATIVE_FORMATS,
    STANDARD_FORMAT,
    format_for_display,
    format_for_filename,
    get_time_delta_seconds,
    normalize_timestamp,
    parse_patient_recording_start_format,
    to_datetime,
    validate_timestamp_format,
)

__all__ = [
    # Core functions
    "linspace",
    "normalize_timestamp",
    "to_datetime",
    # Formatting functions
    "format_for_filename",
    "format_for_display",
    "validate_timestamp_format",
    # Utility functions
    "get_time_delta_seconds",
    "parse_patient_recording_start_format",
    # Constants
    "STANDARD_FORMAT",
    "ALTERNATIVE_FORMATS",
]
