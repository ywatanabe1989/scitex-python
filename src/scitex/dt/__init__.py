#!/usr/bin/env python3
# Timestamp: "2026-01-05 14:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dt/__init__.py

"""
Scitex dt module - alias for scitex.datetime.

This module is an alias for scitex.datetime, providing a shorter name
for convenience. Both modules are fully supported.

Usage:
    from scitex import dt
    dt.linspace(...)

    # Equivalent to:
    from scitex import datetime
    datetime.linspace(...)
"""

# Re-export everything from datetime module
from scitex.datetime import (
    ALTERNATIVE_FORMATS,
    STANDARD_FORMAT,
    format_for_display,
    format_for_filename,
    get_time_delta_seconds,
    linspace,
    normalize_timestamp,
    parse_patient_recording_start_format,
    to_datetime,
    validate_timestamp_format,
)

__all__ = [
    "linspace",
    "normalize_timestamp",
    "to_datetime",
    "format_for_filename",
    "format_for_display",
    "validate_timestamp_format",
    "get_time_delta_seconds",
    "parse_patient_recording_start_format",
    "STANDARD_FORMAT",
    "ALTERNATIVE_FORMATS",
]
