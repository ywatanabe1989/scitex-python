#!/usr/bin/env python3
# Timestamp: "2026-01-05 14:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/datetime/_normalize_timestamp.py

"""
Timestamp Standardization Utilities

Functionality:
- Standardizes timestamps to consistent format defined in CONFIG.FORMATS.TIMESTAMP
- Handles various input formats (datetime objects, strings, timestamps)
- Provides UTC normalization
- Ensures consistent timestamp formatting across the codebase

Input formats supported:
- datetime objects (with or without timezone)
- Unix timestamps (int/float)
- Various string formats

Output:
- Standardized timestamp strings in format: "%Y-%m-%d %H:%M:%S.%f"
- UTC normalized timestamps
- Validation utilities

Prerequisites:
- CONFIG.FORMATS.TIMESTAMP for standard format
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Union

# Default standard format
DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"

# Try to get standard format from config, fallback to default
try:
    import scitex as stx

    CONFIG = stx.io.load_configs()
    STANDARD_FORMAT = (
        getattr(getattr(CONFIG, "FORMATS", None), "TIMESTAMP", None) or DEFAULT_FORMAT
    )
except Exception:
    STANDARD_FORMAT = DEFAULT_FORMAT

# Common alternative formats to try when parsing
ALTERNATIVE_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",  # ISO 8601 with T (no microseconds)
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%d-%m-%Y %H:%M:%S.%f",
    "%d-%m-%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y, %H:%M:%S",  # Format used in REC_START
    "%Y%m%d %H:%M:%S.%f",
    "%Y%m%d %H:%M:%S",
    "%Y-%m-%d_%H:%M:%S.%f",
    "%Y-%m-%d_%H:%M:%S",
]


def normalize_timestamp(
    timestamp: Union[datetime, str, int, float],
    return_as: str = "str",
    normalize_utc: bool = True,
) -> Union[str, datetime, float]:
    """
    Standardize any timestamp format to requested output type.

    Parameters
    ----------
    timestamp : datetime, str, int, or float
        Timestamp in any supported format
    return_as : str
        Output format: "str" (default), "datetime", or "timestamp"
    normalize_utc : bool
        If True, normalize to UTC timezone

    Returns
    -------
    str, datetime, or float
        Standardized timestamp in requested format:
        - "str": String in CONFIG.FORMATS.TIMESTAMP format
        - "datetime": datetime object
        - "timestamp": Unix timestamp (float)

    Examples
    --------
    >>> from datetime import datetime
    >>> dt = datetime(2010, 6, 18, 10, 15, 0)
    >>> normalize_timestamp(dt, return_as="str", normalize_utc=False)
    '2010-06-18 10:15:00'
    """
    # Convert to datetime object
    dt = to_datetime(timestamp)

    # Normalize to UTC if requested
    if normalize_utc:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

    # Return in requested format
    if return_as == "str":
        return dt.strftime(STANDARD_FORMAT)
    elif return_as == "datetime":
        return dt
    elif return_as == "timestamp":
        return dt.timestamp()
    else:
        raise ValueError(
            f"return_as must be 'str', 'datetime', or 'timestamp', got: {return_as}"
        )


def to_datetime(timestamp: Union[datetime, str, int, float]) -> datetime:
    """
    Convert various timestamp formats to datetime object.

    Parameters
    ----------
    timestamp : datetime, str, int, or float
        Timestamp in any supported format

    Returns
    -------
    datetime
        Datetime object

    Raises
    ------
    ValueError
        If string format cannot be parsed
    TypeError
        If timestamp type is not supported
    """
    # Already datetime
    if isinstance(timestamp, datetime):
        return timestamp

    # Unix timestamp (int/float)
    elif isinstance(timestamp, (int, float)):
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)

    # String format
    elif isinstance(timestamp, str):
        # Handle nanosecond precision by truncating to microseconds
        if "." in timestamp and len(timestamp.split(".")[-1]) > 6:
            parts = timestamp.split(".")
            # Keep only first 6 digits of fractional seconds
            truncated_microseconds = parts[-1][:6]
            # Handle cases where there might be additional text after microseconds
            if not truncated_microseconds.isdigit():
                # Extract just the digit portion
                digits = re.match(r"(\d+)", parts[-1])
                if digits:
                    truncated_microseconds = digits.group(1)[:6]
            timestamp = ".".join(parts[:-1] + [truncated_microseconds])

        # Try parsing with various formats
        for fmt in ALTERNATIVE_FORMATS:
            try:
                return datetime.strptime(timestamp, fmt)
            except ValueError:
                continue

        # If no format matched, raise error
        raise ValueError(
            f"Could not parse timestamp string: {timestamp}. "
            f"Tried formats: {ALTERNATIVE_FORMATS}"
        )

    else:
        raise TypeError(
            f"timestamp must be datetime, str, int, or float, got: {type(timestamp)}"
        )


def validate_timestamp_format(timestamp_str: str) -> bool:
    """
    Validate that a timestamp string matches the standard format.

    Parameters
    ----------
    timestamp_str : str
        Timestamp string to validate

    Returns
    -------
    bool
        True if string matches standard format
    """
    try:
        datetime.strptime(timestamp_str, STANDARD_FORMAT)
        return True
    except (ValueError, TypeError):
        return False


def format_for_filename(timestamp: Union[datetime, str]) -> str:
    """
    Format timestamp for use in filenames (no spaces or colons).

    Parameters
    ----------
    timestamp : datetime or str
        Timestamp to format

    Returns
    -------
    str
        Filename-safe timestamp string (YYYYMMDD_HHMMSS)

    Examples
    --------
    >>> from datetime import datetime
    >>> dt = datetime(2010, 6, 18, 10, 15, 0)
    >>> format_for_filename(dt)
    '20100618_101500'
    """
    dt = to_datetime(timestamp)
    return dt.strftime("%Y%m%d_%H%M%S")


def format_for_display(timestamp: Union[datetime, str]) -> str:
    """
    Format timestamp for human-readable display.

    Parameters
    ----------
    timestamp : datetime or str
        Timestamp to format

    Returns
    -------
    str
        Human-readable timestamp string

    Examples
    --------
    >>> from datetime import datetime
    >>> dt = datetime(2010, 6, 18, 10, 15, 0)
    >>> format_for_display(dt)
    '2010-06-18 10:15:00'
    """
    dt = to_datetime(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_patient_recording_start_format(
    patient_recording_start_str: str,
) -> datetime:
    """
    Parse recording start time from CONFIG.PATIENTS.REC_START format.

    Parameters
    ----------
    patient_recording_start_str : str
        Recording start time string in format "DD/MM/YYYY, HH:MM:SS"

    Returns
    -------
    datetime
        Parsed datetime object

    Examples
    --------
    >>> parse_patient_recording_start_format("10/06/2010, 07:40:34")
    datetime.datetime(2010, 6, 10, 7, 40, 34)
    """
    REC_START_FORMAT = "%d/%m/%Y, %H:%M:%S"
    return datetime.strptime(patient_recording_start_str, REC_START_FORMAT)


def get_time_delta_seconds(
    start: Union[datetime, str], end: Union[datetime, str]
) -> float:
    """
    Calculate time difference in seconds between two timestamps.

    Parameters
    ----------
    start : datetime or str
        Start timestamp
    end : datetime or str
        End timestamp

    Returns
    -------
    float
        Time difference in seconds
    """
    start_dt = to_datetime(start)
    end_dt = to_datetime(end)
    delta = end_dt - start_dt
    return delta.total_seconds()


# EOF
