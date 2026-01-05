#!/usr/bin/env python3
# Timestamp: "2026-01-05 14:30:00 (ywatanabe)"
# File: tests/scitex/datetime/test__normalize_timestamp.py

"""Comprehensive tests for datetime._normalize_timestamp module"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestNormalizeTimestamp:
    """Test suite for normalize_timestamp function"""

    def test_normalize_datetime_to_str(self):
        """Test normalizing datetime object to string"""
        from scitex.datetime import normalize_timestamp

        dt = datetime(2010, 6, 18, 10, 15, 0)
        result = normalize_timestamp(dt, return_as="str", normalize_utc=False)

        assert isinstance(result, str)
        assert "2010-06-18" in result
        assert "10:15:00" in result

    def test_normalize_datetime_to_datetime(self):
        """Test normalizing datetime to datetime with UTC"""
        from scitex.datetime import normalize_timestamp

        dt = datetime(2010, 6, 18, 10, 15, 0)
        result = normalize_timestamp(dt, return_as="datetime", normalize_utc=True)

        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_normalize_datetime_to_timestamp(self):
        """Test normalizing datetime to Unix timestamp"""
        from scitex.datetime import normalize_timestamp

        dt = datetime(2010, 6, 18, 10, 15, 0, tzinfo=timezone.utc)
        result = normalize_timestamp(dt, return_as="timestamp")

        assert isinstance(result, float)
        assert result > 0

    def test_normalize_unix_timestamp(self):
        """Test normalizing Unix timestamp"""
        from scitex.datetime import normalize_timestamp

        unix_ts = 1276856100.0
        result = normalize_timestamp(unix_ts, return_as="datetime")

        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_normalize_string_iso_format(self):
        """Test normalizing ISO format string"""
        from scitex.datetime import normalize_timestamp

        ts_str = "2010-06-18T10:15:00"
        result = normalize_timestamp(ts_str, return_as="datetime", normalize_utc=False)

        assert isinstance(result, datetime)
        assert result.year == 2010
        assert result.month == 6
        assert result.day == 18

    def test_normalize_string_with_microseconds(self):
        """Test normalizing string with microseconds"""
        from scitex.datetime import normalize_timestamp

        ts_str = "2010-06-18 10:15:00.123456"
        result = normalize_timestamp(ts_str, return_as="datetime", normalize_utc=False)

        assert isinstance(result, datetime)
        assert result.microsecond == 123456

    def test_normalize_various_string_formats(self):
        """Test normalizing various string formats"""
        from scitex.datetime import normalize_timestamp

        formats = [
            "2010-06-18 10:15:00",
            "2010/06/18 10:15:00",
            "18-06-2010 10:15:00",
            "18/06/2010 10:15:00",
        ]

        for ts_str in formats:
            result = normalize_timestamp(
                ts_str, return_as="datetime", normalize_utc=False
            )
            assert isinstance(result, datetime)

    def test_normalize_without_utc(self):
        """Test normalize_utc=False preserves naive datetime"""
        from scitex.datetime import normalize_timestamp

        dt = datetime(2010, 6, 18, 10, 15, 0)
        result = normalize_timestamp(dt, return_as="datetime", normalize_utc=False)

        assert result.tzinfo is None

    def test_normalize_with_utc_aware_input(self):
        """Test normalizing timezone-aware datetime to UTC"""
        from scitex.datetime import normalize_timestamp

        # Create a datetime with a non-UTC timezone
        dt = datetime(2010, 6, 18, 10, 15, 0, tzinfo=timezone(timedelta(hours=5)))
        result = normalize_timestamp(dt, return_as="datetime", normalize_utc=True)

        assert result.tzinfo == timezone.utc
        # Should be 5 hours earlier in UTC
        assert result.hour == 5

    def test_normalize_invalid_return_as(self):
        """Test that invalid return_as raises ValueError"""
        from scitex.datetime import normalize_timestamp

        dt = datetime(2010, 6, 18, 10, 15, 0)

        with pytest.raises(ValueError, match="return_as must be"):
            normalize_timestamp(dt, return_as="invalid")

    def test_normalize_int_timestamp(self):
        """Test normalizing integer Unix timestamp"""
        from scitex.datetime import normalize_timestamp

        unix_ts = 1276856100
        result = normalize_timestamp(unix_ts, return_as="datetime")

        assert isinstance(result, datetime)


class TestToDatetime:
    """Test suite for to_datetime function"""

    def test_to_datetime_from_datetime(self):
        """Test to_datetime with datetime input"""
        from scitex.datetime import to_datetime

        dt = datetime(2010, 6, 18, 10, 15, 0)
        result = to_datetime(dt)

        assert result is dt  # Should return same object

    def test_to_datetime_from_int(self):
        """Test to_datetime with integer Unix timestamp"""
        from scitex.datetime import to_datetime

        unix_ts = 1276856100
        result = to_datetime(unix_ts)

        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_to_datetime_from_float(self):
        """Test to_datetime with float Unix timestamp"""
        from scitex.datetime import to_datetime

        unix_ts = 1276856100.123456
        result = to_datetime(unix_ts)

        assert isinstance(result, datetime)
        assert result.tzinfo == timezone.utc

    def test_to_datetime_from_string_iso(self):
        """Test to_datetime with ISO format string"""
        from scitex.datetime import to_datetime

        ts_str = "2010-06-18T10:15:00"
        result = to_datetime(ts_str)

        assert result.year == 2010
        assert result.month == 6
        assert result.day == 18
        assert result.hour == 10
        assert result.minute == 15

    def test_to_datetime_from_string_standard(self):
        """Test to_datetime with standard format string"""
        from scitex.datetime import to_datetime

        ts_str = "2010-06-18 10:15:00"
        result = to_datetime(ts_str)

        assert result.year == 2010
        assert result.month == 6
        assert result.day == 18

    def test_to_datetime_nanosecond_truncation(self):
        """Test that nanosecond precision is truncated to microseconds"""
        from scitex.datetime import to_datetime

        ts_str = "2010-06-18 10:15:00.123456789"
        result = to_datetime(ts_str)

        assert result.microsecond == 123456

    def test_to_datetime_invalid_string(self):
        """Test that invalid string raises ValueError"""
        from scitex.datetime import to_datetime

        with pytest.raises(ValueError, match="Could not parse timestamp string"):
            to_datetime("not-a-valid-timestamp")

    def test_to_datetime_invalid_type(self):
        """Test that invalid type raises TypeError"""
        from scitex.datetime import to_datetime

        with pytest.raises(TypeError, match="timestamp must be"):
            to_datetime([2010, 6, 18])

    def test_to_datetime_alternative_formats(self):
        """Test parsing various alternative formats"""
        from scitex.datetime import to_datetime

        test_cases = [
            ("2010-06-18T10:15:00.123456", 2010, 6, 18),
            ("2010/06/18 10:15:00", 2010, 6, 18),
            ("18-06-2010 10:15:00", 2010, 6, 18),
            ("18/06/2010 10:15:00", 2010, 6, 18),
            ("18/06/2010, 10:15:00", 2010, 6, 18),  # REC_START format
            ("20100618 10:15:00", 2010, 6, 18),
            ("2010-06-18_10:15:00", 2010, 6, 18),
        ]

        for ts_str, year, month, day in test_cases:
            result = to_datetime(ts_str)
            assert result.year == year, f"Failed for {ts_str}"
            assert result.month == month, f"Failed for {ts_str}"
            assert result.day == day, f"Failed for {ts_str}"


class TestValidateTimestampFormat:
    """Test suite for validate_timestamp_format function"""

    def test_validate_valid_format(self):
        """Test validation of correctly formatted string"""
        from scitex.datetime import STANDARD_FORMAT, validate_timestamp_format

        # Create a string in the standard format
        dt = datetime(2010, 6, 18, 10, 15, 0)
        ts_str = dt.strftime(STANDARD_FORMAT)

        assert validate_timestamp_format(ts_str) is True

    def test_validate_invalid_format(self):
        """Test validation of incorrectly formatted string"""
        from scitex.datetime import validate_timestamp_format

        assert validate_timestamp_format("not-a-timestamp") is False

    def test_validate_wrong_format(self):
        """Test validation of timestamp in wrong format"""
        from scitex.datetime import validate_timestamp_format

        # ISO format might not match standard format
        assert validate_timestamp_format("2010-06-18T10:15:00") is False

    def test_validate_none_input(self):
        """Test validation with None input"""
        from scitex.datetime import validate_timestamp_format

        assert validate_timestamp_format(None) is False


class TestFormatForFilename:
    """Test suite for format_for_filename function"""

    def test_format_datetime_for_filename(self):
        """Test formatting datetime for filename"""
        from scitex.datetime import format_for_filename

        dt = datetime(2010, 6, 18, 10, 15, 0)
        result = format_for_filename(dt)

        assert result == "20100618_101500"
        assert " " not in result
        assert ":" not in result

    def test_format_string_for_filename(self):
        """Test formatting string timestamp for filename"""
        from scitex.datetime import format_for_filename

        ts_str = "2010-06-18 10:15:00"
        result = format_for_filename(ts_str)

        assert result == "20100618_101500"

    def test_format_for_filename_no_special_chars(self):
        """Test that filename format has no special characters"""
        from scitex.datetime import format_for_filename

        dt = datetime(2010, 6, 18, 10, 15, 30)
        result = format_for_filename(dt)

        # Should only contain digits and underscore
        assert all(c.isdigit() or c == "_" for c in result)


class TestFormatForDisplay:
    """Test suite for format_for_display function"""

    def test_format_datetime_for_display(self):
        """Test formatting datetime for display"""
        from scitex.datetime import format_for_display

        dt = datetime(2010, 6, 18, 10, 15, 0)
        result = format_for_display(dt)

        assert result == "2010-06-18 10:15:00"

    def test_format_string_for_display(self):
        """Test formatting string timestamp for display"""
        from scitex.datetime import format_for_display

        ts_str = "2010/06/18 10:15:00"
        result = format_for_display(ts_str)

        assert result == "2010-06-18 10:15:00"

    def test_format_for_display_readable(self):
        """Test that display format is human-readable"""
        from scitex.datetime import format_for_display

        dt = datetime(2010, 6, 18, 10, 15, 30)
        result = format_for_display(dt)

        # Should have readable date and time separated by space
        assert " " in result
        assert "-" in result
        assert ":" in result


class TestParsePatientRecordingStartFormat:
    """Test suite for parse_patient_recording_start_format function"""

    def test_parse_rec_start_format(self):
        """Test parsing REC_START format"""
        from scitex.datetime import parse_patient_recording_start_format

        ts_str = "10/06/2010, 07:40:34"
        result = parse_patient_recording_start_format(ts_str)

        assert result.year == 2010
        assert result.month == 6
        assert result.day == 10
        assert result.hour == 7
        assert result.minute == 40
        assert result.second == 34

    def test_parse_rec_start_invalid_format(self):
        """Test that invalid REC_START format raises ValueError"""
        from scitex.datetime import parse_patient_recording_start_format

        with pytest.raises(ValueError):
            parse_patient_recording_start_format("2010-06-10 07:40:34")


class TestGetTimeDeltaSeconds:
    """Test suite for get_time_delta_seconds function"""

    def test_get_delta_positive(self):
        """Test positive time delta"""
        from scitex.datetime import get_time_delta_seconds

        start = datetime(2010, 6, 18, 10, 0, 0)
        end = datetime(2010, 6, 18, 10, 1, 0)

        result = get_time_delta_seconds(start, end)

        assert result == 60.0

    def test_get_delta_negative(self):
        """Test negative time delta (end before start)"""
        from scitex.datetime import get_time_delta_seconds

        start = datetime(2010, 6, 18, 10, 1, 0)
        end = datetime(2010, 6, 18, 10, 0, 0)

        result = get_time_delta_seconds(start, end)

        assert result == -60.0

    def test_get_delta_with_strings(self):
        """Test time delta with string inputs"""
        from scitex.datetime import get_time_delta_seconds

        start = "2010-06-18 10:00:00"
        end = "2010-06-18 10:01:00"

        result = get_time_delta_seconds(start, end)

        assert result == 60.0

    def test_get_delta_mixed_inputs(self):
        """Test time delta with mixed input types"""
        from scitex.datetime import get_time_delta_seconds

        start = datetime(2010, 6, 18, 10, 0, 0)
        end = "2010-06-18 11:00:00"

        result = get_time_delta_seconds(start, end)

        assert result == 3600.0  # 1 hour

    def test_get_delta_same_time(self):
        """Test time delta when start equals end"""
        from scitex.datetime import get_time_delta_seconds

        dt = datetime(2010, 6, 18, 10, 0, 0)

        result = get_time_delta_seconds(dt, dt)

        assert result == 0.0

    def test_get_delta_large_difference(self):
        """Test time delta with large time difference"""
        from scitex.datetime import get_time_delta_seconds

        start = datetime(2010, 1, 1, 0, 0, 0)
        end = datetime(2011, 1, 1, 0, 0, 0)  # 1 year later

        result = get_time_delta_seconds(start, end)

        # Should be approximately 365 days in seconds
        assert result == pytest.approx(365 * 24 * 60 * 60, rel=0.01)

    def test_get_delta_with_microseconds(self):
        """Test time delta with microsecond precision"""
        from scitex.datetime import get_time_delta_seconds

        start = datetime(2010, 6, 18, 10, 0, 0, 0)
        end = datetime(2010, 6, 18, 10, 0, 0, 500000)  # 0.5 seconds

        result = get_time_delta_seconds(start, end)

        assert result == 0.5


class TestConstants:
    """Test suite for module constants"""

    def test_standard_format_defined(self):
        """Test that STANDARD_FORMAT is defined"""
        from scitex.datetime import STANDARD_FORMAT

        assert STANDARD_FORMAT is not None
        assert isinstance(STANDARD_FORMAT, str)
        assert "%" in STANDARD_FORMAT

    def test_alternative_formats_defined(self):
        """Test that ALTERNATIVE_FORMATS is defined and non-empty"""
        from scitex.datetime import ALTERNATIVE_FORMATS

        assert ALTERNATIVE_FORMATS is not None
        assert isinstance(ALTERNATIVE_FORMATS, list)
        assert len(ALTERNATIVE_FORMATS) > 0
        assert all(isinstance(fmt, str) for fmt in ALTERNATIVE_FORMATS)


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_epoch_timestamp(self):
        """Test with Unix epoch (0)"""
        from scitex.datetime import to_datetime

        result = to_datetime(0)

        assert result.year == 1970
        assert result.month == 1
        assert result.day == 1

    def test_large_timestamp(self):
        """Test with large Unix timestamp"""
        from scitex.datetime import to_datetime

        # Year 2100
        large_ts = 4102444800
        result = to_datetime(large_ts)

        assert result.year == 2100

    def test_negative_timestamp(self):
        """Test with negative Unix timestamp (before epoch)"""
        from scitex.datetime import to_datetime

        # 1969
        neg_ts = -31536000
        result = to_datetime(neg_ts)

        assert result.year == 1969

    def test_microsecond_precision_preserved(self):
        """Test that microsecond precision is preserved"""
        from scitex.datetime import normalize_timestamp

        dt = datetime(2010, 6, 18, 10, 15, 0, 123456)
        result = normalize_timestamp(dt, return_as="datetime", normalize_utc=False)

        assert result.microsecond == 123456

    def test_roundtrip_datetime_timestamp_datetime(self):
        """Test roundtrip: datetime -> timestamp -> datetime"""
        from scitex.datetime import normalize_timestamp

        original = datetime(2010, 6, 18, 10, 15, 0, tzinfo=timezone.utc)

        # Convert to timestamp
        ts = normalize_timestamp(original, return_as="timestamp")

        # Convert back to datetime
        result = normalize_timestamp(ts, return_as="datetime")

        assert result.year == original.year
        assert result.month == original.month
        assert result.day == original.day
        assert result.hour == original.hour
        assert result.minute == original.minute


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# --------------------------------------------------------------------------------
