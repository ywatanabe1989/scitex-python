#!/usr/bin/env python3
# Timestamp: "2026-01-05 14:40:00 (ywatanabe)"
# File: tests/scitex/dt/test__normalize_timestamp.py

"""Tests for dt module as alias for datetime module.

The dt module is an alias for scitex.datetime, providing a shorter name
for convenience. Both modules are fully supported.
"""

from datetime import datetime, timezone

import pytest


class TestDtModuleAlias:
    """Test that dt module correctly aliases datetime module"""

    def test_dt_module_docstring(self):
        """Test that dt module docstring explains the alias."""
        from scitex import dt

        assert dt.__doc__ is not None
        assert "alias" in dt.__doc__.lower()

    def test_dt_module_has_all_exports(self):
        """Test that dt module exports all the same functions as datetime."""
        from scitex import datetime as stx_datetime
        from scitex import dt

        # Verify key functions are available in both modules
        key_functions = [
            "linspace",
            "normalize_timestamp",
            "to_datetime",
            "format_for_filename",
            "format_for_display",
            "validate_timestamp_format",
            "get_time_delta_seconds",
            "STANDARD_FORMAT",
            "ALTERNATIVE_FORMATS",
        ]
        for func_name in key_functions:
            assert hasattr(dt, func_name), f"dt missing {func_name}"
            assert hasattr(stx_datetime, func_name), f"datetime missing {func_name}"


class TestDtModuleFunctions:
    """Test that dt module functions work correctly"""

    def test_dt_linspace(self):
        """Test that dt.linspace works"""
        from scitex.dt import linspace

        start = datetime(2023, 1, 1, 0, 0, 0)
        end = datetime(2023, 1, 1, 0, 0, 10)

        result = linspace(start, end, n_samples=11)

        assert len(result) == 11

    def test_dt_normalize_timestamp(self):
        """Test that dt.normalize_timestamp works"""
        from scitex.dt import normalize_timestamp

        dt_obj = datetime(2010, 6, 18, 10, 15, 0)
        result = normalize_timestamp(dt_obj, return_as="str", normalize_utc=False)

        assert isinstance(result, str)
        assert "2010-06-18" in result

    def test_dt_to_datetime(self):
        """Test that dt.to_datetime works"""
        from scitex.dt import to_datetime

        result = to_datetime("2010-06-18 10:15:00")

        assert result.year == 2010
        assert result.month == 6
        assert result.day == 18

    def test_dt_format_functions(self):
        """Test that dt format functions work"""
        from scitex.dt import format_for_display, format_for_filename

        dt_obj = datetime(2010, 6, 18, 10, 15, 0)

        filename = format_for_filename(dt_obj)
        display = format_for_display(dt_obj)

        assert filename == "20100618_101500"
        assert display == "2010-06-18 10:15:00"

    def test_dt_get_time_delta_seconds(self):
        """Test that dt.get_time_delta_seconds works"""
        from scitex.dt import get_time_delta_seconds

        start = datetime(2010, 6, 18, 10, 0, 0)
        end = datetime(2010, 6, 18, 10, 1, 0)

        result = get_time_delta_seconds(start, end)

        assert result == 60.0

    def test_dt_validate_timestamp_format(self):
        """Test that dt.validate_timestamp_format works"""
        from scitex.dt import STANDARD_FORMAT, validate_timestamp_format

        dt_obj = datetime(2010, 6, 18, 10, 15, 0)
        ts_str = dt_obj.strftime(STANDARD_FORMAT)

        assert validate_timestamp_format(ts_str) is True
        assert validate_timestamp_format("invalid") is False


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])

# --------------------------------------------------------------------------------
