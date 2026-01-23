#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__docstring.py

"""Tests for scitex.introspect._docstring module."""

import pytest


class TestGetDocstring:
    """Tests for get_docstring function."""

    def test_get_docstring_success(self):
        """Test getting docstring successfully."""
        from scitex.introspect import get_docstring

        result = get_docstring("json.dumps")
        assert result["success"] is True
        assert "docstring" in result
        assert len(result["docstring"]) > 0

    def test_docstring_raw_format(self):
        """Test raw format returns full docstring."""
        from scitex.introspect import get_docstring

        result = get_docstring("json.dumps", format="raw")
        assert result["success"] is True
        assert "docstring" in result

    def test_docstring_summary_format(self):
        """Test summary format returns first line."""
        from scitex.introspect import get_docstring

        result = get_docstring("json.dumps", format="summary")
        assert result["success"] is True
        assert "docstring" in result
        # Summary should be shorter than full docstring
        raw_result = get_docstring("json.dumps", format="raw")
        assert len(result["docstring"]) <= len(raw_result["docstring"])

    def test_docstring_parsed_format(self):
        """Test parsed format extracts sections."""
        from scitex.introspect import get_docstring

        result = get_docstring("json.dumps", format="parsed")
        assert result["success"] is True
        assert "sections" in result

    def test_docstring_no_docstring(self):
        """Test object without docstring."""
        from scitex.introspect import get_docstring

        # Create a function without docstring dynamically
        result = get_docstring("builtins.None")
        # Should handle gracefully
        assert "success" in result

    def test_docstring_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import get_docstring

        result = get_docstring("nonexistent.module")
        assert result["success"] is False
        assert "error" in result
