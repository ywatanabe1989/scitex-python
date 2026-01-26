#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__source.py

"""Tests for scitex.introspect._source module."""

import pytest


class TestQQ:
    """Tests for qq function."""

    def test_qq_success(self):
        """Test getting source successfully."""
        from scitex.introspect import qq

        result = qq("scitex.introspect._resolve.resolve_object")
        assert result["success"] is True
        assert "source" in result
        assert "file" in result
        assert "line_start" in result
        assert "line_count" in result

    def test_qq_contains_def(self):
        """Test source contains function definition."""
        from scitex.introspect import qq

        result = qq("scitex.introspect._resolve.resolve_object")
        assert result["success"] is True
        assert "def resolve_object" in result["source"]

    def test_qq_max_lines(self):
        """Test max_lines limits output."""
        from scitex.introspect import qq

        result = qq("scitex.introspect._resolve.resolve_object", max_lines=5)
        assert result["success"] is True
        lines = result["source"].strip().split("\n")
        # May include a truncation indicator line, so allow +1
        assert len(lines) <= 6

    def test_qq_without_decorators(self):
        """Test source excludes decorators when requested."""
        from scitex.introspect import qq

        result = qq(
            "scitex.introspect._resolve.resolve_object", include_decorators=False
        )
        assert result["success"] is True
        # First line should be def, not @decorator
        first_line = result["source"].strip().split("\n")[0].strip()
        assert first_line.startswith("def ")

    def test_qq_builtin_fails(self):
        """Test getting source of builtin fails gracefully."""
        from scitex.introspect import qq

        result = qq("len")
        assert result["success"] is False
        # Builtins don't have Python source

    def test_qq_class(self):
        """Test getting source of a class."""
        from scitex.introspect import qq

        result = qq("pathlib.PurePath")
        assert result["success"] is True
        assert "class PurePath" in result["source"]

    def test_qq_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import qq

        result = qq("nonexistent.module")
        assert result["success"] is False
        assert "error" in result
