#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__members.py

"""Tests for scitex.introspect._members module."""

import pytest


class TestDir:
    """Tests for dir function."""

    def test_dir_success(self):
        """Test listing members successfully."""
        from scitex.introspect import dir

        result = dir("json")
        assert result["success"] is True
        assert "members" in result
        assert result["count"] > 0

    def test_dir_public_filter(self):
        """Test public filter excludes private members."""
        from scitex.introspect import dir

        result = dir("json", filter="public")
        assert result["success"] is True
        for m in result["members"]:
            assert not m["name"].startswith("_")

    def test_dir_private_filter(self):
        """Test private filter returns private members."""
        from scitex.introspect import dir

        result = dir("json", filter="private")
        assert result["success"] is True
        # All should start with _ but not __
        for m in result["members"]:
            assert m["name"].startswith("_") and not m["name"].startswith("__")

    def test_dir_dunder_filter(self):
        """Test dunder filter returns dunder members."""
        from scitex.introspect import dir

        result = dir("json", filter="dunder")
        assert result["success"] is True
        for m in result["members"]:
            assert m["name"].startswith("__")

    def test_dir_kind_functions(self):
        """Test filtering by function kind."""
        from scitex.introspect import dir

        result = dir("json", kind="functions")
        assert result["success"] is True
        for m in result["members"]:
            assert m["kind"] == "function"

    def test_dir_kind_classes(self):
        """Test filtering by class kind."""
        from scitex.introspect import dir

        result = dir("pathlib", kind="classes")
        assert result["success"] is True
        for m in result["members"]:
            assert m["kind"] == "class"

    def test_dir_has_summary(self):
        """Test members include summary from docstring."""
        from scitex.introspect import dir

        result = dir("json", filter="public")
        assert result["success"] is True
        # At least some members should have summaries
        summaries = [m["summary"] for m in result["members"] if m["summary"]]
        assert len(summaries) > 0

    def test_dir_class_target(self):
        """Test listing members of a class."""
        from scitex.introspect import dir

        result = dir("pathlib.Path")
        assert result["success"] is True
        assert result["count"] > 0

    def test_dir_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import dir

        result = dir("nonexistent.module")
        assert result["success"] is False
        assert "error" in result


class TestGetExports:
    """Tests for get_exports function."""

    def test_get_exports_with_all(self):
        """Test getting exports from module with __all__."""
        from scitex.introspect import get_exports

        result = get_exports("json")
        assert result["success"] is True
        assert "exports" in result
        assert "has_all" in result

    def test_exports_without_all(self):
        """Test getting exports from module without __all__."""
        from scitex.introspect import get_exports

        # Many stdlib modules don't have __all__
        result = get_exports("scitex.introspect._resolve")
        assert result["success"] is True
        # Should still return public members

    def test_exports_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import get_exports

        result = get_exports("nonexistent.module")
        assert result["success"] is False
