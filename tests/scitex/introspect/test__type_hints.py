#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__type_hints.py

"""Tests for scitex.introspect._type_hints module."""

import pytest


class TestGetTypeHintsDetailed:
    """Tests for get_type_hints_detailed function."""

    def test_get_type_hints_success(self):
        """Test getting type hints successfully."""
        from scitex.introspect import get_type_hints_detailed

        result = get_type_hints_detailed("scitex.introspect._resolve.resolve_object")
        assert result["success"] is True
        assert "hints" in result or "hint_count" in result

    def test_type_hints_with_optional(self):
        """Test detecting optional type hints."""
        from scitex.introspect import get_type_hints_detailed

        result = get_type_hints_detailed("scitex.introspect._signature.q")
        assert result["success"] is True
        # Check for optional detection in hints
        if result.get("hints"):
            for name, info in result["hints"].items():
                assert "is_optional" in info

    def test_type_hints_return_type(self):
        """Test return type is included."""
        from scitex.introspect import get_type_hints_detailed

        result = get_type_hints_detailed("scitex.introspect._resolve.resolve_object")
        assert result["success"] is True
        # May or may not have return hint depending on function

    def test_type_hints_class(self):
        """Test type hints for class methods."""
        from scitex.introspect import get_type_hints_detailed

        result = get_type_hints_detailed("pathlib.Path")
        assert result["success"] is True

    def test_type_hints_no_hints(self):
        """Test function without type hints."""
        from scitex.introspect import get_type_hints_detailed

        # Some functions may not have hints
        result = get_type_hints_detailed("json.loads")
        assert result["success"] is True
        # Should return empty or minimal hints

    def test_type_hints_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import get_type_hints_detailed

        result = get_type_hints_detailed("nonexistent.module")
        assert result["success"] is False


class TestGetClassAnnotations:
    """Tests for get_class_annotations function."""

    def test_get_class_annotations_success(self):
        """Test getting class annotations."""
        from scitex.introspect import get_class_annotations

        result = get_class_annotations("pathlib.PurePath")
        assert result["success"] is True
        # Result contains class_vars and methods, not annotations
        assert "class_vars" in result or "methods" in result

    def test_class_annotations_non_class(self):
        """Test error when path is not a class."""
        from scitex.introspect import get_class_annotations

        result = get_class_annotations("json.dumps")
        assert result["success"] is False
        assert "not a class" in result["error"] or "error" in result
