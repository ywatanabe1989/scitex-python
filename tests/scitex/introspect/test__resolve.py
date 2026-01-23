#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__resolve.py

"""Tests for scitex.introspect._resolve module."""

import pytest


class TestResolveObject:
    """Tests for resolve_object function."""

    def test_resolve_module(self):
        """Test resolving a module."""
        from scitex.introspect import resolve_object

        obj, error = resolve_object("json")
        assert error is None
        assert obj is not None
        import json

        assert obj is json

    def test_resolve_function(self):
        """Test resolving a function."""
        from scitex.introspect import resolve_object

        obj, error = resolve_object("json.dumps")
        assert error is None
        import json

        assert obj is json.dumps

    def test_resolve_class(self):
        """Test resolving a class."""
        from scitex.introspect import resolve_object

        obj, error = resolve_object("pathlib.Path")
        assert error is None
        from pathlib import Path

        assert obj is Path

    def test_resolve_nested(self):
        """Test resolving nested attribute."""
        from scitex.introspect import resolve_object

        obj, error = resolve_object("collections.abc.Mapping")
        assert error is None
        from collections.abc import Mapping

        assert obj is Mapping

    def test_resolve_invalid_returns_error(self):
        """Test resolving invalid path returns error."""
        from scitex.introspect import resolve_object

        obj, error = resolve_object("nonexistent.module.thing")
        assert obj is None
        assert error is not None
        assert "Could not resolve" in error


class TestGetTypeInfo:
    """Tests for get_type_info function."""

    def test_type_info_module(self):
        """Test type info for module."""
        import json

        from scitex.introspect import get_type_info

        info = get_type_info(json)
        assert info["kind"] == "module"

    def test_type_info_function(self):
        """Test type info for function."""
        import json

        from scitex.introspect import get_type_info

        info = get_type_info(json.dumps)
        assert info["kind"] == "function"

    def test_type_info_class(self):
        """Test type info for class."""
        from pathlib import Path

        from scitex.introspect import get_type_info

        info = get_type_info(Path)
        assert info["kind"] == "class"

    def test_type_info_data(self):
        """Test type info for data."""
        from scitex.introspect import get_type_info

        info = get_type_info([1, 2, 3])
        assert info["kind"] == "data"
