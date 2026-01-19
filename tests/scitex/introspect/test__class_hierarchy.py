#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__class_hierarchy.py

"""Tests for scitex.introspect._class_hierarchy module."""

import pytest


class TestGetClassHierarchy:
    """Tests for get_class_hierarchy function."""

    def test_get_hierarchy_success(self):
        """Test getting class hierarchy successfully."""
        from scitex.introspect import get_class_hierarchy

        result = get_class_hierarchy("collections.abc.Mapping")
        assert result["success"] is True
        assert "mro" in result
        assert "subclasses" in result
        assert result["mro_count"] > 0

    def test_hierarchy_mro_order(self):
        """Test MRO is in correct order."""
        from scitex.introspect import get_class_hierarchy

        result = get_class_hierarchy("collections.abc.MutableMapping")
        assert result["success"] is True
        # MutableMapping should have Mapping in its MRO
        mro_names = [c["name"] for c in result["mro"]]
        assert "MutableMapping" in mro_names
        assert "Mapping" in mro_names

    def test_hierarchy_without_builtins(self):
        """Test hierarchy excludes builtins by default."""
        from scitex.introspect import get_class_hierarchy

        result = get_class_hierarchy("pathlib.Path", include_builtins=False)
        assert result["success"] is True
        mro_names = [c["name"] for c in result["mro"]]
        assert "object" not in mro_names

    def test_hierarchy_with_builtins(self):
        """Test hierarchy includes builtins when requested."""
        from scitex.introspect import get_class_hierarchy

        result = get_class_hierarchy("pathlib.Path", include_builtins=True)
        assert result["success"] is True
        mro_names = [c["name"] for c in result["mro"]]
        assert "object" in mro_names

    def test_hierarchy_non_class_error(self):
        """Test error when path is not a class."""
        from scitex.introspect import get_class_hierarchy

        result = get_class_hierarchy("json.dumps")
        assert result["success"] is False
        assert "not a class" in result["error"]

    def test_hierarchy_max_depth(self):
        """Test max_depth limits subclass traversal."""
        from scitex.introspect import get_class_hierarchy

        result = get_class_hierarchy("collections.abc.Mapping", max_depth=1)
        assert result["success"] is True
        # With max_depth=1, nested subclasses should not have children
        for sub in result.get("subclasses", []):
            assert "subclasses" not in sub or len(sub["subclasses"]) == 0


class TestGetMro:
    """Tests for get_mro function."""

    def test_get_mro_success(self):
        """Test getting MRO successfully."""
        from scitex.introspect import get_mro

        result = get_mro("collections.OrderedDict")
        assert result["success"] is True
        assert "mro" in result
        assert len(result["mro"]) > 0

    def test_mro_qualnames(self):
        """Test MRO returns qualified names."""
        from scitex.introspect import get_mro

        result = get_mro("pathlib.Path")
        assert result["success"] is True
        # Each entry should be a qualified name
        for name in result["mro"]:
            assert "." in name
