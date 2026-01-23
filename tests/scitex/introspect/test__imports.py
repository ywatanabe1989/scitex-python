#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__imports.py

"""Tests for scitex.introspect._imports module."""

import pytest


class TestGetImports:
    """Tests for get_imports function."""

    def test_get_imports_success(self):
        """Test getting imports successfully."""
        from scitex.introspect import get_imports

        result = get_imports("scitex.introspect._resolve")
        assert result["success"] is True
        assert "imports" in result
        assert result["import_count"] > 0

    def test_imports_categorized(self):
        """Test imports are categorized correctly."""
        from scitex.introspect import get_imports

        result = get_imports("scitex.introspect._resolve", categorize=True)
        assert result["success"] is True
        assert "categories" in result
        assert "stdlib" in result["categories"]
        assert "third_party" in result["categories"]
        assert "local" in result["categories"]

    def test_imports_not_categorized(self):
        """Test imports without categorization."""
        from scitex.introspect import get_imports

        result = get_imports("scitex.introspect._resolve", categorize=False)
        assert result["success"] is True
        assert "categories" not in result
        assert "imports" in result

    def test_imports_include_from_imports(self):
        """Test from...import statements are included."""
        from scitex.introspect import get_imports

        result = get_imports("scitex.introspect._resolve")
        assert result["success"] is True
        # Should have 'from' type imports
        from_imports = [i for i in result["imports"] if i["type"] == "from"]
        assert len(from_imports) > 0

    def test_imports_non_module_error(self):
        """Test error when path is not a module."""
        from scitex.introspect import get_imports

        result = get_imports("json.dumps")
        assert result["success"] is False
        assert "not a module" in result["error"]


class TestGetDependencies:
    """Tests for get_dependencies function."""

    def test_get_dependencies_success(self):
        """Test getting dependencies successfully."""
        from scitex.introspect import get_dependencies

        result = get_dependencies("scitex.introspect._resolve")
        assert result["success"] is True
        assert "dependencies" in result
        assert result["dependency_count"] >= 0

    def test_dependencies_non_recursive(self):
        """Test non-recursive dependencies."""
        from scitex.introspect import get_dependencies

        result = get_dependencies("scitex.introspect._resolve", recursive=False)
        assert result["success"] is True
        assert "tree" not in result

    @pytest.mark.skip(reason="Recursive deps can timeout due to stdlib scanning")
    def test_dependencies_recursive(self):
        """Test recursive dependencies."""
        from scitex.introspect import get_dependencies

        # Use json module which has minimal dependencies
        result = get_dependencies("json", recursive=True, max_depth=1)
        assert result["success"] is True
        # May have tree structure for recursive mode
        assert "dependencies" in result or "tree" in result
