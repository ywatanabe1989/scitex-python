#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__signature.py

"""Tests for scitex.introspect._signature module."""

import pytest


class TestQ:
    """Tests for q function."""

    def test_q_success(self):
        """Test getting signature successfully."""
        from scitex.introspect import q

        result = q("json.dumps")
        assert result["success"] is True
        assert result["name"] == "dumps"
        assert "signature" in result
        assert "parameters" in result

    def test_q_with_annotations(self):
        """Test signature includes type annotations."""
        from scitex.introspect import q

        result = q("json.dumps", include_annotations=True)
        assert result["success"] is True
        # json.dumps has annotations in newer Python
        assert "parameters" in result

    def test_q_without_defaults(self):
        """Test signature without default values."""
        from scitex.introspect import q

        result = q("json.dumps", include_defaults=False)
        assert result["success"] is True
        # Check no defaults in parameters
        for param in result["parameters"]:
            assert "default" not in param

    def test_q_class(self):
        """Test getting signature of a class (its __init__)."""
        from scitex.introspect import q

        result = q("pathlib.Path")
        assert result["success"] is True
        assert result["type_info"]["kind"] == "class"

    def test_q_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import q

        result = q("nonexistent.module")
        assert result["success"] is False
        assert "error" in result

    def test_q_builtin(self):
        """Test signature of builtin may fail gracefully."""
        from scitex.introspect import q

        result = q("len")
        # Builtins may not have introspectable signatures
        assert "success" in result
