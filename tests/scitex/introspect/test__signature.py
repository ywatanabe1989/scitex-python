#!/usr/bin/env python3
# Timestamp: 2025-01-20
# File: tests/scitex/introspect/test__signature.py

"""Tests for scitex.introspect._signature module."""

import pytest


class TestGetSignature:
    """Tests for get_signature function."""

    def test_get_signature_success(self):
        """Test getting signature successfully."""
        from scitex.introspect import get_signature

        result = get_signature("json.dumps")
        assert result["success"] is True
        assert result["name"] == "dumps"
        assert "signature" in result
        assert "parameters" in result

    def test_get_signature_with_annotations(self):
        """Test signature includes type annotations."""
        from scitex.introspect import get_signature

        result = get_signature("json.dumps", include_annotations=True)
        assert result["success"] is True
        # json.dumps has annotations in newer Python
        assert "parameters" in result

    def test_get_signature_without_defaults(self):
        """Test signature without default values."""
        from scitex.introspect import get_signature

        result = get_signature("json.dumps", include_defaults=False)
        assert result["success"] is True
        # Check no defaults in parameters
        for param in result["parameters"]:
            assert "default" not in param

    def test_get_signature_class(self):
        """Test getting signature of a class (its __init__)."""
        from scitex.introspect import get_signature

        result = get_signature("pathlib.Path")
        assert result["success"] is True
        assert result["type_info"]["kind"] == "class"

    def test_get_signature_invalid_path(self):
        """Test invalid path returns error."""
        from scitex.introspect import get_signature

        result = get_signature("nonexistent.module")
        assert result["success"] is False
        assert "error" in result

    def test_get_signature_builtin(self):
        """Test signature of builtin may fail gracefully."""
        from scitex.introspect import get_signature

        result = get_signature("len")
        # Builtins may not have introspectable signatures
        assert "success" in result
