#!/usr/bin/env python3
"""Tests for scitex.dev.plt.mpl.get_signatures_details module."""

import inspect
import os

import pytest

from scitex.dev.plt.mpl.get_signatures_details import (
    MANUAL_ARGS_PATTERNS,
    extract_args_from_docstring,
    format_signature,
    get_default_value,
    get_typehint_str,
    parse_args_pattern,
    parse_parameter_types,
)


class TestParseParameterTypes:
    """Tests for parse_parameter_types function."""

    def test_none_docstring_returns_empty_dict(self):
        """Test that None docstring returns empty dict."""
        result = parse_parameter_types(None)
        assert result == {}

    def test_empty_docstring_returns_empty_dict(self):
        """Test that empty docstring returns empty dict."""
        result = parse_parameter_types("")
        assert result == {}

    def test_parses_numpy_style_parameters(self):
        """Test parsing NumPy-style Parameters section."""
        # Note: The regex expects specific formatting with newlines between params
        docstring = """Short description.

Parameters
----------
x : array-like
    The x values.

y : float
    The y value.
"""
        result = parse_parameter_types(docstring)
        assert "x" in result
        assert "array-like" in result["x"]

    def test_parses_multiple_names(self):
        """Test parsing parameters with multiple names like 'x, y'."""
        docstring = """
        Parameters
        ----------
        x, y : array-like
            The coordinate values.
        """
        result = parse_parameter_types(docstring)
        assert "x" in result
        assert "y" in result

    def test_strips_optional_from_type(self):
        """Test that 'optional' is stripped from type string."""
        docstring = """
        Parameters
        ----------
        x : float, optional
            The value.
        """
        result = parse_parameter_types(docstring)
        assert "optional" not in result.get("x", "optional")


class TestParseArgsPattern:
    """Tests for parse_args_pattern function."""

    def test_empty_string_returns_empty_list(self):
        """Test that empty string returns empty list."""
        result = parse_args_pattern("", {})
        assert result == []

    def test_parses_simple_args(self):
        """Test parsing simple arg pattern."""
        result = parse_args_pattern("x, y", {})
        assert len(result) == 2
        assert result[0]["name"] == "x"
        assert result[1]["name"] == "y"

    def test_parses_optional_args(self):
        """Test parsing optional args in brackets."""
        result = parse_args_pattern("[x], y", {})
        assert len(result) == 2
        assert result[0]["optional"] is True
        assert result[1]["optional"] is False

    def test_uses_param_types(self):
        """Test that param_types are used for type lookup."""
        param_types = {"x": "float", "y": "int"}
        result = parse_args_pattern("x, y", param_types)
        assert result[0]["type"] == "float"
        assert result[1]["type"] == "int"


class TestExtractArgsFromDocstring:
    """Tests for extract_args_from_docstring function."""

    def test_none_docstring_returns_empty_list(self):
        """Test that None docstring returns empty list."""
        result = extract_args_from_docstring(None)
        assert result == []

    def test_uses_manual_pattern_when_available(self):
        """Test that manual patterns are used when available."""
        # 'fill' is in MANUAL_ARGS_PATTERNS
        result = extract_args_from_docstring("Some docstring", "fill")
        assert len(result) > 0

    def test_manual_patterns_exist_for_known_functions(self):
        """Test that MANUAL_ARGS_PATTERNS has expected functions."""
        expected_funcs = ["fill", "stem", "quiver", "barbs"]
        for func in expected_funcs:
            assert func in MANUAL_ARGS_PATTERNS


class TestGetTypehintStr:
    """Tests for get_typehint_str function."""

    def test_empty_annotation_returns_none(self):
        """Test that empty annotation returns None."""
        result = get_typehint_str(inspect.Parameter.empty)
        assert result is None

    def test_returns_type_name(self):
        """Test that types return their __name__."""
        result = get_typehint_str(str)
        assert result == "str"


class TestGetDefaultValue:
    """Tests for get_default_value function."""

    def test_empty_returns_none(self):
        """Test that empty parameter returns None."""
        result = get_default_value(inspect.Parameter.empty)
        assert result is None

    def test_serializable_value(self):
        """Test that serializable values are returned as-is."""
        assert get_default_value(42) == 42
        assert get_default_value("text") == "text"


class TestFormatSignature:
    """Tests for format_signature function."""

    def test_error_case(self):
        """Test formatting when there's an error."""
        info = {"error": "Cannot inspect"}
        result = format_signature("my_func", info)
        assert "Cannot inspect" in result

    def test_args_and_kwargs_formatting(self):
        """Test formatting with args and kwargs."""
        info = {
            "args": [{"name": "x", "type": "int"}],
            "kwargs": [{"name": "y", "type": "str", "default": "test"}],
        }
        result = format_signature("my_func", info)
        assert "x" in result
        assert "y=" in result


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
