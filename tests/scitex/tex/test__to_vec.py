#!/usr/bin/env python3
# Time-stamp: "2026-01-05 14:00:00 (ywatanabe)"
# File: ./tests/scitex/tex/test__to_vec.py

"""Tests for to_vec function that converts strings to LaTeX vector notation."""

import pytest

from scitex.tex import to_vec
from scitex.tex._to_vec import FALLBACK_AVAILABLE, safe_to_vec, vector_notation


class TestToVecWithoutFallback:
    """Tests for to_vec with enable_fallback=False (raw LaTeX output)."""

    def test_basic_vector(self):
        """Test basic vector conversion returns raw LaTeX."""
        result = to_vec("AB", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{AB}}"

    def test_single_character(self):
        """Test with single character vector."""
        result = to_vec("v", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{v}}"

        result = to_vec("x", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{x}}"

    def test_numeric_string(self):
        """Test with numeric strings."""
        result = to_vec("12", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{12}}"

        result = to_vec("0", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{0}}"

    def test_empty_string(self):
        """Test with empty string returns empty string."""
        result = to_vec("", enable_fallback=False)
        assert result == ""

    def test_special_characters(self):
        """Test with special characters that are valid in LaTeX."""
        result = to_vec("v_1", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{v_1}}"

        result = to_vec("PQ", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{PQ}}"

        result = to_vec("A'", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{A'}}"

    def test_long_string(self):
        """Test with longer vector names."""
        result = to_vec("velocity", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{velocity}}"

        result = to_vec("F_net", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{F_net}}"

    def test_unicode(self):
        """Test with unicode characters."""
        result = to_vec("αβ", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{αβ}}"

        result = to_vec("∇φ", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{∇φ}}"

    def test_spaces(self):
        """Test with strings containing spaces."""
        result = to_vec("A B", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{A B}}"

        result = to_vec(" CD ", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{ CD }}"

    def test_latex_special_chars(self):
        """Test with characters that have special meaning in LaTeX."""
        result = to_vec("$x$", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{$x$}}"

        result = to_vec("a&b", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{a&b}}"

        result = to_vec("x^2", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{x^2}}"

    def test_braces(self):
        """Test with braces in input."""
        result = to_vec("{AB}", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{{AB}}}"

        result = to_vec("a{b}c", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{a{b}c}}"

    def test_escape_sequences(self):
        """Test with escape sequences."""
        result = to_vec(r"\vec", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{\vec}}"

        result = to_vec("A\nB", enable_fallback=False)
        assert result == "\\overrightarrow{\\mathrm{A\nB}}"

    def test_mathematical_notation(self):
        """Test with common mathematical vector notations."""
        result = to_vec("r", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{r}}"

        result = to_vec("F", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{F}}"

        result = to_vec("r_0", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{r_0}}"

    def test_raw_string_output(self):
        """Test that output is valid raw string for LaTeX."""
        result = to_vec("PQ", enable_fallback=False)
        assert r"\overrightarrow" in result
        assert r"\mathrm" in result
        assert "{PQ}" in result

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        long_str = "A" * 100
        result = to_vec(long_str, enable_fallback=False)
        assert result == rf"\overrightarrow{{\mathrm{{{long_str}}}}}"

        result = to_vec("_-_-_", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{_-_-_}}"

        result = to_vec("123456", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{123456}}"


class TestToVecWithFallback:
    """Tests for to_vec with enable_fallback=True (default behavior)."""

    def test_basic_vector_fallback(self):
        """Test basic vector with fallback enabled returns unicode."""
        result = to_vec("AB")  # default enable_fallback=True
        # When fallback is available, returns unicode combining arrow
        if FALLBACK_AVAILABLE:
            # Could be mathtext processed or unicode fallback
            assert "AB" in result or "⃗" in result
        else:
            assert result == r"\overrightarrow{\mathrm{AB}}"

    def test_empty_string_fallback(self):
        """Test empty string returns empty with fallback enabled."""
        result = to_vec("")
        assert result == ""

    def test_fallback_preserves_content(self):
        """Test that the vector content is preserved in fallback output."""
        result = to_vec("XYZ")
        assert "X" in result or "XYZ" in result

    def test_unicode_fallback_strategy(self):
        """Test explicit unicode fallback strategy."""
        result = to_vec("AB", fallback_strategy="unicode")
        if FALLBACK_AVAILABLE:
            assert result == "AB⃗"
        else:
            assert result == r"\overrightarrow{\mathrm{AB}}"

    def test_plain_fallback_strategy(self):
        """Test explicit plain fallback strategy."""
        result = to_vec("AB", fallback_strategy="plain")
        if FALLBACK_AVAILABLE:
            assert result == "vec(AB)"
        else:
            assert result == r"\overrightarrow{\mathrm{AB}}"


class TestToVecConsistency:
    """Tests for consistent behavior of to_vec."""

    def test_repeated_calls_consistent(self):
        """Test that repeated calls produce consistent results."""
        input_str = "XY"
        result1 = to_vec(input_str, enable_fallback=False)
        result2 = to_vec(input_str, enable_fallback=False)
        assert result1 == result2 == r"\overrightarrow{\mathrm{XY}}"

    def test_type_preservation(self):
        """Test that function returns string type."""
        result = to_vec("AB", enable_fallback=False)
        assert isinstance(result, str)

        result = to_vec("")
        assert isinstance(result, str)

        result = to_vec("test")
        assert isinstance(result, str)

    def test_practical_usage(self):
        """Test practical usage in LaTeX documents."""
        vec_ab = to_vec("AB", enable_fallback=False)
        vec_bc = to_vec("BC", enable_fallback=False)

        assert vec_ab.startswith(r"\overrightarrow")
        assert vec_bc.startswith(r"\overrightarrow")

        latex_expr = f"{vec_ab} + {vec_bc}"
        assert r"\overrightarrow{\mathrm{AB}}" in latex_expr
        assert r"\overrightarrow{\mathrm{BC}}" in latex_expr


class TestSafeToVec:
    """Tests for safe_to_vec function."""

    def test_safe_to_vec_basic(self):
        """Test safe_to_vec with basic input."""
        result = safe_to_vec("AB")
        # safe_to_vec always has enable_fallback=True
        if FALLBACK_AVAILABLE:
            assert "AB" in result or "⃗" in result
        else:
            assert result == r"\overrightarrow{\mathrm{AB}}"

    def test_safe_to_vec_unicode_strategy(self):
        """Test safe_to_vec with unicode strategy."""
        result = safe_to_vec("AB", fallback_strategy="unicode")
        if FALLBACK_AVAILABLE:
            assert result == "AB⃗"
        else:
            assert result == r"\overrightarrow{\mathrm{AB}}"

    def test_safe_to_vec_plain_strategy(self):
        """Test safe_to_vec with plain strategy."""
        result = safe_to_vec("AB", fallback_strategy="plain")
        if FALLBACK_AVAILABLE:
            assert result == "vec(AB)"
        else:
            assert result == r"\overrightarrow{\mathrm{AB}}"

    def test_safe_to_vec_empty(self):
        """Test safe_to_vec with empty string."""
        result = safe_to_vec("")
        assert result == ""


class TestVectorNotationAlias:
    """Tests for vector_notation alias."""

    def test_vector_notation_is_to_vec(self):
        """Test that vector_notation is alias for to_vec."""
        assert vector_notation is to_vec

    def test_vector_notation_works(self):
        """Test that vector_notation produces same results."""
        result = vector_notation("AB", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{AB}}"


class TestFallbackAvailability:
    """Tests for fallback availability handling."""

    def test_fallback_available_flag(self):
        """Test FALLBACK_AVAILABLE is a boolean."""
        assert isinstance(FALLBACK_AVAILABLE, bool)

    def test_behavior_without_fallback_module(self):
        """Test behavior when fallback is disabled."""
        result = to_vec("test", enable_fallback=False)
        assert result == r"\overrightarrow{\mathrm{test}}"

if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
