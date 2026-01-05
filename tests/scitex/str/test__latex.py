#!/usr/bin/env python3
# Time-stamp: "2025-06-05 14:40:00 (ywatanabe)"
# File: ./tests/scitex/str/test__latex.py

"""Tests for LaTeX string formatting functions."""

import pytest


class TestToLatexStyle:
    """Tests for to_latex_style function."""

    def test_basic_string(self):
        """Test basic string conversion."""
        from scitex.str._latex import to_latex_style

        assert to_latex_style("aaa") == "$aaa$"
        assert to_latex_style("x^2") == "$x^2$"
        assert to_latex_style("alpha") == "$alpha$"

    def test_numbers(self):
        """Test number conversion."""
        from scitex.str._latex import to_latex_style

        assert to_latex_style(123) == "$123$"
        assert to_latex_style(3.14) == "$3.14$"
        assert to_latex_style(-5) == "$-5$"
        assert to_latex_style(0) == "$0$"

    def test_already_formatted(self):
        """Test that already formatted strings are not double-wrapped."""
        from scitex.str._latex import to_latex_style

        assert to_latex_style("$x^2$") == "$x^2$"
        assert to_latex_style("$\\alpha$") == "$\\alpha$"
        assert to_latex_style("$123$") == "$123$"

    def test_empty_string(self):
        """Test empty string returns empty string."""
        from scitex.str._latex import to_latex_style

        assert to_latex_style("") == ""

    def test_special_chars(self):
        """Test LaTeX special characters."""
        from scitex.str._latex import to_latex_style

        assert to_latex_style("\\frac{1}{2}") == "$\\frac{1}{2}$"
        assert to_latex_style("\\sum_{i=1}^n") == "$\\sum_{i=1}^n$"
        assert to_latex_style("\\alpha + \\beta") == "$\\alpha + \\beta$"

    def test_whitespace(self):
        """Test whitespace handling."""
        from scitex.str._latex import to_latex_style

        assert to_latex_style("x y") == "$x y$"
        assert to_latex_style("x\ty") == "$x\ty$"

    def test_idempotent(self):
        """Test that applying twice is idempotent."""
        from scitex.str._latex import to_latex_style

        text = "formula"
        once = to_latex_style(text)
        twice = to_latex_style(once)
        assert once == twice == "$formula$"


class TestAddHatInLatexStyle:
    """Tests for add_hat_in_latex_style function."""

    def test_basic_string(self):
        """Test basic hat addition."""
        from scitex.str._latex import add_hat_in_latex_style

        assert add_hat_in_latex_style("aaa") == "$\\hat{aaa}$"
        assert add_hat_in_latex_style("x") == "$\\hat{x}$"
        assert add_hat_in_latex_style("beta") == "$\\hat{beta}$"

    def test_numbers(self):
        """Test hat with numbers."""
        from scitex.str._latex import add_hat_in_latex_style

        assert add_hat_in_latex_style(1) == "$\\hat{1}$"
        assert add_hat_in_latex_style(3.14) == "$\\hat{3.14}$"
        assert add_hat_in_latex_style(0) == "$\\hat{0}$"

    def test_complex_expressions(self):
        """Test hat with complex expressions."""
        from scitex.str._latex import add_hat_in_latex_style

        assert add_hat_in_latex_style("x^2") == "$\\hat{x^2}$"
        assert add_hat_in_latex_style("\\alpha") == "$\\hat{\\alpha}$"

    def test_empty_string(self):
        """Test empty string returns empty string."""
        from scitex.str._latex import add_hat_in_latex_style

        assert add_hat_in_latex_style("") == ""

    def test_whitespace(self):
        """Test whitespace in hat."""
        from scitex.str._latex import add_hat_in_latex_style

        assert add_hat_in_latex_style("x y") == "$\\hat{x y}$"


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases."""

    def test_latex_style_alias(self):
        """Test latex_style is alias for to_latex_style."""
        from scitex.str._latex import latex_style, to_latex_style

        assert latex_style is to_latex_style
        assert latex_style("test") == "$test$"

    def test_hat_latex_style_alias(self):
        """Test hat_latex_style is alias for add_hat_in_latex_style."""
        from scitex.str._latex import add_hat_in_latex_style, hat_latex_style

        assert hat_latex_style is add_hat_in_latex_style
        assert hat_latex_style("test") == "$\\hat{test}$"

    def test_safe_versions_available(self):
        """Test safe versions are available and work identically."""
        from scitex.str._latex import (
            add_hat_in_latex_style,
            safe_add_hat_in_latex_style,
            safe_to_latex_style,
            to_latex_style,
        )

        assert safe_to_latex_style("x") == to_latex_style("x")
        assert safe_add_hat_in_latex_style("x") == add_hat_in_latex_style("x")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_none_like_values(self):
        """Test None-like values."""
        from scitex.str._latex import add_hat_in_latex_style, to_latex_style

        # Empty string
        assert to_latex_style("") == ""
        assert add_hat_in_latex_style("") == ""

        # Zero should work (it's falsy but valid)
        assert to_latex_style(0) == "$0$"
        assert add_hat_in_latex_style(0) == "$\\hat{0}$"

    def test_single_dollar_sign(self):
        """Test strings with single dollar signs."""
        from scitex.str._latex import to_latex_style

        # Not already formatted (only one $)
        assert to_latex_style("$x") == "$$x$"
        assert to_latex_style("x$") == "$x$$"

    def test_unicode_content(self):
        """Test unicode content."""
        from scitex.str._latex import add_hat_in_latex_style, to_latex_style

        assert to_latex_style("α") == "$α$"
        assert to_latex_style("日本語") == "$日本語$"
        assert add_hat_in_latex_style("α") == "$\\hat{α}$"

    def test_long_expressions(self):
        """Test long mathematical expressions."""
        from scitex.str._latex import to_latex_style

        expr = "\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}"
        assert to_latex_style(expr) == f"${expr}$"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
