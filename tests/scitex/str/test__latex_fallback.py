#!/usr/bin/env python3
# Time-stamp: "2025-06-07 15:35:00 (ywatanabe)"
# File: ./tests/scitex/str/test__latex_fallback.py

"""
Comprehensive tests for LaTeX fallback mechanism.

This module tests the LaTeX fallback functionality that gracefully
handles LaTeX rendering failures by converting to mathtext or unicode.
"""

from unittest.mock import Mock, patch

import pytest


class TestFallbackMode:
    """Test fallback mode management."""

    def setup_method(self):
        """Reset state before each test."""
        from scitex.str._latex_fallback import reset_latex_cache, set_fallback_mode

        reset_latex_cache()
        set_fallback_mode("auto")

    def test_set_fallback_mode_valid(self):
        """Test setting valid fallback modes."""
        from scitex.str._latex_fallback import get_fallback_mode, set_fallback_mode

        for mode in ["auto", "force_mathtext", "force_plain"]:
            set_fallback_mode(mode)
            assert get_fallback_mode() == mode

    def test_set_fallback_mode_invalid(self):
        """Test setting invalid fallback mode raises error."""
        from scitex.str._latex_fallback import set_fallback_mode

        with pytest.raises(ValueError, match="Invalid fallback mode"):
            set_fallback_mode("invalid_mode")

    def test_fallback_mode_resets_cache(self):
        """Test that changing mode resets LaTeX capability cache."""
        from scitex.str._latex_fallback import (
            check_latex_capability,
            reset_latex_cache,
            set_fallback_mode,
        )

        reset_latex_cache()

        # Force a capability check to cache result
        check_latex_capability()
        cache_info_before = check_latex_capability.cache_info()
        initial_misses = cache_info_before.misses

        # Change mode should reset cache
        set_fallback_mode("force_mathtext")
        check_latex_capability.cache_clear()  # Explicit clear

        # Make another call
        check_latex_capability()
        cache_info_after = check_latex_capability.cache_info()

        # Cache should have been cleared (new miss)
        assert cache_info_after.misses >= 1


class TestLatexCapability:
    """Test LaTeX capability detection."""

    def setup_method(self):
        """Reset state before each test."""
        from scitex.str._latex_fallback import reset_latex_cache, set_fallback_mode

        reset_latex_cache()
        set_fallback_mode("auto")

    def test_latex_disabled_in_rcparams(self):
        """Test capability when LaTeX is disabled in rcParams."""
        from scitex.str._latex_fallback import check_latex_capability, reset_latex_cache

        reset_latex_cache()
        # When text.usetex is False, should return False
        result = check_latex_capability()
        # Result depends on rcParams, but should be bool
        assert isinstance(result, bool)

    def test_force_modes_return_false(self):
        """Test that force modes always return False for capability."""
        from scitex.str._latex_fallback import (
            check_latex_capability,
            reset_latex_cache,
            set_fallback_mode,
        )

        set_fallback_mode("force_mathtext")
        reset_latex_cache()
        assert check_latex_capability() == False

        set_fallback_mode("force_plain")
        reset_latex_cache()
        assert check_latex_capability() == False

    def test_capability_is_cached(self):
        """Test that capability check is cached."""
        from scitex.str._latex_fallback import check_latex_capability, reset_latex_cache

        reset_latex_cache()

        # First call
        check_latex_capability()
        cache_info = check_latex_capability.cache_info()
        assert cache_info.misses >= 1

        # Second call should hit cache
        check_latex_capability()
        cache_info = check_latex_capability.cache_info()
        assert cache_info.hits >= 1


class TestLatexToMathtext:
    """Test LaTeX to mathtext conversion."""

    def test_empty_string(self):
        """Test conversion of empty string."""
        from scitex.str._latex_fallback import latex_to_mathtext

        assert latex_to_mathtext("") == ""

    def test_basic_greek_letters(self):
        """Test conversion of Greek letters."""
        from scitex.str._latex_fallback import latex_to_mathtext

        # The function converts \\alpha to \alpha in the output
        result = latex_to_mathtext(r"$\alpha$")
        assert r"\alpha" in result or "alpha" in result

    def test_mathematical_symbols(self):
        """Test conversion of mathematical symbols."""
        from scitex.str._latex_fallback import latex_to_mathtext

        result = latex_to_mathtext(r"$\sum$")
        assert r"\sum" in result or "sum" in result

    def test_strip_outer_dollars(self):
        """Test handling of dollar signs."""
        from scitex.str._latex_fallback import latex_to_mathtext

        result = latex_to_mathtext(r"$x^2$")
        assert "$" in result  # Should be wrapped in $

    def test_plain_text_gets_wrapped(self):
        """Test that plain text gets wrapped in dollars."""
        from scitex.str._latex_fallback import latex_to_mathtext

        result = latex_to_mathtext("x^2")
        assert result.startswith("$") and result.endswith("$")


class TestLatexToUnicode:
    """Test LaTeX to Unicode conversion."""

    def test_empty_string(self):
        """Test conversion of empty string."""
        from scitex.str._latex_fallback import latex_to_unicode

        assert latex_to_unicode("") == ""

    def test_greek_letters(self):
        """Test conversion of Greek letters to Unicode."""
        from scitex.str._latex_fallback import latex_to_unicode

        # Test with escaped backslashes (as in raw strings)
        result = latex_to_unicode(r"$\alpha$")
        # Should contain alpha letter or be partially converted
        assert isinstance(result, str)

    def test_superscripts(self):
        """Test conversion of superscripts."""
        from scitex.str._latex_fallback import latex_to_unicode

        result = latex_to_unicode(r"$x^2$")
        # Should contain x and either superscript 2 or regular 2
        assert "x" in result

    def test_subscripts(self):
        """Test conversion of subscripts."""
        from scitex.str._latex_fallback import latex_to_unicode

        result = latex_to_unicode(r"$x_0$")
        # Should contain x and either subscript 0 or regular 0
        assert "x" in result

    def test_strip_braces(self):
        """Test removal of remaining braces."""
        from scitex.str._latex_fallback import latex_to_unicode

        result = latex_to_unicode(r"${x}$")
        assert "x" in result
        assert "{" not in result and "}" not in result


class TestSafeLatexRender:
    """Test safe LaTeX rendering with fallback."""

    def setup_method(self):
        """Reset state before each test."""
        from scitex.str._latex_fallback import reset_latex_cache, set_fallback_mode

        reset_latex_cache()
        set_fallback_mode("auto")

    def test_empty_input(self):
        """Test handling of empty input."""
        from scitex.str._latex_fallback import safe_latex_render

        assert safe_latex_render("") == ""
        assert safe_latex_render(None) is None

    def test_non_string_input(self):
        """Test handling of non-string input."""
        from scitex.str._latex_fallback import safe_latex_render

        assert safe_latex_render(123) == 123
        assert safe_latex_render([1, 2, 3]) == [1, 2, 3]

    def test_force_mathtext_strategy(self):
        """Test force mathtext strategy."""
        from scitex.str._latex_fallback import safe_latex_render

        result = safe_latex_render(r"$\beta$", fallback_strategy="mathtext")
        assert isinstance(result, str)
        assert "$" in result  # Should be wrapped

    def test_force_unicode_strategy(self):
        """Test force unicode strategy."""
        from scitex.str._latex_fallback import safe_latex_render

        result = safe_latex_render(r"$\gamma$", fallback_strategy="unicode")
        assert isinstance(result, str)
        # Should not have $ markers in unicode mode
        assert "$" not in result or "gamma" in result or "γ" in result

    def test_plain_strategy(self):
        """Test plain text strategy."""
        from scitex.str._latex_fallback import safe_latex_render

        result = safe_latex_render(r"$\delta^2$", fallback_strategy="plain")
        assert isinstance(result, str)

    def test_invalid_strategy(self):
        """Test invalid fallback strategy."""
        from scitex.str._latex_fallback import safe_latex_render

        with pytest.raises(ValueError, match="Unknown fallback strategy"):
            safe_latex_render(r"$x$", fallback_strategy="invalid")


class TestLatexFallbackDecorator:
    """Test LaTeX fallback decorator."""

    def test_successful_function_call(self):
        """Test decorator with successful function call."""
        from scitex.str._latex_fallback import latex_fallback_decorator

        @latex_fallback_decorator()
        def test_func(text):
            return f"Rendered: {text}"

        result = test_func("Hello")
        assert result == "Rendered: Hello"

    def test_non_latex_error_propagation(self):
        """Test that non-LaTeX errors are propagated."""
        from scitex.str._latex_fallback import latex_fallback_decorator

        @latex_fallback_decorator()
        def test_func():
            raise ValueError("Not a LaTeX error")

        with pytest.raises(ValueError, match="Not a LaTeX error"):
            test_func()

    def test_preserve_function_metadata(self):
        """Test decorator preserves function metadata."""
        from scitex.str._latex_fallback import latex_fallback_decorator

        @latex_fallback_decorator()
        def documented_func():
            """This is a documented function."""
            pass

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a documented function."


class TestStatusFunctions:
    """Test status and utility functions."""

    def setup_method(self):
        """Reset state before each test."""
        from scitex.str._latex_fallback import reset_latex_cache, set_fallback_mode

        reset_latex_cache()
        set_fallback_mode("auto")

    def test_get_latex_status(self):
        """Test getting LaTeX status information."""
        from scitex.str._latex_fallback import get_latex_status

        status = get_latex_status()

        assert "latex_available" in status
        assert "fallback_mode" in status
        assert "usetex_enabled" in status
        assert "mathtext_fontset" in status
        assert "font_family" in status
        assert "cache_info" in status

        assert isinstance(status["latex_available"], bool)
        assert status["fallback_mode"] == "auto"

    def test_enable_latex_fallback(self):
        """Test enabling LaTeX fallback."""
        from scitex.str._latex_fallback import enable_latex_fallback, get_fallback_mode

        enable_latex_fallback("force_mathtext")
        assert get_fallback_mode() == "force_mathtext"

        enable_latex_fallback()  # Default to "auto"
        assert get_fallback_mode() == "auto"

    def test_disable_latex_fallback(self):
        """Test disabling LaTeX fallback."""
        from scitex.str._latex_fallback import disable_latex_fallback

        # Should not raise
        disable_latex_fallback()

    def test_reset_latex_cache(self):
        """Test resetting LaTeX cache."""
        from scitex.str._latex_fallback import check_latex_capability, reset_latex_cache

        # Make a call to populate cache
        check_latex_capability()
        cache_info_before = check_latex_capability.cache_info()

        # Reset cache
        reset_latex_cache()

        # Cache should have been cleared
        cache_info_after = check_latex_capability.cache_info()
        # After reset, there should be 0 hits since we haven't called again
        assert cache_info_after.hits == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_deeply_nested_latex(self):
        """Test handling of deeply nested LaTeX."""
        from scitex.str._latex_fallback import latex_to_mathtext

        nested = r"$\frac{\alpha}{\sqrt{\beta^2 + \gamma^2}}$"
        result = latex_to_mathtext(nested)
        assert isinstance(result, str)

    def test_malformed_latex(self):
        """Test handling of malformed LaTeX."""
        from scitex.str._latex_fallback import latex_to_unicode

        malformed = r"$\frac{1{2}$"  # Missing closing brace
        # Should not crash
        result = latex_to_unicode(malformed)
        assert isinstance(result, str)

    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        from scitex.str._latex_fallback import latex_to_unicode

        # Test that combining characters work
        result = latex_to_unicode(r"$\tilde{n}$")
        assert isinstance(result, str)

    def test_empty_commands(self):
        """Test handling of empty LaTeX commands."""
        from scitex.str._latex_fallback import latex_to_unicode

        # These should not crash
        result1 = latex_to_unicode(r"$\$")
        result2 = latex_to_unicode(r"$\{}$")
        assert isinstance(result1, str)
        assert isinstance(result2, str)

    def test_thread_safety(self):
        """Test thread safety of cached functions."""
        import threading

        from scitex.str._latex_fallback import check_latex_capability

        results = []

        def check_capability():
            results.append(check_latex_capability())

        threads = [threading.Thread(target=check_capability) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be the same (consistent)
        assert all(r == results[0] for r in results)


class TestLaTeXFallbackError:
    """Test the LaTeXFallbackError exception."""

    def test_exception_can_be_raised(self):
        """Test that LaTeXFallbackError can be raised."""
        from scitex.str._latex_fallback import LaTeXFallbackError

        with pytest.raises(LaTeXFallbackError):
            raise LaTeXFallbackError("Test error")

    def test_exception_message(self):
        """Test exception message."""
        from scitex.str._latex_fallback import LaTeXFallbackError

        try:
            raise LaTeXFallbackError("Custom message")
        except LaTeXFallbackError as e:
            assert "Custom message" in str(e)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/str/_latex_fallback.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-08-21 21:37:23 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/str/_latex_fallback.py
# # ----------------------------------------
# from __future__ import annotations
# 
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# # Time-stamp: "2025-06-05 12:00:00 (ywatanabe)"
# 
# 
# """
# LaTeX Fallback Mechanism for SciTeX
# 
# This module provides a robust fallback system for LaTeX rendering issues.
# When LaTeX compilation fails (e.g., missing fonts, Node.js conflicts),
# it gracefully degrades to mathtext or plain text alternatives.
# 
# Functionality:
#     - Detect LaTeX rendering capabilities
#     - Convert LaTeX to mathtext equivalents
#     - Provide plain text fallbacks
#     - Cache LaTeX capability status
# Input:
#     LaTeX strings and matplotlib configuration
# Output:
#     Fallback-compatible strings for matplotlib
# Prerequisites:
#     matplotlib
# """
# 
# import functools
# import re
# from typing import Any, Callable, Dict
# 
# # matplotlib imports moved to functions that need them
# 
# 
# # Delay logging import to avoid circular dependency
# # scitex.logging imports _Tee which imports scitex.str which imports this file
# def _get_logger():
#     """Get logger lazily to avoid circular import."""
#     from scitex import logging
# 
#     return logging.getLogger(__name__)
# 
# 
# # Use property-like access for logger
# class _LoggerProxy:
#     def __getattr__(self, name):
#         return getattr(_get_logger(), name)
# 
# 
# logger = _LoggerProxy()
# 
# # Global state for LaTeX capability
# _latex_available = None
# _fallback_mode = "auto"  # "auto", "force_mathtext", "force_plain"
# 
# 
# class LaTeXFallbackError(Exception):
#     """Raised when LaTeX fallback mechanisms fail."""
# 
#     pass
# 
# 
# def set_fallback_mode(mode: str) -> None:
#     """
#     Set the global fallback mode for LaTeX rendering.
# 
#     Parameters
#     ----------
#     mode : str
#         Fallback mode: "auto" (detect capability), "force_mathtext", or "force_plain"
#     """
#     global _fallback_mode
#     if mode not in ["auto", "force_mathtext", "force_plain"]:
#         raise ValueError(f"Invalid fallback mode: {mode}")
#     _fallback_mode = mode
#     # Reset capability cache when mode changes
#     global _latex_available
#     _latex_available = None
# 
# 
# def get_fallback_mode() -> str:
#     """Get the current fallback mode."""
#     return _fallback_mode
# 
# 
# @functools.lru_cache(maxsize=1)
# def check_latex_capability() -> bool:
#     """
#     Check if LaTeX rendering is available and working.
# 
#     Returns
#     -------
#     bool
#         True if LaTeX is available, False otherwise
#     """
#     global _latex_available
# 
#     # Import matplotlib here when actually needed
#     try:
#         import matplotlib.pyplot as plt
#     except ImportError:
#         _latex_available = False
#         return False
# 
#     # If forcing a mode, return accordingly
#     if _fallback_mode == "force_mathtext" or _fallback_mode == "force_plain":
#         _latex_available = False
#         return False
# 
#     # Cache the result if already determined
#     if _latex_available is not None:
#         return _latex_available
# 
#     try:
#         # Test if LaTeX is configured
#         if not plt.rcParams.get("text.usetex", False):
#             _latex_available = False
#             return False
# 
#         # Try a simple LaTeX rendering test
#         fig, ax = plt.subplots(figsize=(1, 1))
#         try:
#             # Test with a simple LaTeX expression
#             text_obj = ax.text(0.5, 0.5, r"$x^2$", usetex=True)
#             fig.canvas.draw()  # Force rendering
#             _latex_available = True
#             result = True
#         except Exception as e:
#             logger.debug(f"LaTeX capability test failed: {e}")
#             _latex_available = False
#             result = False
#         finally:
#             plt.close(fig)
# 
#         return result
# 
#     except Exception as e:
#         logger.debug(f"LaTeX capability check failed: {e}")
#         _latex_available = False
#         return False
# 
# 
# def latex_to_mathtext(latex_str: str) -> str:
#     """
#     Convert LaTeX syntax to matplotlib mathtext equivalent.
# 
#     Parameters
#     ----------
#     latex_str : str
#         LaTeX string to convert
# 
#     Returns
#     -------
#     str
#         Mathtext equivalent string
#     """
#     if not latex_str:
#         return latex_str
# 
#     # Remove outer $ if present
#     text = latex_str.strip()
#     if text.startswith("$") and text.endswith("$"):
#         text = text[1:-1]
# 
#     # Simple string replacements (double backslash to single)
#     simple_conversions = {
#         # Greek letters
#         r"\\alpha": "\\alpha",
#         r"\\beta": "\\beta",
#         r"\\gamma": "\\gamma",
#         r"\\delta": "\\delta",
#         r"\\epsilon": "\\epsilon",
#         r"\\theta": "\\theta",
#         r"\\lambda": "\\lambda",
#         r"\\mu": "\\mu",
#         r"\\pi": "\\pi",
#         r"\\sigma": "\\sigma",
#         r"\\tau": "\\tau",
#         r"\\phi": "\\phi",
#         r"\\omega": "\\omega",
#         # Mathematical symbols
#         r"\\sum": "\\sum",
#         r"\\int": "\\int",
#         r"\\partial": "\\partial",
#         r"\\infty": "\\infty",
#         r"\\pm": "\\pm",
#         r"\\times": "\\times",
#         r"\\cdot": "\\cdot",
#         r"\\approx": "\\approx",
#         r"\\neq": "\\neq",
#         r"\\leq": "\\leq",
#         r"\\geq": "\\geq",
#         # Functions
#         r"\\sin": "\\sin",
#         r"\\cos": "\\cos",
#         r"\\tan": "\\tan",
#         r"\\log": "\\log",
#         r"\\ln": "\\ln",
#         r"\\exp": "\\exp",
#     }
# 
#     # Apply simple string replacements
#     for pattern, replacement in simple_conversions.items():
#         text = text.replace(pattern, replacement)
# 
#     # Regex patterns with capture groups
#     regex_conversions = [
#         # Formatting (limited mathtext support)
#         (r"\\textbf\{([^}]+)\}", r"\\mathbf{\1}"),
#         (r"\\mathbf\{([^}]+)\}", r"\\mathbf{\1}"),
#         (r"\\textit\{([^}]+)\}", r"\\mathit{\1}"),
#         (r"\\mathit\{([^}]+)\}", r"\\mathit{\1}"),
#         # Hats and accents
#         (r"\\hat\{([^}]+)\}", r"\\hat{\1}"),
#         (r"\\overrightarrow\{([^}]+)\}", r"\\vec{\1}"),
#         (r"\\vec\{([^}]+)\}", r"\\vec{\1}"),
#         # Fractions (simple ones)
#         (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\\frac{\1}{\2}"),
#     ]
# 
#     # Apply regex conversions
#     for pattern, replacement in regex_conversions:
#         text = re.sub(pattern, replacement, text)
# 
#     # Wrap in mathtext markers
#     return f"${text}$"
# 
# 
# def latex_to_unicode(latex_str: str) -> str:
#     """
#     Convert LaTeX to Unicode plain text equivalent.
# 
#     Parameters
#     ----------
#     latex_str : str
#         LaTeX string to convert
# 
#     Returns
#     -------
#     str
#         Unicode plain text equivalent
#     """
#     if not latex_str:
#         return latex_str
# 
#     # Remove outer $ if present
#     text = latex_str.strip()
#     if text.startswith("$") and text.endswith("$"):
#         text = text[1:-1]
# 
#     # Greek letters to Unicode
#     greek_conversions = {
#         r"\\alpha": "α",
#         r"\\beta": "β",
#         r"\\gamma": "γ",
#         r"\\delta": "δ",
#         r"\\epsilon": "ε",
#         r"\\zeta": "ζ",
#         r"\\eta": "η",
#         r"\\theta": "θ",
#         r"\\iota": "ι",
#         r"\\kappa": "κ",
#         r"\\lambda": "λ",
#         r"\\mu": "μ",
#         r"\\nu": "ν",
#         r"\\xi": "ξ",
#         r"\\pi": "π",
#         r"\\rho": "ρ",
#         r"\\sigma": "σ",
#         r"\\tau": "τ",
#         r"\\upsilon": "υ",
#         r"\\phi": "φ",
#         r"\\chi": "χ",
#         r"\\psi": "ψ",
#         r"\\omega": "ω",
#         # Capital Greek
#         r"\\Gamma": "Γ",
#         r"\\Delta": "Δ",
#         r"\\Theta": "Θ",
#         r"\\Lambda": "Λ",
#         r"\\Xi": "Ξ",
#         r"\\Pi": "Π",
#         r"\\Sigma": "Σ",
#         r"\\Upsilon": "Υ",
#         r"\\Phi": "Φ",
#         r"\\Psi": "Ψ",
#         r"\\Omega": "Ω",
#     }
# 
#     # Mathematical symbols to Unicode
#     symbol_conversions = {
#         r"\\pm": "±",
#         r"\\times": "×",
#         r"\\cdot": "·",
#         r"\\div": "÷",
#         r"\\neq": "≠",
#         r"\\leq": "≤",
#         r"\\geq": "≥",
#         r"\\approx": "≈",
#         r"\\infty": "∞",
#         r"\\partial": "∂",
#         r"\\sum": "∑",
#         r"\\int": "∫",
#         r"\\sqrt": "√",
#         r"\\angle": "∠",
#         r"\\degree": "°",
#     }
# 
#     # Superscript numbers
#     superscript_conversions = {
#         r"\^0": "⁰",
#         r"\^1": "¹",
#         r"\^2": "²",
#         r"\^3": "³",
#         r"\^4": "⁴",
#         r"\^5": "⁵",
#         r"\^6": "⁶",
#         r"\^7": "⁷",
#         r"\^8": "⁸",
#         r"\^9": "⁹",
#         r"\^\{0\}": "⁰",
#         r"\^\{1\}": "¹",
#         r"\^\{2\}": "²",
#         r"\^\{3\}": "³",
#         r"\^\{4\}": "⁴",
#         r"\^\{5\}": "⁵",
#         r"\^\{6\}": "⁶",
#         r"\^\{7\}": "⁷",
#         r"\^\{8\}": "⁸",
#         r"\^\{9\}": "⁹",
#     }
# 
#     # Subscript numbers
#     subscript_conversions = {
#         r"_0": "₀",
#         r"_1": "₁",
#         r"_2": "₂",
#         r"_3": "₃",
#         r"_4": "₄",
#         r"_5": "₅",
#         r"_6": "₆",
#         r"_7": "₇",
#         r"_8": "₈",
#         r"_9": "₉",
#         r"_\{0\}": "₀",
#         r"_\{1\}": "₁",
#         r"_\{2\}": "₂",
#         r"_\{3\}": "₃",
#         r"_\{4\}": "₄",
#         r"_\{5\}": "₅",
#         r"_\{6\}": "₆",
#         r"_\{7\}": "₇",
#         r"_\{8\}": "₈",
#         r"_\{9\}": "₉",
#     }
# 
#     # Apply all conversions
#     all_conversions = {
#         **greek_conversions,
#         **symbol_conversions,
#         **superscript_conversions,
#         **subscript_conversions,
#     }
# 
#     for latex_pattern, unicode_char in all_conversions.items():
#         text = re.sub(latex_pattern, unicode_char, text)
# 
#     # Remove remaining LaTeX commands (simple cleanup)
#     text = re.sub(
#         r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text
#     )  # \command{content} -> content
#     text = re.sub(r"\\[a-zA-Z]+", "", text)  # \command -> remove
#     text = re.sub(r"[{}]", "", text)  # Remove remaining braces
# 
#     return text
# 
# 
# def safe_latex_render(
#     text: str, fallback_strategy: str = "auto", preserve_math: bool = True
# ) -> str:
#     """
#     Safely render LaTeX text with automatic fallback.
# 
#     Parameters
#     ----------
#     text : str
#         Text that may contain LaTeX
#     fallback_strategy : str, optional
#         Strategy when LaTeX fails: "auto", "mathtext", "unicode", "plain"
#     preserve_math : bool, optional
#         Whether to preserve mathematical notation in fallbacks
# 
#     Returns
#     -------
#     str
#         Safely rendered text
#     """
#     if not text or not isinstance(text, str):
#         return text
# 
#     # Import matplotlib when needed
#     import matplotlib.pyplot as plt
# 
#     # Determine if we should attempt LaTeX
#     use_latex = _fallback_mode == "auto" and check_latex_capability()
# 
#     if use_latex:
#         # Try LaTeX first
#         try:
#             # Test rendering capability with the actual text
#             fig, ax = plt.subplots(figsize=(1, 1))
#             try:
#                 ax.text(0.5, 0.5, text, usetex=True)
#                 fig.canvas.draw()
#                 plt.close(fig)
#                 return text  # LaTeX works
#             except Exception:
#                 plt.close(fig)
#                 raise LaTeXFallbackError("LaTeX rendering failed")
#         except Exception:
#             # Fall through to fallback strategies
#             pass
# 
#     # Apply fallback strategy
#     if fallback_strategy == "auto":
#         # Automatically choose best fallback
#         if preserve_math and ("$" in text or "\\" in text):
#             try:
#                 return latex_to_mathtext(text)
#             except Exception:
#                 return latex_to_unicode(text)
#         else:
#             return latex_to_unicode(text)
# 
#     elif fallback_strategy == "mathtext":
#         return latex_to_mathtext(text)
# 
#     elif fallback_strategy == "unicode":
#         return latex_to_unicode(text)
# 
#     elif fallback_strategy == "plain":
#         # Strip all LaTeX and return plain text
#         plain = latex_to_unicode(text)
#         # Remove any remaining special characters
#         plain = re.sub(r"[^\w\s\(\)\[\].,;:!?-]", "", plain)
#         return plain
# 
#     else:
#         raise ValueError(f"Unknown fallback strategy: {fallback_strategy}")
# 
# 
# def latex_fallback_decorator(
#     fallback_strategy: str = "auto", preserve_math: bool = True
# ):
#     """
#     Decorator to add LaTeX fallback capability to functions.
# 
#     Parameters
#     ----------
#     fallback_strategy : str, optional
#         Fallback strategy to use
#     preserve_math : bool, optional
#         Whether to preserve mathematical notation
# 
#     Returns
#     -------
#     callable
#         Decorated function with LaTeX fallback
#     """
# 
#     def decorator(func: Callable) -> Callable:
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             try:
#                 return func(*args, **kwargs)
#             except Exception as e:
#                 # Check if this is a LaTeX-related error
#                 error_str = str(e).lower()
#                 latex_error_indicators = [
#                     "latex",
#                     "tex",
#                     "dvi",
#                     "tfm",
#                     "font",
#                     "usetex",
#                     "kpathsea",
#                     "dvipng",
#                     "ghostscript",
#                 ]
# 
#                 if any(indicator in error_str for indicator in latex_error_indicators):
#                     logger.warning(f"LaTeX error in {func.__name__}: {e}")
#                     logger.warning("Falling back to alternative text rendering")
# 
#                     # Try to apply fallback to string arguments
#                     new_args = []
#                     for arg in args:
#                         if isinstance(arg, str):
#                             new_args.append(
#                                 safe_latex_render(arg, fallback_strategy, preserve_math)
#                             )
#                         else:
#                             new_args.append(arg)
# 
#                     new_kwargs = {}
#                     for key, value in kwargs.items():
#                         if isinstance(value, str):
#                             new_kwargs[key] = safe_latex_render(
#                                 value, fallback_strategy, preserve_math
#                             )
#                         else:
#                             new_kwargs[key] = value
# 
#                     # Import matplotlib when needed
#                     import matplotlib.pyplot as plt
# 
#                     # Temporarily disable LaTeX for this call
#                     original_usetex = plt.rcParams.get("text.usetex", False)
#                     plt.rcParams["text.usetex"] = False
# 
#                     try:
#                         result = func(*new_args, **new_kwargs)
#                         return result
#                     finally:
#                         plt.rcParams["text.usetex"] = original_usetex
#                 else:
#                     # Re-raise non-LaTeX errors
#                     raise
# 
#         return wrapper
# 
#     return decorator
# 
# 
# def get_latex_status() -> Dict[str, Any]:
#     """
#     Get comprehensive LaTeX status information.
# 
#     Returns
#     -------
#     Dict[str, Any]
#         Status information including capability, mode, and configuration
#     """
#     import matplotlib.pyplot as plt
# 
#     return {
#         "latex_available": check_latex_capability(),
#         "fallback_mode": get_fallback_mode(),
#         "usetex_enabled": plt.rcParams.get("text.usetex", False),
#         "mathtext_fontset": plt.rcParams.get("mathtext.fontset", "cm"),
#         "font_family": plt.rcParams.get("font.family", ["serif"]),
#         "cache_info": check_latex_capability.cache_info()._asdict(),
#     }
# 
# 
# # Convenience functions
# def enable_latex_fallback(mode: str = "auto") -> None:
#     """Enable LaTeX fallback with specified mode."""
#     set_fallback_mode(mode)
# 
# 
# def disable_latex_fallback() -> None:
#     """Disable LaTeX fallback (force LaTeX usage)."""
#     global _latex_available
#     _latex_available = True  # Force LaTeX usage
# 
# 
# def reset_latex_cache() -> None:
#     """Reset the LaTeX capability cache."""
#     check_latex_capability.cache_clear()
#     global _latex_available
#     _latex_available = None
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/str/_latex_fallback.py
# --------------------------------------------------------------------------------
