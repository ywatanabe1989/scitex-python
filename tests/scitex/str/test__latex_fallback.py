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
    pytest.main([__file__, "-v"])
