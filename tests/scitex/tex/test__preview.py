#!/usr/bin/env python3
# Time-stamp: "2026-01-05 14:00:00 (ywatanabe)"
# File: ./tests/scitex/tex/test__preview.py

"""Comprehensive tests for tex._preview module"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Required for scitex.tex module
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Use non-interactive backend for testing
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from scitex.tex._preview import FALLBACK_AVAILABLE


class TestPreviewWithoutFallback:
    """Tests for preview function with enable_fallback=False."""

    def test_preview_single_tex_string(self):
        """Test preview with single LaTeX string."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = preview(["x^2"], enable_fallback=False)

            assert result == mock_fig
            mock_subplots.assert_called_once_with(nrows=1, ncols=1, figsize=(10, 3))
            assert mock_ax.text.call_count == 2
            mock_ax.text.assert_any_call(
                0.5, 0.7, "x^2", size=20, ha="center", va="center"
            )
            mock_ax.text.assert_any_call(
                0.5, 0.3, "$x^2$", size=20, ha="center", va="center"
            )
            mock_ax.hide_spines.assert_called_once()
            mock_fig.tight_layout.assert_called_once()

    def test_preview_multiple_tex_strings(self):
        """Test preview with multiple LaTeX strings."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_axes = [Mock(spec=Axes) for _ in range(3)]
            for ax in mock_axes:
                ax.text = Mock()
                ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)

            tex_strings = ["x^2", r"\sum_{i=1}^n i", r"\frac{a}{b}"]
            result = preview(tex_strings, enable_fallback=False)

            assert result == mock_fig
            mock_subplots.assert_called_once_with(nrows=3, ncols=1, figsize=(10, 9))

            for ax, tex_str in zip(mock_axes, tex_strings):
                assert ax.text.call_count == 2
                ax.text.assert_any_call(
                    0.5, 0.7, tex_str, size=20, ha="center", va="center"
                )
                ax.text.assert_any_call(
                    0.5, 0.3, f"${tex_str}$", size=20, ha="center", va="center"
                )
                ax.hide_spines.assert_called_once()

            mock_fig.tight_layout.assert_called_once()

    def test_preview_empty_list(self):
        """Test preview with empty list."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_subplots.return_value = (mock_fig, np.array([]))
            mock_fig.tight_layout = Mock()

            result = preview([], enable_fallback=False)

            assert result == mock_fig
            mock_subplots.assert_called_once_with(nrows=0, ncols=1, figsize=(10, 0))
            mock_fig.tight_layout.assert_called_once()

    def test_preview_complex_latex(self):
        """Test preview with complex LaTeX expressions."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            complex_tex = r"\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}"
            result = preview([complex_tex], enable_fallback=False)

            assert result == mock_fig
            mock_ax.text.assert_any_call(
                0.5, 0.7, complex_tex, size=20, ha="center", va="center"
            )
            mock_ax.text.assert_any_call(
                0.5, 0.3, f"${complex_tex}$", size=20, ha="center", va="center"
            )

    def test_preview_special_characters(self):
        """Test preview with special LaTeX characters."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            special_tex = r"\alpha \beta \gamma \delta \epsilon"
            result = preview([special_tex], enable_fallback=False)

            assert result == mock_fig
            mock_ax.text.assert_any_call(
                0.5, 0.7, special_tex, size=20, ha="center", va="center"
            )
            mock_ax.text.assert_any_call(
                0.5, 0.3, f"${special_tex}$", size=20, ha="center", va="center"
            )

    def test_preview_text_positioning(self):
        """Test that text is positioned correctly."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = preview(["E=mc^2"], enable_fallback=False)

            calls = mock_ax.text.call_args_list
            assert len(calls) == 2
            # First call: raw string at y=0.7
            assert calls[0][0] == (0.5, 0.7, "E=mc^2")
            assert calls[0][1] == {"size": 20, "ha": "center", "va": "center"}
            # Second call: LaTeX string at y=0.3
            assert calls[1][0] == (0.5, 0.3, "$E=mc^2$")
            assert calls[1][1] == {"size": 20, "ha": "center", "va": "center"}

    def test_preview_figure_size_scaling(self):
        """Test that figure size scales with number of strings."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_axes = [Mock(spec=Axes) for _ in range(5)]
            for ax in mock_axes:
                ax.text = Mock()
                ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)

            tex_strings = ["a", "b", "c", "d", "e"]
            result = preview(tex_strings, enable_fallback=False)

            mock_subplots.assert_called_once_with(nrows=5, ncols=1, figsize=(10, 15))

    def test_preview_matrix_latex(self):
        """Test preview with matrix LaTeX notation."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            matrix_tex = r"\begin{pmatrix} a & b \\ c & d \end{pmatrix}"
            result = preview([matrix_tex], enable_fallback=False)

            assert result == mock_fig
            mock_ax.text.assert_any_call(
                0.5, 0.7, matrix_tex, size=20, ha="center", va="center"
            )
            mock_ax.text.assert_any_call(
                0.5, 0.3, f"${matrix_tex}$", size=20, ha="center", va="center"
            )


class TestPreviewWithFallback:
    """Tests for preview function with enable_fallback=True (default)."""

    def test_preview_with_fallback_enabled(self):
        """Test preview with fallback enabled (default)."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = preview(["x^2"])  # default enable_fallback=True

            assert result == mock_fig
            assert mock_ax.text.call_count == 2
            # When fallback is available, text may be converted
            if FALLBACK_AVAILABLE:
                # Verify text was called twice (raw and latex formatted)
                calls = mock_ax.text.call_args_list
                # First call is raw at y=0.7
                assert calls[0][0][0] == 0.5
                assert calls[0][0][1] == 0.7
                # Second call is latex at y=0.3
                assert calls[1][0][0] == 0.5
                assert calls[1][0][1] == 0.3

    def test_preview_fallback_converts_text(self):
        """Test that fallback converts superscripts to unicode."""
        from scitex.tex import preview

        if not FALLBACK_AVAILABLE:
            pytest.skip("Fallback module not available")

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = preview(["x^2"])

            calls = mock_ax.text.call_args_list
            # First call (raw at y=0.7) should have unicode conversion
            first_call_text = calls[0][0][2]
            # Should contain unicode superscript 2
            assert "x" in first_call_text
            # The second character should be superscript 2 (²)
            assert "²" in first_call_text or "^" in first_call_text


class TestPreviewEdgeCases:
    """Tests for edge cases in preview function."""

    def test_preview_single_string_input(self):
        """Test preview converts single string to list."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Single string should be converted to list
            result = preview("x^2", enable_fallback=False)

            assert result == mock_fig
            mock_subplots.assert_called_once_with(nrows=1, ncols=1, figsize=(10, 3))

    def test_preview_with_numpy_array_axes(self):
        """Test preview handles numpy array of axes correctly."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, np.array(mock_ax))

            result = preview(["test"], enable_fallback=False)

            assert result == mock_fig
            mock_ax.text.assert_any_call(
                0.5, 0.7, "test", size=20, ha="center", va="center"
            )

    def test_preview_unicode_strings(self):
        """Test preview with Unicode strings."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            unicode_tex = "∑ᵢ₌₁ⁿ xᵢ²"
            result = preview([unicode_tex], enable_fallback=False)

            assert result == mock_fig
            mock_ax.text.assert_any_call(
                0.5, 0.7, unicode_tex, size=20, ha="center", va="center"
            )
            mock_ax.text.assert_any_call(
                0.5, 0.3, f"${unicode_tex}$", size=20, ha="center", va="center"
            )

    def test_preview_already_wrapped_in_dollars(self):
        """Test preview handles strings already wrapped in $ correctly."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # Already wrapped in $ should not double wrap
            result = preview(["$x^2$"], enable_fallback=False)

            assert result == mock_fig
            calls = mock_ax.text.call_args_list
            # Second call should use the string as-is (no double wrapping)
            assert calls[1][0][2] == "$x^2$"


class TestPreviewErrorHandling:
    """Tests for error handling in preview function."""

    def test_preview_error_recovery(self):
        """Test preview handles errors gracefully."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock(side_effect=Exception("Layout error"))
            mock_subplots.return_value = (mock_fig, mock_ax)

            with pytest.raises(Exception, match="Layout error"):
                preview(["test"], enable_fallback=False)

    def test_preview_none_input(self):
        """Test preview with None input - wraps in list and proceeds."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # None gets wrapped in list: [None], nrows=1
            result = preview(None, enable_fallback=False)
            assert result == mock_fig
            mock_subplots.assert_called_once_with(nrows=1, ncols=1, figsize=(10, 3))

    def test_preview_int_input(self):
        """Test preview with int input - wraps in list and proceeds."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_ax = Mock(spec=Axes)
            mock_ax.text = Mock()
            mock_ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            # int gets wrapped in list: [123], nrows=1
            result = preview(123, enable_fallback=False)
            assert result == mock_fig
            mock_subplots.assert_called_once_with(nrows=1, ncols=1, figsize=(10, 3))


class TestPreviewPerformance:
    """Tests for preview performance."""

    def test_preview_long_list_performance(self):
        """Test preview performance with many LaTeX strings."""
        import time

        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_axes = [Mock(spec=Axes) for _ in range(100)]
            for ax in mock_axes:
                ax.text = Mock()
                ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)

            tex_strings = [f"x^{{{i}}}" for i in range(100)]

            start_time = time.time()
            result = preview(tex_strings, enable_fallback=False)
            elapsed = time.time() - start_time

            assert elapsed < 1.0
            assert result == mock_fig


class TestPreviewMixedContent:
    """Tests for preview with mixed content types."""

    def test_preview_mixed_content(self):
        """Test preview with mixed LaTeX and plain text."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_axes = [Mock(spec=Axes) for _ in range(3)]
            for ax in mock_axes:
                ax.text = Mock()
                ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)

            mixed_content = ["Plain text", r"\LaTeX", "x + y = z"]
            result = preview(mixed_content, enable_fallback=False)

            assert result == mock_fig
            for ax, content in zip(mock_axes, mixed_content):
                ax.text.assert_any_call(
                    0.5, 0.7, content, size=20, ha="center", va="center"
                )
                ax.text.assert_any_call(
                    0.5, 0.3, f"${content}$", size=20, ha="center", va="center"
                )


class TestPreviewDocstrings:
    """Tests for docstring examples."""

    def test_preview_docstring_example(self):
        """Test the example from the docstring works."""
        from scitex.tex import preview

        with patch("scitex.plt.subplots") as mock_subplots:
            mock_fig = Mock(spec=Figure)
            mock_axes = [Mock(spec=Axes) for _ in range(3)]
            for ax in mock_axes:
                ax.text = Mock()
                ax.hide_spines = Mock()
            mock_fig.tight_layout = Mock()
            mock_subplots.return_value = (mock_fig, mock_axes)

            # Example from docstring
            tex_strings = ["x^2", r"\sum_{i=1}^n i", r"\alpha + \beta"]
            fig = preview(tex_strings, enable_fallback=False)

            assert fig == mock_fig

            # Verify strings were rendered
            mock_axes[0].text.assert_any_call(
                0.5, 0.7, "x^2", size=20, ha="center", va="center"
            )
            mock_axes[0].text.assert_any_call(
                0.5, 0.3, "$x^2$", size=20, ha="center", va="center"
            )


class TestFallbackAvailability:
    """Tests for fallback availability."""

    def test_fallback_available_is_bool(self):
        """Test FALLBACK_AVAILABLE is a boolean."""
        assert isinstance(FALLBACK_AVAILABLE, bool)


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
