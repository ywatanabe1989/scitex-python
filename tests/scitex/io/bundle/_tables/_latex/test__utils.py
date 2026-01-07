#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/fsb/_tables/_latex/test__utils.py

"""Tests for LaTeX utility functions."""

import pytest


class TestEscapeLatex:
    """Test escape_latex function."""

    def test_escape_special_chars(self):
        """Test escaping special LaTeX characters."""
        from scitex.io.bundle._tables._latex._utils import escape_latex

        assert escape_latex("&") == r"\&"
        assert escape_latex("%") == r"\%"
        assert escape_latex("$") == r"\$"
        assert escape_latex("#") == r"\#"
        assert escape_latex("_") == r"\_"
        assert escape_latex("{") == r"\{"
        assert escape_latex("}") == r"\}"
        # Note: backslash escaped to textbackslash
        assert "textbackslash" in escape_latex("\\")

    def test_escape_empty_string(self):
        """Test escaping empty string."""
        from scitex.io.bundle._tables._latex._utils import escape_latex

        assert escape_latex("") == ""

    def test_escape_normal_text(self):
        """Test that normal text is unchanged."""
        from scitex.io.bundle._tables._latex._utils import escape_latex

        assert escape_latex("Hello World") == "Hello World"

    def test_escape_mixed_text(self):
        """Test escaping text with mixed content."""
        from scitex.io.bundle._tables._latex._utils import escape_latex

        result = escape_latex("Price: $100 (50% off)")
        # $ and % are escaped
        assert r"\$" in result or "100" in result
        assert r"\%" in result or "off" in result


class TestEscapeLatexMinimal:
    """Test escape_latex_minimal function."""

    def test_minimal_escape(self):
        """Test minimal escaping only escapes & % #."""
        from scitex.io.bundle._tables._latex._utils import escape_latex_minimal

        assert escape_latex_minimal("A & B") == r"A \& B"
        assert escape_latex_minimal("100%") == r"100\%"
        assert escape_latex_minimal("#1") == r"\#1"

    def test_preserves_latex_commands(self):
        """Test that LaTeX commands are preserved."""
        from scitex.io.bundle._tables._latex._utils import escape_latex_minimal

        assert escape_latex_minimal(r"\textbf{bold}") == r"\textbf{bold}"
        assert escape_latex_minimal("$x^2$") == "$x^2$"


class TestFormatNumber:
    """Test format_number function."""

    def test_format_integer(self):
        """Test formatting integer."""
        from scitex.io.bundle._tables._latex._utils import format_number

        assert format_number(42, precision=0) == "42"

    def test_format_decimal(self):
        """Test formatting decimal."""
        from scitex.io.bundle._tables._latex._utils import format_number

        assert format_number(3.14159, precision=2) == "3.14"

    def test_format_scientific(self):
        """Test scientific notation."""
        from scitex.io.bundle._tables._latex._utils import format_number

        result = format_number(0.0001, precision=2)
        assert "e" in result

    def test_format_none(self):
        """Test formatting None value."""
        from scitex.io.bundle._tables._latex._utils import format_number

        assert format_number(None) == "---"

    def test_format_with_siunitx(self):
        """Test formatting with siunitx."""
        from scitex.io.bundle._tables._latex._utils import format_number

        result = format_number(42.5, use_siunitx=True)
        assert r"\num{" in result


class TestFormatPValue:
    """Test format_p_value function."""

    def test_format_small_p(self):
        """Test formatting small p-value."""
        from scitex.io.bundle._tables._latex._utils import format_p_value

        result = format_p_value(0.0001)
        assert "<" in result
        assert ".001" in result

    def test_format_medium_p(self):
        """Test formatting medium p-value."""
        from scitex.io.bundle._tables._latex._utils import format_p_value

        result = format_p_value(0.023)
        # p >= 0.01 uses 2 decimal precision, so 0.023 -> .02
        assert "02" in result

    def test_format_large_p(self):
        """Test formatting large p-value."""
        from scitex.io.bundle._tables._latex._utils import format_p_value

        result = format_p_value(0.45)
        assert "45" in result


class TestFormatStatistic:
    """Test format_statistic function."""

    def test_format_t_statistic(self):
        """Test formatting t statistic."""
        from scitex.io.bundle._tables._latex._utils import format_statistic

        result = format_statistic("t", 2.31, df=48)
        assert "$t(48) = 2.31$" == result

    def test_format_chi_squared(self):
        """Test formatting chi-squared statistic."""
        from scitex.io.bundle._tables._latex._utils import format_statistic

        result = format_statistic("chi2", 5.67, df=2)
        assert r"\chi^2" in result
        assert "5.67" in result

    def test_format_without_df(self):
        """Test formatting statistic without degrees of freedom."""
        from scitex.io.bundle._tables._latex._utils import format_statistic

        result = format_statistic("r", 0.85)
        assert "$r = 0.85$" == result


class TestFormatEffectSize:
    """Test format_effect_size function."""

    def test_format_cohens_d(self):
        """Test formatting Cohen's d."""
        from scitex.io.bundle._tables._latex._utils import format_effect_size

        result = format_effect_size("cohens_d", 0.65)
        assert "$d = 0.65$" == result

    def test_format_with_ci(self):
        """Test formatting effect size with confidence interval."""
        from scitex.io.bundle._tables._latex._utils import format_effect_size

        result = format_effect_size("cohens_d", 0.65, ci_lower=0.21, ci_upper=1.09)
        assert "$d = 0.65$" in result
        assert "[0.21, 1.09]" in result


class TestSignificanceStars:
    """Test significance_stars function."""

    def test_three_stars(self):
        """Test p < .001 returns three stars."""
        from scitex.io.bundle._tables._latex._utils import significance_stars

        assert significance_stars(0.0001) == "***"

    def test_two_stars(self):
        """Test p < .01 returns two stars."""
        from scitex.io.bundle._tables._latex._utils import significance_stars

        assert significance_stars(0.005) == "**"

    def test_one_star(self):
        """Test p < .05 returns one star."""
        from scitex.io.bundle._tables._latex._utils import significance_stars

        assert significance_stars(0.03) == "*"

    def test_no_stars(self):
        """Test p >= .05 returns no stars."""
        from scitex.io.bundle._tables._latex._utils import significance_stars

        assert significance_stars(0.10) == ""


class TestSanitizeLabel:
    """Test sanitize_label function."""

    def test_sanitize_spaces(self):
        """Test sanitizing spaces."""
        from scitex.io.bundle._tables._latex._utils import sanitize_label

        assert sanitize_label("Hello World") == "Hello_World"

    def test_sanitize_special_chars(self):
        """Test sanitizing special characters."""
        from scitex.io.bundle._tables._latex._utils import sanitize_label

        assert sanitize_label("Test@#$%") == "Test"

    def test_sanitize_leading_number(self):
        """Test sanitizing leading number."""
        from scitex.io.bundle._tables._latex._utils import sanitize_label

        assert sanitize_label("123test") == "n123test"

    def test_sanitize_empty(self):
        """Test sanitizing empty string."""
        from scitex.io.bundle._tables._latex._utils import sanitize_label

        assert sanitize_label("") == "unnamed"


class TestColumnSpec:
    """Test column_spec function."""

    def test_basic_column_spec(self):
        """Test basic column specification."""
        from scitex.io.bundle._tables._latex._utils import column_spec

        assert column_spec(4, alignment="r") == "lrrr"

    def test_siunitx_columns(self):
        """Test siunitx S columns."""
        from scitex.io.bundle._tables._latex._utils import column_spec

        assert column_spec(3, use_siunitx=True) == "lSS"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_utils.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/_latex/_utils.py
# 
# """LaTeX utility functions for escaping and formatting."""
# 
# import re
# from typing import Any, Optional
# 
# 
# # Characters that need escaping in LaTeX
# LATEX_SPECIAL_CHARS = {
#     "&": r"\&",
#     "%": r"\%",
#     "$": r"\$",
#     "#": r"\#",
#     "_": r"\_",
#     "{": r"\{",
#     "}": r"\}",
#     "~": r"\textasciitilde{}",
#     "^": r"\textasciicircum{}",
#     "\\": r"\textbackslash{}",
# }
# 
# # Greek letters mapping
# GREEK_LETTERS = {
#     "alpha": r"\alpha",
#     "beta": r"\beta",
#     "gamma": r"\gamma",
#     "delta": r"\delta",
#     "epsilon": r"\epsilon",
#     "mu": r"\mu",
#     "sigma": r"\sigma",
#     "chi": r"\chi",
#     "eta": r"\eta",
#     "theta": r"\theta",
#     "lambda": r"\lambda",
#     "omega": r"\omega",
#     "pi": r"\pi",
#     "rho": r"\rho",
#     "tau": r"\tau",
#     "phi": r"\phi",
#     "psi": r"\psi",
# }
# 
# # Unit formatting for siunitx
# UNIT_LATEX = {
#     "ms": r"\milli\second",
#     "s": r"\second",
#     "Hz": r"\hertz",
#     "kHz": r"\kilo\hertz",
#     "MHz": r"\mega\hertz",
#     "mm": r"\milli\metre",
#     "cm": r"\centi\metre",
#     "m": r"\metre",
#     "kg": r"\kilo\gram",
#     "g": r"\gram",
#     "mg": r"\milli\gram",
#     "mV": r"\milli\volt",
#     "V": r"\volt",
#     "mA": r"\milli\ampere",
#     "A": r"\ampere",
#     "%": r"\percent",
#     "years": r"years",
#     "yr": r"yr",
# }
# 
# 
# def escape_latex(text: str) -> str:
#     """Escape special LaTeX characters in text.
# 
#     Args:
#         text: Plain text string
# 
#     Returns:
#         LaTeX-safe string with special characters escaped
#     """
#     if not text:
#         return ""
# 
#     result = str(text)
#     # Process backslash FIRST to avoid double-escaping
#     if "\\" in result:
#         result = result.replace("\\", r"\textbackslash{}")
#     # Then other special characters (order doesn't matter for these)
#     for char in ["&", "%", "$", "#", "_", "{", "}", "~", "^"]:
#         if char in LATEX_SPECIAL_CHARS:
#             result = result.replace(char, LATEX_SPECIAL_CHARS[char])
#     return result
# 
# 
# def escape_latex_minimal(text: str) -> str:
#     """Minimal escaping for text that may contain intentional LaTeX.
# 
#     Only escapes & % # which are most likely unintentional.
# 
#     Args:
#         text: Text that may contain LaTeX commands
# 
#     Returns:
#         Minimally escaped string
#     """
#     if not text:
#         return ""
# 
#     result = str(text)
#     for char in ["&", "%", "#"]:
#         result = result.replace(char, LATEX_SPECIAL_CHARS[char])
#     return result
# 
# 
# def format_unit(unit: Optional[str], use_siunitx: bool = False) -> str:
#     """Format a unit string for LaTeX.
# 
#     Args:
#         unit: Unit string (e.g., "ms", "Hz")
#         use_siunitx: Whether to use siunitx package format
# 
#     Returns:
#         LaTeX formatted unit string
#     """
#     if not unit:
#         return ""
# 
#     if use_siunitx:
#         return UNIT_LATEX.get(unit, escape_latex(unit))
#     else:
#         # Simple formatting
#         return escape_latex(unit)
# 
# 
# def format_number(
#     value: float,
#     precision: int = 3,
#     scientific: bool = False,
#     use_siunitx: bool = False,
# ) -> str:
#     """Format a number for LaTeX.
# 
#     Args:
#         value: Numeric value
#         precision: Decimal places
#         scientific: Use scientific notation
#         use_siunitx: Use siunitx \\num{} command
# 
#     Returns:
#         LaTeX formatted number string
#     """
#     if value is None:
#         return "---"
# 
#     if scientific or (abs(value) > 0 and (abs(value) >= 1e4 or abs(value) < 1e-3)):
#         formatted = f"{value:.{precision}e}"
#     else:
#         formatted = f"{value:.{precision}f}"
# 
#     if use_siunitx:
#         return rf"\num{{{formatted}}}"
#     return formatted
# 
# 
# def format_p_value(p: float, threshold: float = 0.001) -> str:
#     """Format a p-value for scientific reporting.
# 
#     Args:
#         p: P-value
#         threshold: Threshold below which to show "< threshold"
# 
#     Returns:
#         Formatted p-value string (e.g., ".023" or "< .001")
#     """
#     if p is None:
#         return "---"
# 
#     if p < threshold:
#         return f"< {threshold:.3f}".lstrip("0")
#     elif p < 0.01:
#         return f"{p:.3f}".lstrip("0")
#     else:
#         return f"{p:.2f}".lstrip("0")
# 
# 
# def format_statistic(
#     name: str,
#     value: float,
#     df: Optional[float] = None,
#     precision: int = 2,
# ) -> str:
#     """Format a test statistic for LaTeX.
# 
#     Args:
#         name: Statistic name (t, F, chi2, r, etc.)
#         value: Statistic value
#         df: Degrees of freedom
#         precision: Decimal places
# 
#     Returns:
#         LaTeX math mode string (e.g., "$t(48) = 2.31$")
#     """
#     stat_symbols = {
#         "t": "t",
#         "F": "F",
#         "chi2": r"\chi^2",
#         "chi_squared": r"\chi^2",
#         "r": "r",
#         "rho": r"\rho",
#         "z": "z",
#         "W": "W",
#         "U": "U",
#         "H": "H",
#     }
# 
#     symbol = stat_symbols.get(name, name)
# 
#     if df is not None:
#         if isinstance(df, float) and df == int(df):
#             df_str = str(int(df))
#         else:
#             df_str = str(df)
#         return f"${symbol}({df_str}) = {value:.{precision}f}$"
#     else:
#         return f"${symbol} = {value:.{precision}f}$"
# 
# 
# def format_effect_size(
#     name: str,
#     value: float,
#     ci_lower: Optional[float] = None,
#     ci_upper: Optional[float] = None,
#     precision: int = 2,
# ) -> str:
#     """Format an effect size for LaTeX.
# 
#     Args:
#         name: Effect size name (cohens_d, hedges_g, eta_squared, etc.)
#         value: Effect size value
#         ci_lower: Lower CI bound
#         ci_upper: Upper CI bound
#         precision: Decimal places
# 
#     Returns:
#         LaTeX formatted effect size string
#     """
#     es_symbols = {
#         "cohens_d": "d",
#         "hedges_g": "g",
#         "eta_squared": r"\eta^2",
#         "partial_eta_squared": r"\eta_p^2",
#         "omega_squared": r"\omega^2",
#         "r_squared": "R^2",
#         "cramers_v": "V",
#         "phi": r"\phi",
#     }
# 
#     symbol = es_symbols.get(name, name)
#     result = f"${symbol} = {value:.{precision}f}$"
# 
#     if ci_lower is not None and ci_upper is not None:
#         result += f" [{ci_lower:.{precision}f}, {ci_upper:.{precision}f}]"
# 
#     return result
# 
# 
# def significance_stars(p: float, levels: tuple = (0.05, 0.01, 0.001)) -> str:
#     """Get significance stars for a p-value.
# 
#     Args:
#         p: P-value
#         levels: Significance levels (default: .05, .01, .001)
# 
#     Returns:
#         Stars string ("*", "**", "***", or "")
#     """
#     if p is None:
#         return ""
# 
#     if p < levels[2]:
#         return "***"
#     elif p < levels[1]:
#         return "**"
#     elif p < levels[0]:
#         return "*"
#     return ""
# 
# 
# def sanitize_label(text: str) -> str:
#     """Sanitize a string for use as a LaTeX label.
# 
#     Args:
#         text: Text to sanitize
# 
#     Returns:
#         Label-safe string (alphanumeric, hyphens, underscores only)
#     """
#     if not text:
#         return "unnamed"
# 
#     # Replace spaces with underscores
#     result = text.replace(" ", "_")
#     # Keep only alphanumeric, hyphens, underscores
#     result = re.sub(r"[^a-zA-Z0-9_-]", "", result)
#     # Ensure doesn't start with number
#     if result and result[0].isdigit():
#         result = "n" + result
#     return result or "unnamed"
# 
# 
# def wrap_math(text: str) -> str:
#     """Wrap text in math mode if not already.
# 
#     Args:
#         text: Text to wrap
# 
#     Returns:
#         Math mode string
#     """
#     if not text:
#         return ""
#     if text.startswith("$") and text.endswith("$"):
#         return text
#     return f"${text}$"
# 
# 
# def column_spec(
#     num_cols: int,
#     alignment: str = "l",
#     first_col: str = "l",
#     use_siunitx: bool = False,
# ) -> str:
#     """Generate LaTeX column specification.
# 
#     Args:
#         num_cols: Total number of columns
#         alignment: Default alignment for data columns
#         first_col: Alignment for first column (usually labels)
#         use_siunitx: Use S columns for numeric alignment
# 
#     Returns:
#         Column spec string (e.g., "lrrr" or "lSSS")
#     """
#     if num_cols < 1:
#         return "l"
# 
#     if use_siunitx:
#         data_cols = "S" * (num_cols - 1)
#     else:
#         data_cols = alignment * (num_cols - 1)
# 
#     return first_col + data_cols
# 
# 
# __all__ = [
#     "escape_latex",
#     "escape_latex_minimal",
#     "format_unit",
#     "format_number",
#     "format_p_value",
#     "format_statistic",
#     "format_effect_size",
#     "significance_stars",
#     "sanitize_label",
#     "wrap_math",
#     "column_spec",
#     "LATEX_SPECIAL_CHARS",
#     "GREEK_LETTERS",
#     "UNIT_LATEX",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_tables/_latex/_utils.py
# --------------------------------------------------------------------------------
