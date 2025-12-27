#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex.stats.auto._styles module.

Tests journal style presets and formatting methods for statistical output.
"""

import pytest

from scitex.stats.auto._styles import (
    StatStyle,
    OutputTarget,
    STAT_STYLES,
    get_stat_style,
    list_styles,
    APA_LATEX_STYLE,
    APA_HTML_STYLE,
    NATURE_LATEX_STYLE,
    NATURE_HTML_STYLE,
    CELL_LATEX_STYLE,
    CELL_HTML_STYLE,
    ELSEVIER_LATEX_STYLE,
    ELSEVIER_HTML_STYLE,
    PLAIN_STYLE,
)


class TestStatStyleDataclass:
    """Tests for StatStyle dataclass creation and defaults."""

    def test_basic_creation(self):
        """Test creating a StatStyle with required fields."""
        style = StatStyle(
            id="test",
            label="Test Style",
            target="plain",
        )
        assert style.id == "test"
        assert style.label == "Test Style"
        assert style.target == "plain"

    def test_default_values(self):
        """Test default values are set correctly."""
        style = StatStyle(id="test", label="Test", target="plain")
        assert style.decimal_places_p == 3
        assert style.decimal_places_stat == 2
        assert style.decimal_places_effect == 2
        assert style.p_format == "p = {p:.3f}"

    def test_custom_decimal_places(self):
        """Test custom decimal places."""
        style = StatStyle(
            id="test",
            label="Test",
            target="plain",
            decimal_places_p=4,
            decimal_places_stat=3,
            decimal_places_effect=3,
        )
        assert style.decimal_places_p == 4
        assert style.decimal_places_stat == 3
        assert style.decimal_places_effect == 3


class TestFormatStat:
    """Tests for format_stat method."""

    def test_format_t_statistic_with_df(self):
        """Test formatting t-statistic with degrees of freedom."""
        style = StatStyle(
            id="test",
            label="Test",
            target="plain",
            stat_symbol_format={"t": "t"},
        )
        result = style.format_stat("t", 2.31, df=28)
        assert result == "t(28.0) = 2.31"

    def test_format_t_statistic_without_df(self):
        """Test formatting t-statistic without degrees of freedom."""
        style = StatStyle(
            id="test",
            label="Test",
            target="plain",
            stat_symbol_format={"t": "t"},
        )
        result = style.format_stat("t", 2.31)
        assert result == "t = 2.31"

    def test_format_with_latex_symbol(self):
        """Test formatting with LaTeX symbol."""
        style = StatStyle(
            id="test",
            label="Test",
            target="latex",
            stat_symbol_format={"t": "\\mathit{t}"},
        )
        result = style.format_stat("t", 2.31, df=28)
        assert "\\mathit{t}" in result
        assert "28.0" in result

    def test_format_unknown_symbol_uses_raw(self):
        """Test that unknown symbols are used as-is."""
        style = StatStyle(id="test", label="Test", target="plain")
        result = style.format_stat("Z", 1.96)
        assert result == "Z = 1.96"

    def test_format_respects_decimal_places(self):
        """Test that decimal places are respected."""
        style = StatStyle(
            id="test",
            label="Test",
            target="plain",
            decimal_places_stat=4,
        )
        result = style.format_stat("t", 2.3456789)
        assert "2.3457" in result


class TestFormatP:
    """Tests for format_p method."""

    def test_format_standard_p_value(self):
        """Test formatting a standard p-value."""
        style = APA_LATEX_STYLE
        result = style.format_p(0.032)
        assert "0.032" in result
        assert "\\mathit{p}" in result

    def test_format_very_small_p_value(self):
        """Test formatting very small p-values."""
        style = APA_LATEX_STYLE
        result = style.format_p(0.0001)
        # Should show as < 0.001 for very small values
        assert "0.001" in result or "0.000" in result

    def test_format_p_html_style(self):
        """Test formatting p-value with HTML style."""
        style = APA_HTML_STYLE
        result = style.format_p(0.045)
        assert "<i>p</i>" in result


class TestFormatEffect:
    """Tests for format_effect method."""

    def test_format_cohens_d(self):
        """Test formatting Cohen's d."""
        style = APA_LATEX_STYLE
        result = style.format_effect("cohens_d_ind", 0.72)
        assert "0.72" in result

    def test_format_unknown_effect(self):
        """Test formatting unknown effect size uses raw name."""
        style = StatStyle(id="test", label="Test", target="plain")
        result = style.format_effect("custom_effect", 0.5)
        assert "0.5" in result


class TestFormatN:
    """Tests for format_n method."""

    def test_format_n_with_group_name(self):
        """Test formatting sample size with group name."""
        style = APA_LATEX_STYLE
        result = style.format_n("A", 30)
        assert "30" in result
        assert "A" in result

    def test_format_n_plain_style(self):
        """Test formatting sample size with plain style."""
        style = PLAIN_STYLE
        result = style.format_n("Control", 25)
        assert "Control" in result
        assert "25" in result

    def test_format_n_without_group_placeholder(self):
        """Test styles that don't include group name in format."""
        style = NATURE_LATEX_STYLE
        result = style.format_n("A", 30)
        assert "30" in result


class TestPToStars:
    """Tests for p_to_stars method."""

    def test_highly_significant(self):
        """Test p < 0.001 returns ***."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(0.0001) == "***"

    def test_very_significant(self):
        """Test p < 0.01 returns **."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(0.005) == "**"

    def test_significant(self):
        """Test p < 0.05 returns *."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(0.03) == "*"

    def test_not_significant(self):
        """Test p >= 0.05 returns ns."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(0.08) == "ns"

    def test_boundary_001(self):
        """Test exact boundary at 0.001."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(0.001) == "**"

    def test_boundary_01(self):
        """Test exact boundary at 0.01."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(0.01) == "*"

    def test_boundary_05(self):
        """Test exact boundary at 0.05."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(0.05) == "ns"

    def test_none_returns_ns(self):
        """Test None p-value returns ns."""
        style = APA_LATEX_STYLE
        assert style.p_to_stars(None) == "ns"


class TestSTAT_STYLESRegistry:
    """Tests for STAT_STYLES registry."""

    def test_registry_is_dict(self):
        """Test that STAT_STYLES is a dictionary."""
        assert isinstance(STAT_STYLES, dict)

    def test_registry_not_empty(self):
        """Test that registry is not empty."""
        assert len(STAT_STYLES) > 0

    def test_registry_has_expected_styles(self):
        """Test that registry contains expected style IDs."""
        expected_ids = [
            "apa_latex",
            "apa_html",
            "nature_latex",
            "nature_html",
            "cell_latex",
            "cell_html",
            "elsevier_latex",
            "elsevier_html",
            "plain",
        ]
        for style_id in expected_ids:
            assert style_id in STAT_STYLES, f"Missing style: {style_id}"

    def test_all_values_are_statstyle(self):
        """Test that all registry values are StatStyle instances."""
        for style_id, style in STAT_STYLES.items():
            assert isinstance(style, StatStyle), f"{style_id} is not StatStyle"

    def test_all_ids_match_keys(self):
        """Test that style IDs match their registry keys."""
        for key, style in STAT_STYLES.items():
            assert key == style.id, f"Key {key} != style.id {style.id}"


class TestGetStatStyle:
    """Tests for get_stat_style function."""

    def test_get_valid_style(self):
        """Test getting a valid style by ID."""
        style = get_stat_style("apa_latex")
        assert style.id == "apa_latex"
        assert isinstance(style, StatStyle)

    def test_get_plain_style(self):
        """Test getting plain style."""
        style = get_stat_style("plain")
        assert style.target == "plain"

    def test_get_invalid_style_returns_fallback(self):
        """Test getting invalid style returns APA LaTeX as fallback."""
        style = get_stat_style("nonexistent_style")
        assert style.id == "apa_latex"


class TestListStyles:
    """Tests for list_styles function."""

    def test_returns_list(self):
        """Test that list_styles returns a list."""
        styles = list_styles()
        assert isinstance(styles, list)

    def test_contains_expected_styles(self):
        """Test that list contains expected style IDs."""
        styles = list_styles()
        assert "apa_latex" in styles
        assert "nature_latex" in styles
        assert "plain" in styles

    def test_count_matches_registry(self):
        """Test that count matches registry size."""
        styles = list_styles()
        assert len(styles) == len(STAT_STYLES)

    def test_filter_by_latex_target(self):
        """Test filtering by LaTeX target."""
        latex_styles = list_styles(target="latex")
        for style_id in latex_styles:
            assert STAT_STYLES[style_id].target == "latex"

    def test_filter_by_html_target(self):
        """Test filtering by HTML target."""
        html_styles = list_styles(target="html")
        for style_id in html_styles:
            assert STAT_STYLES[style_id].target == "html"


class TestJournalStylePresets:
    """Tests for individual journal style presets."""

    def test_apa_latex_style(self):
        """Test APA LaTeX style properties."""
        assert APA_LATEX_STYLE.id == "apa_latex"
        assert APA_LATEX_STYLE.target == "latex"
        assert "\\mathit{t}" in APA_LATEX_STYLE.stat_symbol_format.get("t", "")
        assert "\\mathit{p}" in APA_LATEX_STYLE.stat_symbol_format.get("p", "")

    def test_apa_html_style(self):
        """Test APA HTML style properties."""
        assert APA_HTML_STYLE.id == "apa_html"
        assert APA_HTML_STYLE.target == "html"
        assert "<i>t</i>" in APA_HTML_STYLE.stat_symbol_format.get("t", "")

    def test_nature_latex_style(self):
        """Test Nature LaTeX style properties."""
        assert NATURE_LATEX_STYLE.id == "nature_latex"
        assert NATURE_LATEX_STYLE.target == "latex"

    def test_nature_html_style(self):
        """Test Nature HTML style properties."""
        assert NATURE_HTML_STYLE.id == "nature_html"
        assert NATURE_HTML_STYLE.target == "html"

    def test_cell_latex_style(self):
        """Test Cell LaTeX style properties."""
        assert CELL_LATEX_STYLE.id == "cell_latex"
        assert CELL_LATEX_STYLE.target == "latex"

    def test_cell_html_style(self):
        """Test Cell HTML style properties."""
        assert CELL_HTML_STYLE.id == "cell_html"
        assert CELL_HTML_STYLE.target == "html"

    def test_elsevier_latex_style(self):
        """Test Elsevier LaTeX style properties."""
        assert ELSEVIER_LATEX_STYLE.id == "elsevier_latex"
        assert ELSEVIER_LATEX_STYLE.target == "latex"

    def test_elsevier_html_style(self):
        """Test Elsevier HTML style properties."""
        assert ELSEVIER_HTML_STYLE.id == "elsevier_html"
        assert ELSEVIER_HTML_STYLE.target == "html"

    def test_plain_style(self):
        """Test plain style properties."""
        assert PLAIN_STYLE.id == "plain"
        assert PLAIN_STYLE.target == "plain"


class TestAlphaThresholds:
    """Tests for alpha threshold configurations."""

    def test_apa_has_standard_thresholds(self):
        """Test APA has standard 0.001, 0.01, 0.05 thresholds."""
        thresholds = [t[0] for t in APA_LATEX_STYLE.alpha_thresholds]
        assert 0.001 in thresholds
        assert 0.01 in thresholds
        assert 0.05 in thresholds

    def test_thresholds_in_ascending_order(self):
        """Test that thresholds are in ascending order."""
        for style in STAT_STYLES.values():
            thresholds = [t[0] for t in style.alpha_thresholds]
            assert thresholds == sorted(thresholds)

    def test_stars_decrease_with_threshold(self):
        """Test that star count decreases with increasing threshold."""
        thresholds = APA_LATEX_STYLE.alpha_thresholds
        star_counts = [len(t[1].replace("ns", "")) for t in thresholds]
        # Stars should decrease (or stay same) as p increases
        for i in range(len(star_counts) - 1):
            assert star_counts[i] >= star_counts[i + 1]


class TestOutputTargets:
    """Tests for output target consistency."""

    def test_latex_styles_have_latex_target(self):
        """Test all latex styles have latex target."""
        latex_styles = [s for s in STAT_STYLES.values() if "latex" in s.id]
        for style in latex_styles:
            assert style.target == "latex"

    def test_html_styles_have_html_target(self):
        """Test all HTML styles have HTML target."""
        html_styles = [s for s in STAT_STYLES.values() if "html" in s.id]
        for style in html_styles:
            assert style.target == "html"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_styles.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_styles.py
# 
# """
# Journal Style Presets - Publication-ready statistical formatting.
# 
# This module defines formatting styles for major journal families:
# - APA (American Psychological Association)
# - Nature (Nature family journals)
# - Cell (Cell Press journals)
# - Elsevier (Generic biomedical)
# 
# Each style specifies:
# - How to format statistics (italic, symbols)
# - How to format p-values (thresholds, precision)
# - How to format effect sizes
# - How to format sample sizes
# 
# Supports both LaTeX and HTML output targets.
# """
# 
# from __future__ import annotations
# 
# from dataclasses import dataclass, field
# from typing import Dict, List, Literal, Optional, Tuple, Any
# 
# # =============================================================================
# # Type Aliases
# # =============================================================================
# 
# OutputTarget = Literal["latex", "html", "plain"]
# 
# 
# # =============================================================================
# # StatStyle
# # =============================================================================
# 
# 
# @dataclass
# class StatStyle:
#     """
#     Style configuration for statistical reporting.
# 
#     Defines how to format statistical results for a specific journal
#     or output format.
# 
#     Parameters
#     ----------
#     id : str
#         Unique identifier for this style.
#     label : str
#         Human-readable label (e.g., "APA (LaTeX)").
#     target : OutputTarget
#         Output format: "latex", "html", or "plain".
#     stat_symbol_format : dict
#         Maps statistic symbols to their formatted versions.
#         Example: {"t": "\\mathit{t}", "p": "\\mathit{p}"}
#     p_format : str
#         Format string for p-values. Use {p:.3f} syntax.
#     alpha_thresholds : list of (float, str)
#         P-value thresholds for stars. Lower threshold = more stars.
#         Example: [(0.001, "***"), (0.01, "**"), (0.05, "*")]
#     effect_label_format : dict
#         Maps effect size names to their formatted labels.
#     n_format : str
#         Format string for sample sizes. Uses % formatting.
#         Example: "\\mathit{n}_{%s} = %d"
#     decimal_places_p : int
#         Decimal places for p-values.
#     decimal_places_stat : int
#         Decimal places for test statistics.
#     decimal_places_effect : int
#         Decimal places for effect sizes.
# 
#     Examples
#     --------
#     >>> style = STAT_STYLES["apa_latex"]
#     >>> style.format_p(0.032)
#     '\\\\mathit{p} = 0.032'
#     >>> style.format_stat("t", 2.31, df=28)
#     '\\\\mathit{t}(28.0) = 2.31'
#     """
# 
#     id: str
#     label: str
#     target: OutputTarget
# 
#     # Symbol formatting
#     stat_symbol_format: Dict[str, str] = field(default_factory=dict)
# 
#     # P-value formatting
#     p_format: str = "p = {p:.3f}"
#     alpha_thresholds: List[Tuple[float, str]] = field(default_factory=list)
# 
#     # Effect size labels
#     effect_label_format: Dict[str, str] = field(default_factory=dict)
# 
#     # Sample size formatting
#     n_format: str = "n_{%s} = %d"
# 
#     # Decimal places
#     decimal_places_p: int = 3
#     decimal_places_stat: int = 2
#     decimal_places_effect: int = 2
# 
#     def format_stat(
#         self,
#         symbol: str,
#         value: float,
#         df: Optional[float] = None,
#     ) -> str:
#         """
#         Format a test statistic.
# 
#         Parameters
#         ----------
#         symbol : str
#             Statistic symbol (e.g., "t", "F", "chi2").
#         value : float
#             Statistic value.
#         df : float, optional
#             Degrees of freedom.
# 
#         Returns
#         -------
#         str
#             Formatted statistic string.
#         """
#         fmt_symbol = self.stat_symbol_format.get(symbol, symbol)
#         dp = self.decimal_places_stat
# 
#         if df is not None:
#             return f"{fmt_symbol}({df:.1f}) = {value:.{dp}f}"
#         else:
#             return f"{fmt_symbol} = {value:.{dp}f}"
# 
#     def format_p(self, p_value: float) -> str:
#         """
#         Format a p-value.
# 
#         Parameters
#         ----------
#         p_value : float
#             P-value to format.
# 
#         Returns
#         -------
#         str
#             Formatted p-value string.
#         """
#         p_symbol = self.stat_symbol_format.get("p", "p")
#         dp = self.decimal_places_p
# 
#         # Handle very small p-values
#         if p_value < 0.001:
#             return f"{p_symbol} < 0.001"
#         elif p_value < 0.0001:
#             return f"{p_symbol} < 0.0001"
#         else:
#             return f"{p_symbol} = {p_value:.{dp}f}"
# 
#     def format_effect(self, name: str, value: float) -> str:
#         """
#         Format an effect size.
# 
#         Parameters
#         ----------
#         name : str
#             Effect size name (e.g., "cohens_d_ind").
#         value : float
#             Effect size value.
# 
#         Returns
#         -------
#         str
#             Formatted effect size string.
#         """
#         label = self.effect_label_format.get(name, name)
#         dp = self.decimal_places_effect
#         return f"{label} = {value:.{dp}f}"
# 
#     def format_n(self, group: str, n: int) -> str:
#         """
#         Format a sample size.
# 
#         Parameters
#         ----------
#         group : str
#             Group name/label.
#         n : int
#             Sample size.
# 
#         Returns
#         -------
#         str
#             Formatted sample size string.
#         """
#         # Handle formats with and without group name placeholder
#         if "%s" in self.n_format:
#             return self.n_format % (group, n)
#         else:
#             # Format doesn't include group name, just use n
#             return self.n_format % n
# 
#     def p_to_stars(self, p_value: float) -> str:
#         """
#         Convert p-value to significance stars.
# 
#         Parameters
#         ----------
#         p_value : float
#             P-value.
# 
#         Returns
#         -------
#         str
#             Stars string ("***", "**", "*", or "ns").
#         """
#         if p_value is None:
#             return "ns"
# 
#         for threshold, stars in self.alpha_thresholds:
#             if p_value < threshold:
#                 return stars
#         return "ns"
# 
# 
# # =============================================================================
# # APA Style (LaTeX)
# # =============================================================================
# 
# APA_LATEX_STYLE = StatStyle(
#     id="apa_latex",
#     label="APA (LaTeX)",
#     target="latex",
#     stat_symbol_format={
#         "t": "\\mathit{t}",
#         "F": "\\mathit{F}",
#         "chi2": "\\chi^2",
#         "U": "\\mathit{U}",
#         "W": "\\mathit{W}",
#         "BM": "\\mathit{BM}",
#         "r": "\\mathit{r}",
#         "n": "\\mathit{n}",
#         "p": "\\mathit{p}",
#     },
#     p_format="\\mathit{p} = {p:.3f}",
#     alpha_thresholds=[
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's~d",
#         "cohens_d_paired": "Cohen's~d",
#         "hedges_g": "Hedges'~g",
#         "cliffs_delta": "Cliff's~$\\delta$",
#         "eta_squared": "$\\eta^2$",
#         "partial_eta_squared": "$\\eta^2_{p}$",
#         "effect_size_r": "\\mathit{r}",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     n_format="\\mathit{n}_{%s} = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # APA Style (HTML)
# # =============================================================================
# 
# APA_HTML_STYLE = StatStyle(
#     id="apa_html",
#     label="APA (HTML)",
#     target="html",
#     stat_symbol_format={
#         "t": "<i>t</i>",
#         "F": "<i>F</i>",
#         "chi2": "<i>&chi;</i><sup>2</sup>",
#         "U": "<i>U</i>",
#         "W": "<i>W</i>",
#         "BM": "<i>BM</i>",
#         "r": "<i>r</i>",
#         "n": "<i>n</i>",
#         "p": "<i>p</i>",
#     },
#     p_format="<i>p</i> = {p:.3f}",
#     alpha_thresholds=[
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's d",
#         "cohens_d_paired": "Cohen's d",
#         "hedges_g": "Hedges' g",
#         "cliffs_delta": "Cliff's &delta;",
#         "eta_squared": "&eta;<sup>2</sup>",
#         "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
#         "effect_size_r": "<i>r</i>",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     n_format="<i>n</i><sub>%s</sub> = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Nature Style (LaTeX)
# # =============================================================================
# 
# NATURE_LATEX_STYLE = StatStyle(
#     id="nature_latex",
#     label="Nature (LaTeX)",
#     target="latex",
#     stat_symbol_format={
#         "t": "\\mathit{t}",
#         "F": "\\mathit{F}",
#         "chi2": "\\chi^2",
#         "U": "\\mathit{U}",
#         "W": "\\mathit{W}",
#         "BM": "\\mathit{BM}",
#         "r": "\\mathit{r}",
#         "n": "\\mathit{n}",
#         # Nature often uses uppercase P
#         "p": "\\mathit{P}",
#     },
#     # 3 significant figures
#     p_format="\\mathit{P} = {p:.3g}",
#     alpha_thresholds=[
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's~d",
#         "cohens_d_paired": "Cohen's~d",
#         "hedges_g": "Hedges'~g",
#         "cliffs_delta": "Cliff's~$\\delta$",
#         "eta_squared": "$\\eta^2$",
#         "partial_eta_squared": "$\\eta^2_{p}$",
#         "effect_size_r": "\\mathit{r}",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     # Nature often shows total n rather than per-group
#     n_format="\\mathit{n} = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Nature Style (HTML)
# # =============================================================================
# 
# NATURE_HTML_STYLE = StatStyle(
#     id="nature_html",
#     label="Nature (HTML)",
#     target="html",
#     stat_symbol_format={
#         "t": "<i>t</i>",
#         "F": "<i>F</i>",
#         "chi2": "<i>&chi;</i><sup>2</sup>",
#         "U": "<i>U</i>",
#         "W": "<i>W</i>",
#         "BM": "<i>BM</i>",
#         "r": "<i>r</i>",
#         "n": "<i>n</i>",
#         "p": "<i>P</i>",  # Capital P
#     },
#     p_format="<i>P</i> = {p:.3g}",
#     alpha_thresholds=[
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's d",
#         "cohens_d_paired": "Cohen's d",
#         "hedges_g": "Hedges' g",
#         "cliffs_delta": "Cliff's &delta;",
#         "eta_squared": "&eta;<sup>2</sup>",
#         "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
#         "effect_size_r": "<i>r</i>",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     n_format="<i>n</i> = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Cell Press Style (LaTeX)
# # =============================================================================
# 
# CELL_LATEX_STYLE = StatStyle(
#     id="cell_latex",
#     label="Cell (LaTeX)",
#     target="latex",
#     stat_symbol_format={
#         "t": "\\mathit{t}",
#         "F": "\\mathit{F}",
#         "chi2": "\\chi^2",
#         "U": "\\mathit{U}",
#         "W": "\\mathit{W}",
#         "BM": "\\mathit{BM}",
#         "r": "\\mathit{r}",
#         "n": "\\mathit{n}",
#         "p": "\\mathit{p}",
#     },
#     p_format="\\mathit{p} = {p:.3g}",
#     # Cell often uses **** for p < 0.0001
#     alpha_thresholds=[
#         (0.0001, "****"),
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's~d",
#         "cohens_d_paired": "Cohen's~d",
#         "hedges_g": "Hedges'~g",
#         "cliffs_delta": "Cliff's~$\\delta$",
#         "eta_squared": "$\\eta^2$",
#         "partial_eta_squared": "$\\eta^2_{p}$",
#         "effect_size_r": "\\mathit{r}",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     # Cell often uses "n = X cells from Y mice" style
#     n_format="\\mathit{n} = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Cell Press Style (HTML)
# # =============================================================================
# 
# CELL_HTML_STYLE = StatStyle(
#     id="cell_html",
#     label="Cell (HTML)",
#     target="html",
#     stat_symbol_format={
#         "t": "<i>t</i>",
#         "F": "<i>F</i>",
#         "chi2": "<i>&chi;</i><sup>2</sup>",
#         "U": "<i>U</i>",
#         "W": "<i>W</i>",
#         "BM": "<i>BM</i>",
#         "r": "<i>r</i>",
#         "n": "<i>n</i>",
#         "p": "<i>p</i>",
#     },
#     p_format="<i>p</i> = {p:.3g}",
#     alpha_thresholds=[
#         (0.0001, "****"),
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's d",
#         "cohens_d_paired": "Cohen's d",
#         "hedges_g": "Hedges' g",
#         "cliffs_delta": "Cliff's &delta;",
#         "eta_squared": "&eta;<sup>2</sup>",
#         "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
#         "effect_size_r": "<i>r</i>",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     n_format="<i>n</i> = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Elsevier Style (LaTeX)
# # =============================================================================
# 
# ELSEVIER_LATEX_STYLE = StatStyle(
#     id="elsevier_latex",
#     label="Elsevier (LaTeX)",
#     target="latex",
#     stat_symbol_format={
#         "t": "\\mathit{t}",
#         "F": "\\mathit{F}",
#         "chi2": "\\chi^2",
#         "U": "\\mathit{U}",
#         "W": "\\mathit{W}",
#         "BM": "\\mathit{BM}",
#         "r": "\\mathit{r}",
#         "n": "\\mathit{n}",
#         "p": "\\mathit{p}",
#     },
#     p_format="\\mathit{p} = {p:.3f}",
#     alpha_thresholds=[
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's~d",
#         "cohens_d_paired": "Cohen's~d",
#         "hedges_g": "Hedges'~g",
#         "cliffs_delta": "Cliff's~$\\delta$",
#         "eta_squared": "$\\eta^2$",
#         "partial_eta_squared": "$\\eta^2_{p}$",
#         "effect_size_r": "\\mathit{r}",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     n_format="\\mathit{n} = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Elsevier Style (HTML)
# # =============================================================================
# 
# ELSEVIER_HTML_STYLE = StatStyle(
#     id="elsevier_html",
#     label="Elsevier (HTML)",
#     target="html",
#     stat_symbol_format={
#         "t": "<i>t</i>",
#         "F": "<i>F</i>",
#         "chi2": "<i>&chi;</i><sup>2</sup>",
#         "U": "<i>U</i>",
#         "W": "<i>W</i>",
#         "BM": "<i>BM</i>",
#         "r": "<i>r</i>",
#         "n": "<i>n</i>",
#         "p": "<i>p</i>",
#     },
#     p_format="<i>p</i> = {p:.3f}",
#     alpha_thresholds=[
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's d",
#         "cohens_d_paired": "Cohen's d",
#         "hedges_g": "Hedges' g",
#         "cliffs_delta": "Cliff's &delta;",
#         "eta_squared": "&eta;<sup>2</sup>",
#         "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
#         "effect_size_r": "<i>r</i>",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     n_format="<i>n</i> = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Plain Text Style
# # =============================================================================
# 
# PLAIN_STYLE = StatStyle(
#     id="plain",
#     label="Plain Text",
#     target="plain",
#     stat_symbol_format={
#         "t": "t",
#         "F": "F",
#         "chi2": "chi2",
#         "U": "U",
#         "W": "W",
#         "BM": "BM",
#         "r": "r",
#         "n": "n",
#         "p": "p",
#     },
#     p_format="p = {p:.3f}",
#     alpha_thresholds=[
#         (0.001, "***"),
#         (0.01, "**"),
#         (0.05, "*"),
#     ],
#     effect_label_format={
#         "cohens_d_ind": "Cohen's d",
#         "cohens_d_paired": "Cohen's d",
#         "hedges_g": "Hedges' g",
#         "cliffs_delta": "Cliff's delta",
#         "eta_squared": "eta^2",
#         "partial_eta_squared": "partial eta^2",
#         "effect_size_r": "r",
#         "odds_ratio": "OR",
#         "risk_ratio": "RR",
#         "prob_superiority": "P(X>Y)",
#     },
#     n_format="n_%s = %d",
#     decimal_places_p=3,
#     decimal_places_stat=2,
#     decimal_places_effect=2,
# )
# 
# 
# # =============================================================================
# # Style Registry
# # =============================================================================
# 
# STAT_STYLES: Dict[str, StatStyle] = {
#     "apa_latex": APA_LATEX_STYLE,
#     "apa_html": APA_HTML_STYLE,
#     "nature_latex": NATURE_LATEX_STYLE,
#     "nature_html": NATURE_HTML_STYLE,
#     "cell_latex": CELL_LATEX_STYLE,
#     "cell_html": CELL_HTML_STYLE,
#     "elsevier_latex": ELSEVIER_LATEX_STYLE,
#     "elsevier_html": ELSEVIER_HTML_STYLE,
#     "plain": PLAIN_STYLE,
# }
# 
# 
# def get_stat_style(style_id: str) -> StatStyle:
#     """
#     Look up a statistical reporting style by its ID.
# 
#     Parameters
#     ----------
#     style_id : str
#         Style identifier (e.g., "apa_latex", "nature_html").
# 
#     Returns
#     -------
#     StatStyle
#         The requested style, or APA LaTeX as fallback.
# 
#     Examples
#     --------
#     >>> style = get_stat_style("apa_latex")
#     >>> style.label
#     'APA (LaTeX)'
# 
#     >>> style = get_stat_style("unknown")  # Falls back to APA
#     >>> style.id
#     'apa_latex'
#     """
#     return STAT_STYLES.get(style_id, APA_LATEX_STYLE)
# 
# 
# def list_styles(target: Optional[OutputTarget] = None) -> List[str]:
#     """
#     List available style IDs.
# 
#     Parameters
#     ----------
#     target : OutputTarget or None
#         If provided, only list styles for this output format.
# 
#     Returns
#     -------
#     list of str
#         Style IDs.
#     """
#     if target is None:
#         return list(STAT_STYLES.keys())
#     return [
#         sid for sid, style in STAT_STYLES.items()
#         if style.target == target
#     ]
# 
# 
# # =============================================================================
# # Public API
# # =============================================================================
# 
# __all__ = [
#     "StatStyle",
#     "OutputTarget",
#     "STAT_STYLES",
#     "get_stat_style",
#     "list_styles",
#     # Individual styles
#     "APA_LATEX_STYLE",
#     "APA_HTML_STYLE",
#     "NATURE_LATEX_STYLE",
#     "NATURE_HTML_STYLE",
#     "CELL_LATEX_STYLE",
#     "CELL_HTML_STYLE",
#     "ELSEVIER_LATEX_STYLE",
#     "ELSEVIER_HTML_STYLE",
#     "PLAIN_STYLE",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_styles.py
# --------------------------------------------------------------------------------
