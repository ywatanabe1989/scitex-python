#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_styles.py

"""
Journal Style Presets - Publication-ready statistical formatting.

This module defines formatting styles for major journal families:
- APA (American Psychological Association)
- Nature (Nature family journals)
- Cell (Cell Press journals)
- Elsevier (Generic biomedical)

Each style specifies:
- How to format statistics (italic, symbols)
- How to format p-values (thresholds, precision)
- How to format effect sizes
- How to format sample sizes

Supports both LaTeX and HTML output targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any

# =============================================================================
# Type Aliases
# =============================================================================

OutputTarget = Literal["latex", "html", "plain"]


# =============================================================================
# StatStyle
# =============================================================================


@dataclass
class StatStyle:
    """
    Style configuration for statistical reporting.

    Defines how to format statistical results for a specific journal
    or output format.

    Parameters
    ----------
    id : str
        Unique identifier for this style.
    label : str
        Human-readable label (e.g., "APA (LaTeX)").
    target : OutputTarget
        Output format: "latex", "html", or "plain".
    stat_symbol_format : dict
        Maps statistic symbols to their formatted versions.
        Example: {"t": "\\mathit{t}", "p": "\\mathit{p}"}
    p_format : str
        Format string for p-values. Use {p:.3f} syntax.
    alpha_thresholds : list of (float, str)
        P-value thresholds for stars. Lower threshold = more stars.
        Example: [(0.001, "***"), (0.01, "**"), (0.05, "*")]
    effect_label_format : dict
        Maps effect size names to their formatted labels.
    n_format : str
        Format string for sample sizes. Uses % formatting.
        Example: "\\mathit{n}_{%s} = %d"
    decimal_places_p : int
        Decimal places for p-values.
    decimal_places_stat : int
        Decimal places for test statistics.
    decimal_places_effect : int
        Decimal places for effect sizes.

    Examples
    --------
    >>> style = STAT_STYLES["apa_latex"]
    >>> style.format_p(0.032)
    '\\\\mathit{p} = 0.032'
    >>> style.format_stat("t", 2.31, df=28)
    '\\\\mathit{t}(28.0) = 2.31'
    """

    id: str
    label: str
    target: OutputTarget

    # Symbol formatting
    stat_symbol_format: Dict[str, str] = field(default_factory=dict)

    # P-value formatting
    p_format: str = "p = {p:.3f}"
    alpha_thresholds: List[Tuple[float, str]] = field(default_factory=list)

    # Effect size labels
    effect_label_format: Dict[str, str] = field(default_factory=dict)

    # Sample size formatting
    n_format: str = "n_{%s} = %d"

    # Decimal places
    decimal_places_p: int = 3
    decimal_places_stat: int = 2
    decimal_places_effect: int = 2

    def format_stat(
        self,
        symbol: str,
        value: float,
        df: Optional[float] = None,
    ) -> str:
        """
        Format a test statistic.

        Parameters
        ----------
        symbol : str
            Statistic symbol (e.g., "t", "F", "chi2").
        value : float
            Statistic value.
        df : float, optional
            Degrees of freedom.

        Returns
        -------
        str
            Formatted statistic string.
        """
        fmt_symbol = self.stat_symbol_format.get(symbol, symbol)
        dp = self.decimal_places_stat

        if df is not None:
            return f"{fmt_symbol}({df:.1f}) = {value:.{dp}f}"
        else:
            return f"{fmt_symbol} = {value:.{dp}f}"

    def format_p(self, p_value: float) -> str:
        """
        Format a p-value.

        Parameters
        ----------
        p_value : float
            P-value to format.

        Returns
        -------
        str
            Formatted p-value string.
        """
        p_symbol = self.stat_symbol_format.get("p", "p")
        dp = self.decimal_places_p

        # Handle very small p-values
        if p_value < 0.001:
            return f"{p_symbol} < 0.001"
        elif p_value < 0.0001:
            return f"{p_symbol} < 0.0001"
        else:
            return f"{p_symbol} = {p_value:.{dp}f}"

    def format_effect(self, name: str, value: float) -> str:
        """
        Format an effect size.

        Parameters
        ----------
        name : str
            Effect size name (e.g., "cohens_d_ind").
        value : float
            Effect size value.

        Returns
        -------
        str
            Formatted effect size string.
        """
        label = self.effect_label_format.get(name, name)
        dp = self.decimal_places_effect
        return f"{label} = {value:.{dp}f}"

    def format_n(self, group: str, n: int) -> str:
        """
        Format a sample size.

        Parameters
        ----------
        group : str
            Group name/label.
        n : int
            Sample size.

        Returns
        -------
        str
            Formatted sample size string.
        """
        # Handle formats with and without group name placeholder
        if "%s" in self.n_format:
            return self.n_format % (group, n)
        else:
            # Format doesn't include group name, just use n
            return self.n_format % n

    def p_to_stars(self, p_value: float) -> str:
        """
        Convert p-value to significance stars.

        Parameters
        ----------
        p_value : float
            P-value.

        Returns
        -------
        str
            Stars string ("***", "**", "*", or "ns").
        """
        if p_value is None:
            return "ns"

        for threshold, stars in self.alpha_thresholds:
            if p_value < threshold:
                return stars
        return "ns"


# =============================================================================
# APA Style (LaTeX)
# =============================================================================

APA_LATEX_STYLE = StatStyle(
    id="apa_latex",
    label="APA (LaTeX)",
    target="latex",
    stat_symbol_format={
        "t": "\\mathit{t}",
        "F": "\\mathit{F}",
        "chi2": "\\chi^2",
        "U": "\\mathit{U}",
        "W": "\\mathit{W}",
        "BM": "\\mathit{BM}",
        "r": "\\mathit{r}",
        "n": "\\mathit{n}",
        "p": "\\mathit{p}",
    },
    p_format="\\mathit{p} = {p:.3f}",
    alpha_thresholds=[
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's~d",
        "cohens_d_paired": "Cohen's~d",
        "hedges_g": "Hedges'~g",
        "cliffs_delta": "Cliff's~$\\delta$",
        "eta_squared": "$\\eta^2$",
        "partial_eta_squared": "$\\eta^2_{p}$",
        "effect_size_r": "\\mathit{r}",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    n_format="\\mathit{n}_{%s} = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# APA Style (HTML)
# =============================================================================

APA_HTML_STYLE = StatStyle(
    id="apa_html",
    label="APA (HTML)",
    target="html",
    stat_symbol_format={
        "t": "<i>t</i>",
        "F": "<i>F</i>",
        "chi2": "<i>&chi;</i><sup>2</sup>",
        "U": "<i>U</i>",
        "W": "<i>W</i>",
        "BM": "<i>BM</i>",
        "r": "<i>r</i>",
        "n": "<i>n</i>",
        "p": "<i>p</i>",
    },
    p_format="<i>p</i> = {p:.3f}",
    alpha_thresholds=[
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's d",
        "cohens_d_paired": "Cohen's d",
        "hedges_g": "Hedges' g",
        "cliffs_delta": "Cliff's &delta;",
        "eta_squared": "&eta;<sup>2</sup>",
        "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
        "effect_size_r": "<i>r</i>",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    n_format="<i>n</i><sub>%s</sub> = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Nature Style (LaTeX)
# =============================================================================

NATURE_LATEX_STYLE = StatStyle(
    id="nature_latex",
    label="Nature (LaTeX)",
    target="latex",
    stat_symbol_format={
        "t": "\\mathit{t}",
        "F": "\\mathit{F}",
        "chi2": "\\chi^2",
        "U": "\\mathit{U}",
        "W": "\\mathit{W}",
        "BM": "\\mathit{BM}",
        "r": "\\mathit{r}",
        "n": "\\mathit{n}",
        # Nature often uses uppercase P
        "p": "\\mathit{P}",
    },
    # 3 significant figures
    p_format="\\mathit{P} = {p:.3g}",
    alpha_thresholds=[
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's~d",
        "cohens_d_paired": "Cohen's~d",
        "hedges_g": "Hedges'~g",
        "cliffs_delta": "Cliff's~$\\delta$",
        "eta_squared": "$\\eta^2$",
        "partial_eta_squared": "$\\eta^2_{p}$",
        "effect_size_r": "\\mathit{r}",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    # Nature often shows total n rather than per-group
    n_format="\\mathit{n} = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Nature Style (HTML)
# =============================================================================

NATURE_HTML_STYLE = StatStyle(
    id="nature_html",
    label="Nature (HTML)",
    target="html",
    stat_symbol_format={
        "t": "<i>t</i>",
        "F": "<i>F</i>",
        "chi2": "<i>&chi;</i><sup>2</sup>",
        "U": "<i>U</i>",
        "W": "<i>W</i>",
        "BM": "<i>BM</i>",
        "r": "<i>r</i>",
        "n": "<i>n</i>",
        "p": "<i>P</i>",  # Capital P
    },
    p_format="<i>P</i> = {p:.3g}",
    alpha_thresholds=[
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's d",
        "cohens_d_paired": "Cohen's d",
        "hedges_g": "Hedges' g",
        "cliffs_delta": "Cliff's &delta;",
        "eta_squared": "&eta;<sup>2</sup>",
        "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
        "effect_size_r": "<i>r</i>",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    n_format="<i>n</i> = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Cell Press Style (LaTeX)
# =============================================================================

CELL_LATEX_STYLE = StatStyle(
    id="cell_latex",
    label="Cell (LaTeX)",
    target="latex",
    stat_symbol_format={
        "t": "\\mathit{t}",
        "F": "\\mathit{F}",
        "chi2": "\\chi^2",
        "U": "\\mathit{U}",
        "W": "\\mathit{W}",
        "BM": "\\mathit{BM}",
        "r": "\\mathit{r}",
        "n": "\\mathit{n}",
        "p": "\\mathit{p}",
    },
    p_format="\\mathit{p} = {p:.3g}",
    # Cell often uses **** for p < 0.0001
    alpha_thresholds=[
        (0.0001, "****"),
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's~d",
        "cohens_d_paired": "Cohen's~d",
        "hedges_g": "Hedges'~g",
        "cliffs_delta": "Cliff's~$\\delta$",
        "eta_squared": "$\\eta^2$",
        "partial_eta_squared": "$\\eta^2_{p}$",
        "effect_size_r": "\\mathit{r}",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    # Cell often uses "n = X cells from Y mice" style
    n_format="\\mathit{n} = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Cell Press Style (HTML)
# =============================================================================

CELL_HTML_STYLE = StatStyle(
    id="cell_html",
    label="Cell (HTML)",
    target="html",
    stat_symbol_format={
        "t": "<i>t</i>",
        "F": "<i>F</i>",
        "chi2": "<i>&chi;</i><sup>2</sup>",
        "U": "<i>U</i>",
        "W": "<i>W</i>",
        "BM": "<i>BM</i>",
        "r": "<i>r</i>",
        "n": "<i>n</i>",
        "p": "<i>p</i>",
    },
    p_format="<i>p</i> = {p:.3g}",
    alpha_thresholds=[
        (0.0001, "****"),
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's d",
        "cohens_d_paired": "Cohen's d",
        "hedges_g": "Hedges' g",
        "cliffs_delta": "Cliff's &delta;",
        "eta_squared": "&eta;<sup>2</sup>",
        "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
        "effect_size_r": "<i>r</i>",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    n_format="<i>n</i> = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Elsevier Style (LaTeX)
# =============================================================================

ELSEVIER_LATEX_STYLE = StatStyle(
    id="elsevier_latex",
    label="Elsevier (LaTeX)",
    target="latex",
    stat_symbol_format={
        "t": "\\mathit{t}",
        "F": "\\mathit{F}",
        "chi2": "\\chi^2",
        "U": "\\mathit{U}",
        "W": "\\mathit{W}",
        "BM": "\\mathit{BM}",
        "r": "\\mathit{r}",
        "n": "\\mathit{n}",
        "p": "\\mathit{p}",
    },
    p_format="\\mathit{p} = {p:.3f}",
    alpha_thresholds=[
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's~d",
        "cohens_d_paired": "Cohen's~d",
        "hedges_g": "Hedges'~g",
        "cliffs_delta": "Cliff's~$\\delta$",
        "eta_squared": "$\\eta^2$",
        "partial_eta_squared": "$\\eta^2_{p}$",
        "effect_size_r": "\\mathit{r}",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    n_format="\\mathit{n} = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Elsevier Style (HTML)
# =============================================================================

ELSEVIER_HTML_STYLE = StatStyle(
    id="elsevier_html",
    label="Elsevier (HTML)",
    target="html",
    stat_symbol_format={
        "t": "<i>t</i>",
        "F": "<i>F</i>",
        "chi2": "<i>&chi;</i><sup>2</sup>",
        "U": "<i>U</i>",
        "W": "<i>W</i>",
        "BM": "<i>BM</i>",
        "r": "<i>r</i>",
        "n": "<i>n</i>",
        "p": "<i>p</i>",
    },
    p_format="<i>p</i> = {p:.3f}",
    alpha_thresholds=[
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's d",
        "cohens_d_paired": "Cohen's d",
        "hedges_g": "Hedges' g",
        "cliffs_delta": "Cliff's &delta;",
        "eta_squared": "&eta;<sup>2</sup>",
        "partial_eta_squared": "&eta;<sup>2</sup><sub>p</sub>",
        "effect_size_r": "<i>r</i>",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    n_format="<i>n</i> = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Plain Text Style
# =============================================================================

PLAIN_STYLE = StatStyle(
    id="plain",
    label="Plain Text",
    target="plain",
    stat_symbol_format={
        "t": "t",
        "F": "F",
        "chi2": "chi2",
        "U": "U",
        "W": "W",
        "BM": "BM",
        "r": "r",
        "n": "n",
        "p": "p",
    },
    p_format="p = {p:.3f}",
    alpha_thresholds=[
        (0.001, "***"),
        (0.01, "**"),
        (0.05, "*"),
    ],
    effect_label_format={
        "cohens_d_ind": "Cohen's d",
        "cohens_d_paired": "Cohen's d",
        "hedges_g": "Hedges' g",
        "cliffs_delta": "Cliff's delta",
        "eta_squared": "eta^2",
        "partial_eta_squared": "partial eta^2",
        "effect_size_r": "r",
        "odds_ratio": "OR",
        "risk_ratio": "RR",
        "prob_superiority": "P(X>Y)",
    },
    n_format="n_%s = %d",
    decimal_places_p=3,
    decimal_places_stat=2,
    decimal_places_effect=2,
)


# =============================================================================
# Style Registry
# =============================================================================

STAT_STYLES: Dict[str, StatStyle] = {
    "apa_latex": APA_LATEX_STYLE,
    "apa_html": APA_HTML_STYLE,
    "nature_latex": NATURE_LATEX_STYLE,
    "nature_html": NATURE_HTML_STYLE,
    "cell_latex": CELL_LATEX_STYLE,
    "cell_html": CELL_HTML_STYLE,
    "elsevier_latex": ELSEVIER_LATEX_STYLE,
    "elsevier_html": ELSEVIER_HTML_STYLE,
    "plain": PLAIN_STYLE,
}


def get_stat_style(style_id: str) -> StatStyle:
    """
    Look up a statistical reporting style by its ID.

    Parameters
    ----------
    style_id : str
        Style identifier (e.g., "apa_latex", "nature_html").

    Returns
    -------
    StatStyle
        The requested style, or APA LaTeX as fallback.

    Examples
    --------
    >>> style = get_stat_style("apa_latex")
    >>> style.label
    'APA (LaTeX)'

    >>> style = get_stat_style("unknown")  # Falls back to APA
    >>> style.id
    'apa_latex'
    """
    return STAT_STYLES.get(style_id, APA_LATEX_STYLE)


def list_styles(target: Optional[OutputTarget] = None) -> List[str]:
    """
    List available style IDs.

    Parameters
    ----------
    target : OutputTarget or None
        If provided, only list styles for this output format.

    Returns
    -------
    list of str
        Style IDs.
    """
    if target is None:
        return list(STAT_STYLES.keys())
    return [
        sid for sid, style in STAT_STYLES.items()
        if style.target == target
    ]


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "StatStyle",
    "OutputTarget",
    "STAT_STYLES",
    "get_stat_style",
    "list_styles",
    # Individual styles
    "APA_LATEX_STYLE",
    "APA_HTML_STYLE",
    "NATURE_LATEX_STYLE",
    "NATURE_HTML_STYLE",
    "CELL_LATEX_STYLE",
    "CELL_HTML_STYLE",
    "ELSEVIER_LATEX_STYLE",
    "ELSEVIER_HTML_STYLE",
    "PLAIN_STYLE",
]

# EOF
