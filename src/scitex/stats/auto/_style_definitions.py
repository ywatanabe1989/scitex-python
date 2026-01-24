#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/stats/auto/_style_definitions.py

"""Journal style preset definitions for statistical formatting."""

from ._stat_style import StatStyle

__all__ = [
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
        "p": "\\mathit{P}",  # Nature uses uppercase P
    },
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
        "p": "<i>P</i>",
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


# EOF
