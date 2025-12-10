#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_formatting.py

"""
Statistical Formatting - Publication-ready output generation.

This module provides functions for:
- Computing summary statistics per group
- Formatting complete test result lines
- Converting results for Inspector panel display
- Handling multiple comparison corrections

All formatting respects journal style presets (APA, Nature, Cell, Elsevier).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, TypedDict, Union

import numpy as np

from ._styles import StatStyle, get_stat_style


# =============================================================================
# Type Definitions
# =============================================================================


class SummaryStatsDict(TypedDict, total=False):
    """
    Summary statistics for a single group.

    Attributes
    ----------
    group : str
        Group name/label.
    n : int
        Sample size.
    mean : float or None
        Mean value.
    sd : float or None
        Standard deviation.
    sem : float or None
        Standard error of mean.
    median : float or None
        Median value.
    iqr : float or None
        Interquartile range.
    q1 : float or None
        First quartile (25th percentile).
    q3 : float or None
        Third quartile (75th percentile).
    minimum : float or None
        Minimum value.
    maximum : float or None
        Maximum value.
    """
    group: str
    n: int
    mean: Optional[float]
    sd: Optional[float]
    sem: Optional[float]
    median: Optional[float]
    iqr: Optional[float]
    q1: Optional[float]
    q3: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]


class TestResultDict(TypedDict, total=False):
    """
    Result structure for a single statistical test.

    Attributes
    ----------
    test_name : str
        Internal test name ("ttest_ind", "brunner_munzel", etc.).
    p_raw : float or None
        Raw p-value.
    p_adj : float or None
        Adjusted p-value after multiple correction.
    stat : float or None
        Test statistic value.
    df : float or None
        Degrees of freedom.
    method : str or None
        Human-readable method label.
    correction_method : str or None
        Multiple correction method used ("bonferroni", "fdr_bh", etc.).
    details : dict
        Additional test-specific information.
    """
    test_name: str
    p_raw: Optional[float]
    p_adj: Optional[float]
    stat: Optional[float]
    df: Optional[float]
    method: Optional[str]
    correction_method: Optional[str]
    details: Dict[str, Any]


class EffectResultDict(TypedDict, total=False):
    """
    Result structure for a single effect size measure.

    Attributes
    ----------
    name : str
        Internal name ("cohens_d_ind", "eta_squared", etc.).
    label : str
        Human-readable label.
    value : float
        Effect size value.
    ci_lower : float or None
        Lower bound of confidence interval.
    ci_upper : float or None
        Upper bound of confidence interval.
    note : str or None
        Interpretation note (e.g., "small", "medium", "large").
    """
    name: str
    label: str
    value: float
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    note: Optional[str]


# =============================================================================
# Statistic Symbol Mapping
# =============================================================================

_STAT_SYMBOLS: Dict[str, str] = {
    # Parametric
    "ttest_ind": "t",
    "ttest_rel": "t",
    "anova_oneway": "F",
    "anova_rm_oneway": "F",
    "anova_twoway": "F",
    "anova_twoway_mixed": "F",
    "welch_anova": "F",

    # Nonparametric
    "brunner_munzel": "BM",
    "mannwhitneyu": "U",
    "wilcoxon": "W",
    "kruskal": "H",
    "friedman": "chi2",

    # Categorical
    "chi2_independence": "chi2",
    "fisher_exact": "OR",
    "mcnemar": "chi2",

    # Correlation
    "pearsonr": "r",
    "spearmanr": "r",

    # Normality
    "shapiro": "W",
    "levene": "F",
}


def get_stat_symbol(test_name: str) -> str:
    """
    Get the statistic symbol for a test.

    Parameters
    ----------
    test_name : str
        Internal test name.

    Returns
    -------
    str
        Statistic symbol (e.g., "t", "F", "BM").
    """
    return _STAT_SYMBOLS.get(test_name, "stat")


# =============================================================================
# Summary Statistics
# =============================================================================


def compute_summary_stats(
    y: np.ndarray,
    group: np.ndarray,
) -> List[SummaryStatsDict]:
    """
    Compute per-group summary statistics.

    Parameters
    ----------
    y : np.ndarray
        Outcome values.
    group : np.ndarray
        Group labels for each observation.

    Returns
    -------
    list of SummaryStatsDict
        Summary statistics for each group.

    Examples
    --------
    >>> y = np.array([1, 2, 3, 4, 5, 6])
    >>> group = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
    >>> stats = compute_summary_stats(y, group)
    >>> stats[0]['group']
    'A'
    >>> stats[0]['n']
    3
    """
    y = np.asarray(y, dtype=float)
    group = np.asarray(group)

    stats_list: List[SummaryStatsDict] = []

    for group_value in np.unique(group):
        mask = group == group_value
        vals = y[mask]
        vals = vals[~np.isnan(vals)]  # Remove NaN

        if vals.size == 0:
            continue

        n = int(vals.size)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = float(sd / np.sqrt(n)) if n > 1 else 0.0

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1

        stats_list.append(
            SummaryStatsDict(
                group=str(group_value),
                n=n,
                mean=mean,
                sd=sd,
                sem=sem,
                median=float(med),
                iqr=float(iqr),
                q1=float(q1),
                q3=float(q3),
                minimum=float(vals.min()),
                maximum=float(vals.max()),
            )
        )

    return stats_list


def compute_summary_from_groups(
    groups: List[np.ndarray],
    group_names: Optional[List[str]] = None,
) -> List[SummaryStatsDict]:
    """
    Compute summary statistics from a list of group arrays.

    Parameters
    ----------
    groups : list of np.ndarray
        List of arrays, one per group.
    group_names : list of str, optional
        Names for each group.

    Returns
    -------
    list of SummaryStatsDict
        Summary statistics for each group.
    """
    if group_names is None:
        group_names = [f"Group_{i+1}" for i in range(len(groups))]

    stats_list: List[SummaryStatsDict] = []

    for name, vals in zip(group_names, groups):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            continue

        n = int(vals.size)
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = float(sd / np.sqrt(n)) if n > 1 else 0.0

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1

        stats_list.append(
            SummaryStatsDict(
                group=name,
                n=n,
                mean=mean,
                sd=sd,
                sem=sem,
                median=float(med),
                iqr=float(iqr),
                q1=float(q1),
                q3=float(q3),
                minimum=float(vals.min()),
                maximum=float(vals.max()),
            )
        )

    return stats_list


# =============================================================================
# Test Line Formatting
# =============================================================================


def format_test_line(
    test: TestResultDict,
    effects: Optional[List[EffectResultDict]] = None,
    summary: Optional[List[SummaryStatsDict]] = None,
    style: Optional[Union[str, StatStyle]] = None,
    include_n: bool = True,
    max_effects: int = 2,
) -> str:
    """
    Format a complete statistical result line.

    Produces publication-ready formatted text with proper italics,
    symbols, and formatting according to the specified journal style.

    Parameters
    ----------
    test : TestResultDict
        Test result dictionary.
    effects : list of EffectResultDict, optional
        Effect size results to include.
    summary : list of SummaryStatsDict, optional
        Summary statistics for sample size display.
    style : str or StatStyle, optional
        Style to use. Can be style ID or StatStyle instance.
        Defaults to APA LaTeX.
    include_n : bool
        Whether to include sample sizes in output.
    max_effects : int
        Maximum number of effect sizes to include.

    Returns
    -------
    str
        Formatted result line.

    Examples
    --------
    >>> test = {"test_name": "ttest_ind", "stat": 2.31, "df": 28.0, "p_raw": 0.028}
    >>> effects = [{"name": "cohens_d_ind", "value": 0.72, "label": "Cohen's d"}]
    >>> summary = [{"group": "A", "n": 15}, {"group": "B", "n": 15}]
    >>> line = format_test_line(test, effects, summary, style="apa_latex")
    >>> "\\mathit{t}" in line
    True
    """
    # Get style
    if style is None:
        style = get_stat_style("apa_latex")
    elif isinstance(style, str):
        style = get_stat_style(style)

    parts: List[str] = []

    # Format test statistic
    test_name = test.get("test_name", "")
    stat = test.get("stat")
    df = test.get("df")

    if stat is not None:
        symbol = get_stat_symbol(test_name)
        stat_part = style.format_stat(symbol, stat, df)
        parts.append(stat_part)

    # Format p-value
    p = test.get("p_adj") or test.get("p_raw")
    if p is not None:
        p_part = style.format_p(p)
        parts.append(p_part)

    # Format effect sizes
    if effects:
        for eff in effects[:max_effects]:
            eff_name = eff.get("name", "")
            eff_value = eff.get("value")
            if eff_value is not None:
                eff_part = style.format_effect(eff_name, eff_value)
                parts.append(eff_part)

    # Format sample sizes
    if include_n and summary:
        for s in summary:
            group_name = str(s.get("group", ""))
            n_value = int(s.get("n", 0))
            n_part = style.format_n(group_name, n_value)
            parts.append(n_part)

    return ", ".join(parts)


def format_test_line_compact(
    test: TestResultDict,
    style: Optional[Union[str, StatStyle]] = None,
) -> str:
    """
    Format a compact test result (statistic + p-value only).

    Parameters
    ----------
    test : TestResultDict
        Test result dictionary.
    style : str or StatStyle, optional
        Style to use.

    Returns
    -------
    str
        Compact formatted result.
    """
    return format_test_line(
        test,
        effects=None,
        summary=None,
        style=style,
        include_n=False,
    )


# =============================================================================
# Inspector Panel Formatting
# =============================================================================


def format_for_inspector(
    test_results: List[TestResultDict],
    effect_results: Optional[List[EffectResultDict]] = None,
) -> Dict[str, List[Dict]]:
    """
    Format results for Inspector panel display.

    Produces a structure suitable for displaying in a UI panel
    with tables for tests and effect sizes.

    Parameters
    ----------
    test_results : list of TestResultDict
        Test results to display.
    effect_results : list of EffectResultDict, optional
        Effect size results to display.

    Returns
    -------
    dict
        Dictionary with 'tests' and 'effects' lists.

    Examples
    --------
    >>> tests = [{"test_name": "ttest_ind", "p_raw": 0.03, "stat": 2.2}]
    >>> effects = [{"name": "cohens_d_ind", "value": 0.8, "label": "Cohen's d"}]
    >>> result = format_for_inspector(tests, effects)
    >>> len(result["tests"])
    1
    """
    from ._selector import _pretty_label

    return {
        "tests": [
            {
                "name": r.get("test_name"),
                "label": _pretty_label(r.get("test_name", "")),
                "p_raw": r.get("p_raw"),
                "p_adj": r.get("p_adj"),
                "stat": r.get("stat"),
                "df": r.get("df"),
                "method": r.get("method"),
                "correction": r.get("correction_method"),
                "details": r.get("details", {}),
            }
            for r in test_results
        ],
        "effects": [
            {
                "name": e.get("name"),
                "label": e.get("label", e.get("name", "")),
                "value": e.get("value"),
                "ci_lower": e.get("ci_lower"),
                "ci_upper": e.get("ci_upper"),
                "note": e.get("note"),
            }
            for e in (effect_results or [])
        ],
    }


# =============================================================================
# P-value to Stars
# =============================================================================


def p_to_stars(
    p_value: Optional[float],
    style: Optional[Union[str, StatStyle]] = None,
) -> str:
    """
    Convert p-value to significance stars.

    Uses the alpha thresholds from the specified style.

    Parameters
    ----------
    p_value : float or None
        P-value to convert.
    style : str or StatStyle, optional
        Style to use for thresholds.

    Returns
    -------
    str
        Stars string ("***", "**", "*", or "ns").

    Examples
    --------
    >>> p_to_stars(0.001)
    '***'
    >>> p_to_stars(0.03)
    '*'
    >>> p_to_stars(0.10)
    'ns'
    """
    if style is None:
        style = get_stat_style("apa_latex")
    elif isinstance(style, str):
        style = get_stat_style(style)

    return style.p_to_stars(p_value)


# =============================================================================
# Multiple Comparison Correction
# =============================================================================


CorrectionMethod = Union[str, None]


def apply_multiple_correction(
    results: List[TestResultDict],
    method: CorrectionMethod = "fdr_bh",
) -> List[TestResultDict]:
    """
    Apply multiple-comparison correction to test results.

    Modifies results in-place by setting p_adj and correction_method.

    Parameters
    ----------
    results : list of TestResultDict
        Test results with p_raw values.
    method : str or None
        Correction method:
        - "none": No correction (p_adj = p_raw)
        - "bonferroni": Bonferroni correction
        - "holm": Holm-Bonferroni step-down
        - "fdr_bh": Benjamini-Hochberg FDR

    Returns
    -------
    list of TestResultDict
        Results with p_adj filled in.

    Examples
    --------
    >>> results = [
    ...     {"test_name": "t1", "p_raw": 0.01},
    ...     {"test_name": "t2", "p_raw": 0.03},
    ...     {"test_name": "t3", "p_raw": 0.04},
    ... ]
    >>> corrected = apply_multiple_correction(results, "bonferroni")
    >>> corrected[0]["p_adj"]
    0.03
    """
    if method is None or method == "none":
        for r in results:
            r["p_adj"] = r.get("p_raw")
            r["correction_method"] = "none"
        return results

    # Get valid p-values
    valid_indices = [
        i for i, r in enumerate(results)
        if r.get("p_raw") is not None
    ]

    if not valid_indices:
        return results

    p_values = [results[i]["p_raw"] for i in valid_indices]
    m = len(p_values)

    adjusted: List[float] = []

    if method == "bonferroni":
        adjusted = [min(p * m, 1.0) for p in p_values]

    elif method == "holm":
        # Holm-Bonferroni step-down
        sorted_idx = sorted(range(m), key=lambda i: p_values[i])
        adj = [0.0] * m
        cummax = 0.0
        for rank, idx in enumerate(sorted_idx, start=1):
            adj_val = min((m - rank + 1) * p_values[idx], 1.0)
            adj_val = max(adj_val, cummax)  # Enforce monotonicity
            adj[idx] = adj_val
            cummax = adj_val
        adjusted = adj

    elif method == "fdr_bh":
        # Benjamini-Hochberg
        sorted_idx = sorted(range(m), key=lambda i: p_values[i])
        adj = [0.0] * m
        prev = 1.0
        for rank in range(m, 0, -1):
            idx = sorted_idx[rank - 1]
            p = p_values[idx]
            bh = p * m / rank
            val = min(bh, prev, 1.0)
            adj[idx] = val
            prev = val
        adjusted = adj

    else:
        # Unknown method - no correction
        adjusted = p_values

    # Write back
    for local_i, global_i in enumerate(valid_indices):
        results[global_i]["p_adj"] = float(adjusted[local_i])
        results[global_i]["correction_method"] = method

    return results


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Type definitions
    "SummaryStatsDict",
    "TestResultDict",
    "EffectResultDict",
    "CorrectionMethod",
    # Summary statistics
    "compute_summary_stats",
    "compute_summary_from_groups",
    # Symbol mapping
    "get_stat_symbol",
    # Formatting
    "format_test_line",
    "format_test_line_compact",
    "format_for_inspector",
    "p_to_stars",
    # Correction
    "apply_multiple_correction",
]

# EOF
