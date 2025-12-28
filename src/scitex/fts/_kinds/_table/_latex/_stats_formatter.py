#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_tables/_latex/_stats_formatter.py

"""Format statistical results for LaTeX output."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ._utils import format_effect_size, format_p_value, format_statistic, significance_stars


@dataclass
class FormattedStat:
    """A formatted statistical result.

    Attributes:
        test_type: Type of test (t-test, anova, etc.)
        statistic: Formatted test statistic (e.g., "$t(48) = 2.31$")
        p_value: Formatted p-value (e.g., ".023")
        effect_size: Formatted effect size if available
        stars: Significance stars
        full_report: Complete inline report
    """

    test_type: str
    statistic: str
    p_value: str
    effect_size: Optional[str] = None
    stars: str = ""
    full_report: str = ""


def format_stats_for_latex(stats_dict: Dict[str, Any]) -> List[FormattedStat]:
    """Format stats.json contents for LaTeX.

    Args:
        stats_dict: Dictionary from stats.json

    Returns:
        List of FormattedStat objects
    """
    results = []

    analyses = stats_dict.get("analyses", [])
    for analysis in analyses:
        formatted = _format_analysis(analysis)
        if formatted:
            results.append(formatted)

    return results


def _format_analysis(analysis: Dict[str, Any]) -> Optional[FormattedStat]:
    """Format a single analysis result.

    Args:
        analysis: Single analysis dictionary

    Returns:
        FormattedStat or None if cannot format
    """
    method = analysis.get("method", {})
    result = analysis.get("results", {})

    if not result:
        return None

    # Extract test info
    test_name = method.get("name", "test")
    variant = method.get("variant", "")

    # Format statistic
    stat_name = result.get("statistic_name", "t")
    stat_value = result.get("statistic")
    df = result.get("df")

    if stat_value is None:
        return None

    statistic_str = format_statistic(stat_name, stat_value, df)

    # Format p-value
    p = result.get("p_value")
    p_str = format_p_value(p) if p is not None else "---"
    stars = significance_stars(p) if p is not None else ""

    # Format effect size
    effect = result.get("effect_size")
    effect_str = None
    if effect:
        es_name = effect.get("name", "d")
        es_value = effect.get("value")
        ci_lower = effect.get("ci_lower")
        ci_upper = effect.get("ci_upper")
        if es_value is not None:
            effect_str = format_effect_size(es_name, es_value, ci_lower, ci_upper)

    # Build full report
    full_parts = [statistic_str, f"$p = {p_str}$"]
    if effect_str:
        full_parts.append(effect_str)
    full_report = ", ".join(full_parts)

    # Determine test type label
    test_type = _get_test_type_label(test_name, variant)

    return FormattedStat(
        test_type=test_type,
        statistic=statistic_str,
        p_value=p_str,
        effect_size=effect_str,
        stars=stars,
        full_report=full_report,
    )


def _get_test_type_label(name: str, variant: str = "") -> str:
    """Get human-readable test type label.

    Args:
        name: Test method name
        variant: Test variant

    Returns:
        Human-readable label
    """
    labels = {
        "t_test": "t-test",
        "t-test": "t-test",
        "ttest": "t-test",
        "anova": "ANOVA",
        "one_way_anova": "one-way ANOVA",
        "two_way_anova": "two-way ANOVA",
        "repeated_measures_anova": "repeated-measures ANOVA",
        "chi_squared": "chi-squared test",
        "chi2": "chi-squared test",
        "fisher_exact": "Fisher's exact test",
        "mann_whitney": "Mann-Whitney U test",
        "wilcoxon": "Wilcoxon signed-rank test",
        "kruskal_wallis": "Kruskal-Wallis test",
        "friedman": "Friedman test",
        "correlation": "correlation",
        "pearson": "Pearson correlation",
        "spearman": "Spearman correlation",
        "regression": "regression",
        "linear_regression": "linear regression",
    }

    base = labels.get(name.lower(), name)

    if variant:
        variant_labels = {
            "independent": "independent-samples",
            "paired": "paired-samples",
            "one_sample": "one-sample",
            "two_sample": "two-sample",
            "welch": "Welch's",
        }
        variant_str = variant_labels.get(variant.lower(), variant)
        return f"{variant_str} {base}"

    return base


def format_inline_stat(
    statistic_name: str,
    statistic_value: float,
    p_value: float,
    df: Optional[float] = None,
    effect_name: Optional[str] = None,
    effect_value: Optional[float] = None,
    include_stars: bool = True,
) -> str:
    """Format a statistical result for inline reporting.

    Args:
        statistic_name: Name of test statistic (t, F, chi2, etc.)
        statistic_value: Value of test statistic
        p_value: P-value
        df: Degrees of freedom
        effect_name: Effect size name
        effect_value: Effect size value
        include_stars: Include significance stars

    Returns:
        LaTeX formatted inline report
    """
    parts = []

    # Statistic
    stat_str = format_statistic(statistic_name, statistic_value, df)
    parts.append(stat_str)

    # P-value
    p_str = format_p_value(p_value)
    parts.append(f"$p = {p_str}$")

    # Effect size
    if effect_name and effect_value is not None:
        es_str = format_effect_size(effect_name, effect_value)
        parts.append(es_str)

    result = ", ".join(parts)

    # Add stars
    if include_stars:
        stars = significance_stars(p_value)
        if stars:
            result += f" {stars}"

    return result


def format_stat_note(stats: List[FormattedStat]) -> str:
    """Generate a table note explaining significance levels.

    Args:
        stats: List of formatted statistics

    Returns:
        LaTeX note string
    """
    has_stars = any(s.stars for s in stats)

    if not has_stars:
        return ""

    # Check which levels are used
    max_stars = max((len(s.stars) for s in stats if s.stars), default=0)

    note_parts = []
    if max_stars >= 1:
        note_parts.append("$^{*}p < .05$")
    if max_stars >= 2:
        note_parts.append("$^{**}p < .01$")
    if max_stars >= 3:
        note_parts.append("$^{***}p < .001$")

    if note_parts:
        return "\\textit{Note.} " + ", ".join(note_parts) + "."

    return ""


def format_stats_paragraph(stats: List[FormattedStat]) -> str:
    """Generate a results paragraph from formatted stats.

    Args:
        stats: List of formatted statistics

    Returns:
        LaTeX paragraph summarizing results
    """
    if not stats:
        return ""

    lines = []
    for stat in stats:
        line = f"The {stat.test_type} revealed {stat.full_report}."
        lines.append(line)

    return " ".join(lines)


__all__ = [
    "FormattedStat",
    "format_stats_for_latex",
    "format_inline_stat",
    "format_stat_note",
    "format_stats_paragraph",
]

# EOF
