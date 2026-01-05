#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex.stats.auto._formatting module.

Tests summary statistics, formatting functions, and multiple comparison correction.
"""

import numpy as np
import pytest

from scitex.stats.auto._formatting import (
    SummaryStatsDict,
    TestResultDict,
    EffectResultDict,
    CorrectionMethod,
    compute_summary_stats,
    compute_summary_from_groups,
    get_stat_symbol,
    format_test_line,
    format_test_line_compact,
    format_for_inspector,
    p_to_stars,
    apply_multiple_correction,
)


class TestGetStatSymbol:
    """Tests for get_stat_symbol function."""

    def test_t_test_returns_t(self):
        """Test t-test returns t symbol."""
        assert get_stat_symbol("ttest_ind") == "t"
        assert get_stat_symbol("ttest_rel") == "t"

    def test_anova_returns_F(self):
        """Test ANOVA tests return F symbol."""
        assert get_stat_symbol("anova_oneway") == "F"
        assert get_stat_symbol("anova_rm_oneway") == "F"
        assert get_stat_symbol("welch_anova") == "F"

    def test_brunner_munzel_returns_BM(self):
        """Test Brunner-Munzel returns BM symbol."""
        assert get_stat_symbol("brunner_munzel") == "BM"

    def test_mannwhitneyu_returns_U(self):
        """Test Mann-Whitney returns U symbol."""
        assert get_stat_symbol("mannwhitneyu") == "U"

    def test_chi2_returns_chi2(self):
        """Test chi-square tests return chi2 symbol."""
        assert get_stat_symbol("chi2_independence") == "chi2"

    def test_correlation_returns_r(self):
        """Test correlation tests return r symbol."""
        assert get_stat_symbol("pearsonr") == "r"
        assert get_stat_symbol("spearmanr") == "r"

    def test_unknown_returns_stat(self):
        """Test unknown test returns generic 'stat'."""
        assert get_stat_symbol("unknown_test") == "stat"


class TestComputeSummaryStats:
    """Tests for compute_summary_stats function."""

    def test_basic_two_groups(self):
        """Test summary stats for two groups."""
        y = np.array([1, 2, 3, 4, 5, 6])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        stats = compute_summary_stats(y, group)

        assert len(stats) == 2
        assert stats[0]["group"] == "A"
        assert stats[0]["n"] == 3
        assert stats[1]["group"] == "B"
        assert stats[1]["n"] == 3

    def test_mean_calculation(self):
        """Test mean is calculated correctly."""
        y = np.array([1, 2, 3, 10, 11, 12])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        stats = compute_summary_stats(y, group)

        assert stats[0]["mean"] == pytest.approx(2.0)
        assert stats[1]["mean"] == pytest.approx(11.0)

    def test_sd_calculation(self):
        """Test standard deviation calculation."""
        y = np.array([1, 2, 3, 10, 11, 12])
        group = np.array(["A", "A", "A", "B", "B", "B"])
        stats = compute_summary_stats(y, group)

        # SD with ddof=1
        assert stats[0]["sd"] == pytest.approx(1.0)
        assert stats[1]["sd"] == pytest.approx(1.0)

    def test_median_and_quartiles(self):
        """Test median and quartiles are computed."""
        y = np.array([1, 2, 3, 4, 5])
        group = np.array(["A", "A", "A", "A", "A"])
        stats = compute_summary_stats(y, group)

        assert stats[0]["median"] == pytest.approx(3.0)
        assert stats[0]["q1"] is not None
        assert stats[0]["q3"] is not None
        assert stats[0]["iqr"] is not None

    def test_min_max(self):
        """Test minimum and maximum."""
        y = np.array([5, 10, 15, 20, 25])
        group = np.array(["A", "A", "A", "A", "A"])
        stats = compute_summary_stats(y, group)

        assert stats[0]["minimum"] == 5.0
        assert stats[0]["maximum"] == 25.0

    def test_handles_nan(self):
        """Test NaN values are excluded from computation."""
        y = np.array([1, 2, np.nan, 4, 5])
        group = np.array(["A", "A", "A", "A", "A"])
        stats = compute_summary_stats(y, group)

        assert stats[0]["n"] == 4  # NaN excluded

    def test_unequal_group_sizes(self):
        """Test with unequal group sizes."""
        y = np.array([1, 2, 3, 4, 5, 6, 7])
        group = np.array(["A", "A", "B", "B", "B", "B", "B"])
        stats = compute_summary_stats(y, group)

        assert stats[0]["n"] == 2
        assert stats[1]["n"] == 5


class TestComputeSummaryFromGroups:
    """Tests for compute_summary_from_groups function."""

    def test_basic_usage(self):
        """Test basic usage with list of arrays."""
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        stats = compute_summary_from_groups(groups)

        assert len(stats) == 2
        assert stats[0]["n"] == 3
        assert stats[1]["n"] == 3

    def test_custom_names(self):
        """Test custom group names."""
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        stats = compute_summary_from_groups(groups, group_names=["Control", "Treatment"])

        assert stats[0]["group"] == "Control"
        assert stats[1]["group"] == "Treatment"

    def test_default_names(self):
        """Test default group names."""
        groups = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        stats = compute_summary_from_groups(groups)

        assert stats[0]["group"] == "Group_1"
        assert stats[1]["group"] == "Group_2"


class TestFormatTestLine:
    """Tests for format_test_line function."""

    def test_basic_formatting(self):
        """Test basic test line formatting."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "df": 28.0, "p_raw": 0.028}
        line = format_test_line(test, style="plain")

        assert "2.31" in line
        assert "0.028" in line

    def test_with_effects(self):
        """Test formatting with effect sizes."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "p_raw": 0.028}
        effects = [{"name": "cohens_d_ind", "value": 0.72, "label": "Cohen's d"}]
        line = format_test_line(test, effects=effects, style="plain")

        assert "0.72" in line

    def test_with_summary(self):
        """Test formatting with sample sizes."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "p_raw": 0.028}
        summary = [{"group": "A", "n": 15}, {"group": "B", "n": 15}]
        line = format_test_line(test, summary=summary, style="plain", include_n=True)

        assert "15" in line

    def test_latex_style(self):
        """Test LaTeX formatting."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "df": 28.0, "p_raw": 0.028}
        line = format_test_line(test, style="apa_latex")

        assert "\\mathit{t}" in line
        assert "\\mathit{p}" in line

    def test_html_style(self):
        """Test HTML formatting."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "df": 28.0, "p_raw": 0.028}
        line = format_test_line(test, style="apa_html")

        assert "<i>t</i>" in line
        assert "<i>p</i>" in line

    def test_uses_p_adj_when_available(self):
        """Test that p_adj is used when available."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "p_raw": 0.01, "p_adj": 0.03}
        line = format_test_line(test, style="plain")

        assert "0.03" in line  # Should use p_adj

    def test_max_effects_limit(self):
        """Test maximum effects limit."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "p_raw": 0.028}
        effects = [
            {"name": "effect1", "value": 0.5},
            {"name": "effect2", "value": 0.6},
            {"name": "effect3", "value": 0.7},
        ]
        line = format_test_line(test, effects=effects, style="plain", max_effects=1)

        # Should only include first effect
        assert "0.5" in line
        assert "0.7" not in line


class TestFormatTestLineCompact:
    """Tests for format_test_line_compact function."""

    def test_compact_excludes_n(self):
        """Test that compact format excludes sample sizes."""
        test = {"test_name": "ttest_ind", "stat": 2.31, "p_raw": 0.028}
        line = format_test_line_compact(test, style="plain")

        # Should not have sample size formatting
        assert "n" not in line or "n =" not in line


class TestFormatForInspector:
    """Tests for format_for_inspector function."""

    def test_basic_structure(self):
        """Test returned structure has tests and effects keys."""
        tests = [{"test_name": "ttest_ind", "p_raw": 0.03, "stat": 2.2}]
        effects = [{"name": "cohens_d_ind", "value": 0.8, "label": "Cohen's d"}]
        result = format_for_inspector(tests, effects)

        assert "tests" in result
        assert "effects" in result
        assert len(result["tests"]) == 1
        assert len(result["effects"]) == 1

    def test_empty_effects(self):
        """Test with no effect sizes."""
        tests = [{"test_name": "ttest_ind", "p_raw": 0.03}]
        result = format_for_inspector(tests, None)

        assert len(result["effects"]) == 0

    def test_test_fields_included(self):
        """Test that all test fields are included."""
        tests = [
            {
                "test_name": "ttest_ind",
                "p_raw": 0.03,
                "p_adj": 0.06,
                "stat": 2.2,
                "df": 28,
                "method": "Two-sample t-test",
            }
        ]
        result = format_for_inspector(tests)

        test_item = result["tests"][0]
        assert test_item["name"] == "ttest_ind"
        assert test_item["p_raw"] == 0.03
        assert test_item["p_adj"] == 0.06


class TestPToStars:
    """Tests for p_to_stars function."""

    def test_highly_significant(self):
        """Test p < 0.001 returns ***."""
        assert p_to_stars(0.0001) == "***"

    def test_very_significant(self):
        """Test p < 0.01 returns **."""
        assert p_to_stars(0.005) == "**"

    def test_significant(self):
        """Test p < 0.05 returns *."""
        assert p_to_stars(0.03) == "*"

    def test_not_significant(self):
        """Test p >= 0.05 returns ns."""
        assert p_to_stars(0.08) == "ns"

    def test_none_returns_ns(self):
        """Test None p-value returns ns."""
        assert p_to_stars(None) == "ns"

    def test_with_custom_style(self):
        """Test using custom style."""
        result = p_to_stars(0.03, style="apa_latex")
        assert result == "*"


class TestApplyMultipleCorrection:
    """Tests for apply_multiple_correction function."""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction multiplies by m."""
        results = [
            {"test_name": "t1", "p_raw": 0.01},
            {"test_name": "t2", "p_raw": 0.02},
            {"test_name": "t3", "p_raw": 0.03},
        ]
        corrected = apply_multiple_correction(results, method="bonferroni")

        assert corrected[0]["p_adj"] == pytest.approx(0.03)  # 0.01 * 3
        assert corrected[1]["p_adj"] == pytest.approx(0.06)  # 0.02 * 3
        assert corrected[2]["p_adj"] == pytest.approx(0.09)  # 0.03 * 3

    def test_bonferroni_caps_at_1(self):
        """Test Bonferroni caps at 1.0."""
        results = [
            {"test_name": "t1", "p_raw": 0.5},
            {"test_name": "t2", "p_raw": 0.6},
        ]
        corrected = apply_multiple_correction(results, method="bonferroni")

        assert corrected[0]["p_adj"] == 1.0  # Would be 1.0, capped
        assert corrected[1]["p_adj"] == 1.0  # Would be 1.2, capped

    def test_no_correction(self):
        """Test no correction method."""
        results = [
            {"test_name": "t1", "p_raw": 0.01},
            {"test_name": "t2", "p_raw": 0.03},
        ]
        corrected = apply_multiple_correction(results, method="none")

        assert corrected[0]["p_adj"] == 0.01  # Unchanged
        assert corrected[1]["p_adj"] == 0.03  # Unchanged

    def test_none_method(self):
        """Test None method."""
        results = [
            {"test_name": "t1", "p_raw": 0.01},
        ]
        corrected = apply_multiple_correction(results, method=None)

        assert corrected[0]["p_adj"] == 0.01

    def test_fdr_bh_correction(self):
        """Test FDR Benjamini-Hochberg correction."""
        results = [
            {"test_name": "t1", "p_raw": 0.01},
            {"test_name": "t2", "p_raw": 0.02},
            {"test_name": "t3", "p_raw": 0.03},
        ]
        corrected = apply_multiple_correction(results, method="fdr_bh")

        # FDR should preserve order and be >= p_raw
        assert corrected[0]["p_adj"] >= corrected[0]["p_raw"]
        assert all(r["p_adj"] <= 1.0 for r in corrected)

    def test_holm_correction(self):
        """Test Holm step-down correction."""
        results = [
            {"test_name": "t1", "p_raw": 0.01},
            {"test_name": "t2", "p_raw": 0.03},
            {"test_name": "t3", "p_raw": 0.05},
        ]
        corrected = apply_multiple_correction(results, method="holm")

        # Holm should be >= p_raw and monotonic
        for r in corrected:
            assert r["p_adj"] >= r["p_raw"]
            assert r["p_adj"] <= 1.0

    def test_sets_correction_method(self):
        """Test that correction_method is set."""
        results = [{"test_name": "t1", "p_raw": 0.01}]
        corrected = apply_multiple_correction(results, method="bonferroni")

        assert corrected[0]["correction_method"] == "bonferroni"

    def test_handles_none_p_values(self):
        """Test results with None p-values are skipped."""
        results = [
            {"test_name": "t1", "p_raw": 0.01},
            {"test_name": "t2", "p_raw": None},
            {"test_name": "t3", "p_raw": 0.03},
        ]
        corrected = apply_multiple_correction(results, method="bonferroni")

        # Only 2 valid p-values, so multiply by 2
        assert corrected[0]["p_adj"] == pytest.approx(0.02)
        assert corrected[2]["p_adj"] == pytest.approx(0.06)

    def test_empty_results(self):
        """Test empty results list."""
        results = []
        corrected = apply_multiple_correction(results, method="bonferroni")
        assert corrected == []

    def test_all_none_p_values(self):
        """Test when all p-values are None."""
        results = [
            {"test_name": "t1", "p_raw": None},
            {"test_name": "t2", "p_raw": None},
        ]
        corrected = apply_multiple_correction(results, method="bonferroni")

        # Should return unchanged
        assert len(corrected) == 2


class TestTypeDefinitions:
    """Tests for TypedDict definitions."""

    def test_summary_stats_dict_creation(self):
        """Test SummaryStatsDict can be created."""
        stats: SummaryStatsDict = {
            "group": "A",
            "n": 10,
            "mean": 5.0,
            "sd": 1.0,
        }
        assert stats["group"] == "A"
        assert stats["n"] == 10

    def test_test_result_dict_creation(self):
        """Test TestResultDict can be created."""
        result: TestResultDict = {
            "test_name": "ttest_ind",
            "p_raw": 0.03,
            "stat": 2.2,
        }
        assert result["test_name"] == "ttest_ind"

    def test_effect_result_dict_creation(self):
        """Test EffectResultDict can be created."""
        effect: EffectResultDict = {
            "name": "cohens_d_ind",
            "value": 0.8,
            "label": "Cohen's d",
        }
        assert effect["value"] == 0.8

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_formatting.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_formatting.py
# 
# """
# Statistical Formatting - Publication-ready output generation.
# 
# This module provides functions for:
# - Computing summary statistics per group
# - Formatting complete test result lines
# - Converting results for Inspector panel display
# - Handling multiple comparison corrections
# 
# All formatting respects journal style presets (APA, Nature, Cell, Elsevier).
# """
# 
# from __future__ import annotations
# 
# from dataclasses import dataclass
# from typing import Dict, List, Optional, Any, TypedDict, Union
# 
# import numpy as np
# 
# from ._styles import StatStyle, get_stat_style
# 
# 
# # =============================================================================
# # Type Definitions
# # =============================================================================
# 
# 
# class SummaryStatsDict(TypedDict, total=False):
#     """
#     Summary statistics for a single group.
# 
#     Attributes
#     ----------
#     group : str
#         Group name/label.
#     n : int
#         Sample size.
#     mean : float or None
#         Mean value.
#     sd : float or None
#         Standard deviation.
#     sem : float or None
#         Standard error of mean.
#     median : float or None
#         Median value.
#     iqr : float or None
#         Interquartile range.
#     q1 : float or None
#         First quartile (25th percentile).
#     q3 : float or None
#         Third quartile (75th percentile).
#     minimum : float or None
#         Minimum value.
#     maximum : float or None
#         Maximum value.
#     """
#     group: str
#     n: int
#     mean: Optional[float]
#     sd: Optional[float]
#     sem: Optional[float]
#     median: Optional[float]
#     iqr: Optional[float]
#     q1: Optional[float]
#     q3: Optional[float]
#     minimum: Optional[float]
#     maximum: Optional[float]
# 
# 
# class TestResultDict(TypedDict, total=False):
#     """
#     Result structure for a single statistical test.
# 
#     Attributes
#     ----------
#     test_name : str
#         Internal test name ("ttest_ind", "brunner_munzel", etc.).
#     p_raw : float or None
#         Raw p-value.
#     p_adj : float or None
#         Adjusted p-value after multiple correction.
#     stat : float or None
#         Test statistic value.
#     df : float or None
#         Degrees of freedom.
#     method : str or None
#         Human-readable method label.
#     correction_method : str or None
#         Multiple correction method used ("bonferroni", "fdr_bh", etc.).
#     details : dict
#         Additional test-specific information.
#     """
#     test_name: str
#     p_raw: Optional[float]
#     p_adj: Optional[float]
#     stat: Optional[float]
#     df: Optional[float]
#     method: Optional[str]
#     correction_method: Optional[str]
#     details: Dict[str, Any]
# 
# 
# class EffectResultDict(TypedDict, total=False):
#     """
#     Result structure for a single effect size measure.
# 
#     Attributes
#     ----------
#     name : str
#         Internal name ("cohens_d_ind", "eta_squared", etc.).
#     label : str
#         Human-readable label.
#     value : float
#         Effect size value.
#     ci_lower : float or None
#         Lower bound of confidence interval.
#     ci_upper : float or None
#         Upper bound of confidence interval.
#     note : str or None
#         Interpretation note (e.g., "small", "medium", "large").
#     """
#     name: str
#     label: str
#     value: float
#     ci_lower: Optional[float]
#     ci_upper: Optional[float]
#     note: Optional[str]
# 
# 
# # =============================================================================
# # Statistic Symbol Mapping
# # =============================================================================
# 
# _STAT_SYMBOLS: Dict[str, str] = {
#     # Parametric
#     "ttest_ind": "t",
#     "ttest_rel": "t",
#     "anova_oneway": "F",
#     "anova_rm_oneway": "F",
#     "anova_twoway": "F",
#     "anova_twoway_mixed": "F",
#     "welch_anova": "F",
# 
#     # Nonparametric
#     "brunner_munzel": "BM",
#     "mannwhitneyu": "U",
#     "wilcoxon": "W",
#     "kruskal": "H",
#     "friedman": "chi2",
# 
#     # Categorical
#     "chi2_independence": "chi2",
#     "fisher_exact": "OR",
#     "mcnemar": "chi2",
# 
#     # Correlation
#     "pearsonr": "r",
#     "spearmanr": "r",
# 
#     # Normality
#     "shapiro": "W",
#     "levene": "F",
# }
# 
# 
# def get_stat_symbol(test_name: str) -> str:
#     """
#     Get the statistic symbol for a test.
# 
#     Parameters
#     ----------
#     test_name : str
#         Internal test name.
# 
#     Returns
#     -------
#     str
#         Statistic symbol (e.g., "t", "F", "BM").
#     """
#     return _STAT_SYMBOLS.get(test_name, "stat")
# 
# 
# # =============================================================================
# # Summary Statistics
# # =============================================================================
# 
# 
# def compute_summary_stats(
#     y: np.ndarray,
#     group: np.ndarray,
# ) -> List[SummaryStatsDict]:
#     """
#     Compute per-group summary statistics.
# 
#     Parameters
#     ----------
#     y : np.ndarray
#         Outcome values.
#     group : np.ndarray
#         Group labels for each observation.
# 
#     Returns
#     -------
#     list of SummaryStatsDict
#         Summary statistics for each group.
# 
#     Examples
#     --------
#     >>> y = np.array([1, 2, 3, 4, 5, 6])
#     >>> group = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
#     >>> stats = compute_summary_stats(y, group)
#     >>> stats[0]['group']
#     'A'
#     >>> stats[0]['n']
#     3
#     """
#     y = np.asarray(y, dtype=float)
#     group = np.asarray(group)
# 
#     stats_list: List[SummaryStatsDict] = []
# 
#     for group_value in np.unique(group):
#         mask = group == group_value
#         vals = y[mask]
#         vals = vals[~np.isnan(vals)]  # Remove NaN
# 
#         if vals.size == 0:
#             continue
# 
#         n = int(vals.size)
#         mean = float(vals.mean())
#         sd = float(vals.std(ddof=1)) if n > 1 else 0.0
#         sem = float(sd / np.sqrt(n)) if n > 1 else 0.0
# 
#         q1, med, q3 = np.percentile(vals, [25, 50, 75])
#         iqr = q3 - q1
# 
#         stats_list.append(
#             SummaryStatsDict(
#                 group=str(group_value),
#                 n=n,
#                 mean=mean,
#                 sd=sd,
#                 sem=sem,
#                 median=float(med),
#                 iqr=float(iqr),
#                 q1=float(q1),
#                 q3=float(q3),
#                 minimum=float(vals.min()),
#                 maximum=float(vals.max()),
#             )
#         )
# 
#     return stats_list
# 
# 
# def compute_summary_from_groups(
#     groups: List[np.ndarray],
#     group_names: Optional[List[str]] = None,
# ) -> List[SummaryStatsDict]:
#     """
#     Compute summary statistics from a list of group arrays.
# 
#     Parameters
#     ----------
#     groups : list of np.ndarray
#         List of arrays, one per group.
#     group_names : list of str, optional
#         Names for each group.
# 
#     Returns
#     -------
#     list of SummaryStatsDict
#         Summary statistics for each group.
#     """
#     if group_names is None:
#         group_names = [f"Group_{i+1}" for i in range(len(groups))]
# 
#     stats_list: List[SummaryStatsDict] = []
# 
#     for name, vals in zip(group_names, groups):
#         vals = np.asarray(vals, dtype=float)
#         vals = vals[~np.isnan(vals)]
# 
#         if vals.size == 0:
#             continue
# 
#         n = int(vals.size)
#         mean = float(vals.mean())
#         sd = float(vals.std(ddof=1)) if n > 1 else 0.0
#         sem = float(sd / np.sqrt(n)) if n > 1 else 0.0
# 
#         q1, med, q3 = np.percentile(vals, [25, 50, 75])
#         iqr = q3 - q1
# 
#         stats_list.append(
#             SummaryStatsDict(
#                 group=name,
#                 n=n,
#                 mean=mean,
#                 sd=sd,
#                 sem=sem,
#                 median=float(med),
#                 iqr=float(iqr),
#                 q1=float(q1),
#                 q3=float(q3),
#                 minimum=float(vals.min()),
#                 maximum=float(vals.max()),
#             )
#         )
# 
#     return stats_list
# 
# 
# # =============================================================================
# # Test Line Formatting
# # =============================================================================
# 
# 
# def format_test_line(
#     test: TestResultDict,
#     effects: Optional[List[EffectResultDict]] = None,
#     summary: Optional[List[SummaryStatsDict]] = None,
#     style: Optional[Union[str, StatStyle]] = None,
#     include_n: bool = True,
#     max_effects: int = 2,
# ) -> str:
#     """
#     Format a complete statistical result line.
# 
#     Produces publication-ready formatted text with proper italics,
#     symbols, and formatting according to the specified journal style.
# 
#     Parameters
#     ----------
#     test : TestResultDict
#         Test result dictionary.
#     effects : list of EffectResultDict, optional
#         Effect size results to include.
#     summary : list of SummaryStatsDict, optional
#         Summary statistics for sample size display.
#     style : str or StatStyle, optional
#         Style to use. Can be style ID or StatStyle instance.
#         Defaults to APA LaTeX.
#     include_n : bool
#         Whether to include sample sizes in output.
#     max_effects : int
#         Maximum number of effect sizes to include.
# 
#     Returns
#     -------
#     str
#         Formatted result line.
# 
#     Examples
#     --------
#     >>> test = {"test_name": "ttest_ind", "stat": 2.31, "df": 28.0, "p_raw": 0.028}
#     >>> effects = [{"name": "cohens_d_ind", "value": 0.72, "label": "Cohen's d"}]
#     >>> summary = [{"group": "A", "n": 15}, {"group": "B", "n": 15}]
#     >>> line = format_test_line(test, effects, summary, style="apa_latex")
#     >>> "\\mathit{t}" in line
#     True
#     """
#     # Get style
#     if style is None:
#         style = get_stat_style("apa_latex")
#     elif isinstance(style, str):
#         style = get_stat_style(style)
# 
#     parts: List[str] = []
# 
#     # Format test statistic
#     test_name = test.get("test_name", "")
#     stat = test.get("stat")
#     df = test.get("df")
# 
#     if stat is not None:
#         symbol = get_stat_symbol(test_name)
#         stat_part = style.format_stat(symbol, stat, df)
#         parts.append(stat_part)
# 
#     # Format p-value
#     p = test.get("p_adj") or test.get("p_raw")
#     if p is not None:
#         p_part = style.format_p(p)
#         parts.append(p_part)
# 
#     # Format effect sizes
#     if effects:
#         for eff in effects[:max_effects]:
#             eff_name = eff.get("name", "")
#             eff_value = eff.get("value")
#             if eff_value is not None:
#                 eff_part = style.format_effect(eff_name, eff_value)
#                 parts.append(eff_part)
# 
#     # Format sample sizes
#     if include_n and summary:
#         for s in summary:
#             group_name = str(s.get("group", ""))
#             n_value = int(s.get("n", 0))
#             n_part = style.format_n(group_name, n_value)
#             parts.append(n_part)
# 
#     return ", ".join(parts)
# 
# 
# def format_test_line_compact(
#     test: TestResultDict,
#     style: Optional[Union[str, StatStyle]] = None,
# ) -> str:
#     """
#     Format a compact test result (statistic + p-value only).
# 
#     Parameters
#     ----------
#     test : TestResultDict
#         Test result dictionary.
#     style : str or StatStyle, optional
#         Style to use.
# 
#     Returns
#     -------
#     str
#         Compact formatted result.
#     """
#     return format_test_line(
#         test,
#         effects=None,
#         summary=None,
#         style=style,
#         include_n=False,
#     )
# 
# 
# # =============================================================================
# # Inspector Panel Formatting
# # =============================================================================
# 
# 
# def format_for_inspector(
#     test_results: List[TestResultDict],
#     effect_results: Optional[List[EffectResultDict]] = None,
# ) -> Dict[str, List[Dict]]:
#     """
#     Format results for Inspector panel display.
# 
#     Produces a structure suitable for displaying in a UI panel
#     with tables for tests and effect sizes.
# 
#     Parameters
#     ----------
#     test_results : list of TestResultDict
#         Test results to display.
#     effect_results : list of EffectResultDict, optional
#         Effect size results to display.
# 
#     Returns
#     -------
#     dict
#         Dictionary with 'tests' and 'effects' lists.
# 
#     Examples
#     --------
#     >>> tests = [{"test_name": "ttest_ind", "p_raw": 0.03, "stat": 2.2}]
#     >>> effects = [{"name": "cohens_d_ind", "value": 0.8, "label": "Cohen's d"}]
#     >>> result = format_for_inspector(tests, effects)
#     >>> len(result["tests"])
#     1
#     """
#     from ._selector import _pretty_label
# 
#     return {
#         "tests": [
#             {
#                 "name": r.get("test_name"),
#                 "label": _pretty_label(r.get("test_name", "")),
#                 "p_raw": r.get("p_raw"),
#                 "p_adj": r.get("p_adj"),
#                 "stat": r.get("stat"),
#                 "df": r.get("df"),
#                 "method": r.get("method"),
#                 "correction": r.get("correction_method"),
#                 "details": r.get("details", {}),
#             }
#             for r in test_results
#         ],
#         "effects": [
#             {
#                 "name": e.get("name"),
#                 "label": e.get("label", e.get("name", "")),
#                 "value": e.get("value"),
#                 "ci_lower": e.get("ci_lower"),
#                 "ci_upper": e.get("ci_upper"),
#                 "note": e.get("note"),
#             }
#             for e in (effect_results or [])
#         ],
#     }
# 
# 
# # =============================================================================
# # P-value to Stars
# # =============================================================================
# 
# 
# def p_to_stars(
#     p_value: Optional[float],
#     style: Optional[Union[str, StatStyle]] = None,
# ) -> str:
#     """
#     Convert p-value to significance stars.
# 
#     Uses the alpha thresholds from the specified style.
# 
#     Parameters
#     ----------
#     p_value : float or None
#         P-value to convert.
#     style : str or StatStyle, optional
#         Style to use for thresholds.
# 
#     Returns
#     -------
#     str
#         Stars string ("***", "**", "*", or "ns").
# 
#     Examples
#     --------
#     >>> p_to_stars(0.001)
#     '***'
#     >>> p_to_stars(0.03)
#     '*'
#     >>> p_to_stars(0.10)
#     'ns'
#     """
#     if style is None:
#         style = get_stat_style("apa_latex")
#     elif isinstance(style, str):
#         style = get_stat_style(style)
# 
#     return style.p_to_stars(p_value)
# 
# 
# # =============================================================================
# # Multiple Comparison Correction
# # =============================================================================
# 
# 
# CorrectionMethod = Union[str, None]
# 
# 
# def apply_multiple_correction(
#     results: List[TestResultDict],
#     method: CorrectionMethod = "fdr_bh",
# ) -> List[TestResultDict]:
#     """
#     Apply multiple-comparison correction to test results.
# 
#     Modifies results in-place by setting p_adj and correction_method.
# 
#     Parameters
#     ----------
#     results : list of TestResultDict
#         Test results with p_raw values.
#     method : str or None
#         Correction method:
#         - "none": No correction (p_adj = p_raw)
#         - "bonferroni": Bonferroni correction
#         - "holm": Holm-Bonferroni step-down
#         - "fdr_bh": Benjamini-Hochberg FDR
# 
#     Returns
#     -------
#     list of TestResultDict
#         Results with p_adj filled in.
# 
#     Examples
#     --------
#     >>> results = [
#     ...     {"test_name": "t1", "p_raw": 0.01},
#     ...     {"test_name": "t2", "p_raw": 0.03},
#     ...     {"test_name": "t3", "p_raw": 0.04},
#     ... ]
#     >>> corrected = apply_multiple_correction(results, "bonferroni")
#     >>> corrected[0]["p_adj"]
#     0.03
#     """
#     if method is None or method == "none":
#         for r in results:
#             r["p_adj"] = r.get("p_raw")
#             r["correction_method"] = "none"
#         return results
# 
#     # Get valid p-values
#     valid_indices = [
#         i for i, r in enumerate(results)
#         if r.get("p_raw") is not None
#     ]
# 
#     if not valid_indices:
#         return results
# 
#     p_values = [results[i]["p_raw"] for i in valid_indices]
#     m = len(p_values)
# 
#     adjusted: List[float] = []
# 
#     if method == "bonferroni":
#         adjusted = [min(p * m, 1.0) for p in p_values]
# 
#     elif method == "holm":
#         # Holm-Bonferroni step-down
#         sorted_idx = sorted(range(m), key=lambda i: p_values[i])
#         adj = [0.0] * m
#         cummax = 0.0
#         for rank, idx in enumerate(sorted_idx, start=1):
#             adj_val = min((m - rank + 1) * p_values[idx], 1.0)
#             adj_val = max(adj_val, cummax)  # Enforce monotonicity
#             adj[idx] = adj_val
#             cummax = adj_val
#         adjusted = adj
# 
#     elif method == "fdr_bh":
#         # Benjamini-Hochberg
#         sorted_idx = sorted(range(m), key=lambda i: p_values[i])
#         adj = [0.0] * m
#         prev = 1.0
#         for rank in range(m, 0, -1):
#             idx = sorted_idx[rank - 1]
#             p = p_values[idx]
#             bh = p * m / rank
#             val = min(bh, prev, 1.0)
#             adj[idx] = val
#             prev = val
#         adjusted = adj
# 
#     else:
#         # Unknown method - no correction
#         adjusted = p_values
# 
#     # Write back
#     for local_i, global_i in enumerate(valid_indices):
#         results[global_i]["p_adj"] = float(adjusted[local_i])
#         results[global_i]["correction_method"] = method
# 
#     return results
# 
# 
# # =============================================================================
# # Public API
# # =============================================================================
# 
# __all__ = [
#     # Type definitions
#     "SummaryStatsDict",
#     "TestResultDict",
#     "EffectResultDict",
#     "CorrectionMethod",
#     # Summary statistics
#     "compute_summary_stats",
#     "compute_summary_from_groups",
#     # Symbol mapping
#     "get_stat_symbol",
#     # Formatting
#     "format_test_line",
#     "format_test_line_compact",
#     "format_for_inspector",
#     "p_to_stars",
#     # Correction
#     "apply_multiple_correction",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_formatting.py
# --------------------------------------------------------------------------------
