#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for test selector functions.

Tests cover:
- check_applicable() returns (bool, List[str]) with correct reasons
- get_menu_items() returns properly structured menu items
- recommend_tests() returns top-k tests sorted by priority
- Brunner-Munzel is recommended first for 2-group between-subjects
- Tests are correctly disabled with appropriate tooltips when assumptions fail
- Family filtering for recommendations and menu generation
"""

import pytest
from scitex.stats.auto._context import StatContext
from scitex.stats.auto._rules import TEST_RULES
from scitex.stats.auto._selector import (
    check_applicable,
    get_menu_items,
    recommend_tests,
    recommend_effect_sizes,
    recommend_posthoc,
)


class TestCheckApplicable:
    """Tests for check_applicable() function."""

    def test_returns_tuple_with_bool_and_list(self):
        """Test that check_applicable returns (bool, list) tuple."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert isinstance(ok, bool)
        assert isinstance(reasons, list)
        assert all(isinstance(r, str) for r in reasons)

    def test_ttest_ind_applicable_with_normality(self):
        """Test that t-test is applicable when normality is met."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is True
        assert len(reasons) == 0

    def test_ttest_ind_not_applicable_without_normality(self):
        """Test that t-test is not applicable when normality fails."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("normality" in r.lower() for r in reasons)

    def test_brunner_munzel_applicable_without_assumptions(self):
        """Test that Brunner-Munzel is applicable without assumptions."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
            variance_homogeneity_ok=False,
        )
        rule = TEST_RULES["brunner_munzel"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is True
        assert len(reasons) == 0

    def test_not_applicable_wrong_group_count_too_few(self):
        """Test rejection when too few groups."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["anova_oneway"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("at least" in r.lower() and "groups" in r.lower() for r in reasons)

    def test_not_applicable_wrong_group_count_too_many(self):
        """Test rejection when too many groups."""
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("maximum" in r.lower() and "groups" in r.lower() for r in reasons)

    def test_not_applicable_wrong_outcome_type(self):
        """Test rejection for wrong outcome type."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("continuous" in r.lower() for r in reasons)

    def test_not_applicable_paired_mismatch(self):
        """Test rejection for paired/unpaired mismatch."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("paired" in r.lower() or "independent" in r.lower() for r in reasons)

    def test_not_applicable_design_mismatch(self):
        """Test rejection for design mismatch."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="within",
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("design" in r.lower() for r in reasons)

    def test_not_applicable_sample_size_too_small_total(self):
        """Test rejection for insufficient total sample size."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 1],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("sample size" in r.lower() for r in reasons)

    def test_not_applicable_sample_size_too_small_per_group(self):
        """Test rejection for insufficient per-group sample size."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[1, 100],
            outcome_type="continuous",
            design="between",
        )
        rule = TEST_RULES["brunner_munzel"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("each group" in r.lower() or "smallest group" in r.lower() for r in reasons)

    def test_not_applicable_equal_variance_assumption_fails(self):
        """Test rejection when equal variance assumption fails."""
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=False,
        )
        rule = TEST_RULES["anova_oneway"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("equal variance" in r.lower() or "variance" in r.lower() for r in reasons)

    def test_not_applicable_requires_control_group(self):
        """Test rejection when control group is required but missing."""
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            has_control_group=False,
            normality_ok=True,
            variance_homogeneity_ok=True,
        )
        rule = TEST_RULES["dunnett"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("control group" in r.lower() for r in reasons)

    def test_not_applicable_too_few_factors(self):
        """Test rejection for insufficient factors."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            n_factors=1,
        )
        rule = TEST_RULES["anova_twoway"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("factor" in r.lower() for r in reasons)

    def test_not_applicable_too_many_factors(self):
        """Test rejection for too many factors."""
        ctx = StatContext(
            n_groups=8,
            sample_sizes=[10] * 8,
            outcome_type="continuous",
            design="between",
            n_factors=3,
        )
        rule = TEST_RULES["anova_oneway"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) > 0
        assert any("factor" in r.lower() for r in reasons)

    def test_multiple_reasons_for_rejection(self):
        """Test that multiple rejection reasons are collected."""
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[5],
            outcome_type="categorical",
            design="within",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) >= 2


class TestGetMenuItems:
    """Tests for get_menu_items() function."""

    def test_returns_list_of_dicts(self):
        """Test that get_menu_items returns a list of dictionaries."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx)

        assert isinstance(items, list)
        assert all(isinstance(item, dict) for item in items)

    def test_menu_items_have_required_fields(self):
        """Test that menu items have all required fields."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx)

        required_fields = ["id", "label", "family", "enabled", "tooltip", "priority"]
        for item in items:
            for field in required_fields:
                assert field in item, f"Item should have '{field}' field"

    def test_menu_items_enabled_for_brunner_munzel(self):
        """Test that Brunner-Munzel is enabled for 2-group comparison."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx)
        bm_items = [i for i in items if i["id"] == "brunner_munzel"]

        assert len(bm_items) == 1
        assert bm_items[0]["enabled"] is True
        assert bm_items[0]["tooltip"] is None

    def test_menu_items_disabled_with_tooltip(self):
        """Test that disabled items have tooltip explaining why."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )

        items = get_menu_items(ctx)
        ttest_items = [i for i in items if i["id"] == "ttest_ind"]

        assert len(ttest_items) == 1
        assert ttest_items[0]["enabled"] is False
        assert ttest_items[0]["tooltip"] is not None
        assert "normality" in ttest_items[0]["tooltip"].lower()

    def test_menu_items_sorted_enabled_first(self):
        """Test that enabled items appear before disabled items."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )

        items = get_menu_items(ctx)

        enabled_indices = [i for i, item in enumerate(items) if item["enabled"]]
        disabled_indices = [i for i, item in enumerate(items) if not item["enabled"]]

        if enabled_indices and disabled_indices:
            assert max(enabled_indices) < min(disabled_indices)

    def test_menu_items_sorted_by_priority(self):
        """Test that enabled items are sorted by priority (descending)."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx)
        enabled_items = [i for i in items if i["enabled"]]

        priorities = [item["priority"] for item in enabled_items]
        assert priorities == sorted(priorities, reverse=True)

    def test_menu_items_include_families_filter(self):
        """Test filtering by include_families."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx, include_families=["nonparametric"])

        assert all(item["family"] == "nonparametric" for item in items)
        assert any(item["id"] == "brunner_munzel" for item in items)
        assert not any(item["id"] == "ttest_ind" for item in items)

    def test_menu_items_exclude_families_filter(self):
        """Test filtering by exclude_families."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx, exclude_families=["parametric"])

        assert all(item["family"] != "parametric" for item in items)
        assert any(item["id"] == "brunner_munzel" for item in items)
        assert not any(item["id"] == "ttest_ind" for item in items)

    def test_menu_items_has_brunner_munzel_first(self):
        """Test that Brunner-Munzel appears first for 2-group comparison."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx)
        enabled_items = [i for i in items if i["enabled"]]

        if enabled_items:
            assert enabled_items[0]["id"] == "brunner_munzel"


class TestRecommendTests:
    """Tests for recommend_tests() function."""

    def test_returns_list_of_strings(self):
        """Test that recommend_tests returns list of test names."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        recommended = recommend_tests(ctx, top_k=3)

        assert isinstance(recommended, list)
        assert all(isinstance(name, str) for name in recommended)

    def test_brunner_munzel_recommended_first_for_two_groups(self):
        """Test that Brunner-Munzel is recommended first for 2-group comparison."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        recommended = recommend_tests(ctx, top_k=3)

        assert len(recommended) > 0
        assert recommended[0] == "brunner_munzel"

    def test_recommend_tests_respects_top_k(self):
        """Test that recommend_tests returns at most top_k tests."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        recommended = recommend_tests(ctx, top_k=2)

        assert len(recommended) <= 2

    def test_recommend_tests_sorted_by_priority(self):
        """Test that recommended tests are sorted by priority."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        recommended = recommend_tests(ctx, top_k=5)

        priorities = [TEST_RULES[name].priority for name in recommended]
        assert priorities == sorted(priorities, reverse=True)

    def test_recommend_tests_only_applicable(self):
        """Test that only applicable tests are recommended."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )

        recommended = recommend_tests(ctx, top_k=10)

        assert "brunner_munzel" in recommended
        assert "ttest_ind" not in recommended

    def test_recommend_tests_with_families_filter(self):
        """Test filtering by families parameter."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        recommended = recommend_tests(ctx, top_k=5, families=["nonparametric"])

        assert all(TEST_RULES[name].family == "nonparametric" for name in recommended)
        assert "brunner_munzel" in recommended
        assert "ttest_ind" not in recommended

    def test_recommend_tests_for_paired_design(self):
        """Test recommendations for paired design."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )

        recommended = recommend_tests(ctx, top_k=3)

        assert all(TEST_RULES[name].supports_paired for name in recommended)
        assert any(name in ["ttest_rel", "wilcoxon"] for name in recommended)

    def test_recommend_tests_for_three_groups(self):
        """Test recommendations for three groups."""
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
        )

        recommended = recommend_tests(ctx, top_k=3)

        assert all(TEST_RULES[name].min_groups <= 3 for name in recommended)
        assert "brunner_munzel" not in recommended

    def test_recommend_tests_empty_when_none_applicable(self):
        """Test that empty list is returned when no tests are applicable."""
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[10],
            outcome_type="continuous",
            design="between",
        )

        recommended = recommend_tests(ctx, top_k=3, families=["parametric", "nonparametric"])

        two_group_tests = [
            name for name in recommended
            if TEST_RULES[name].min_groups > 1
        ]
        assert len(two_group_tests) == 0


class TestRecommendEffectSizes:
    """Tests for recommend_effect_sizes() function."""

    def test_returns_list_of_effect_sizes(self):
        """Test that recommend_effect_sizes returns effect size names."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        effect_sizes = recommend_effect_sizes(ctx, top_k=3)

        assert isinstance(effect_sizes, list)
        assert all(isinstance(name, str) for name in effect_sizes)
        assert all(TEST_RULES[name].family == "effect_size" for name in effect_sizes)

    def test_recommend_cohens_d_for_two_groups(self):
        """Test that Cohen's d is recommended for 2-group comparison."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        effect_sizes = recommend_effect_sizes(ctx, top_k=3)

        assert "cohens_d_ind" in effect_sizes

    def test_recommend_cohens_d_paired_for_within_design(self):
        """Test that Cohen's d paired is recommended for within design."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 30],
            outcome_type="continuous",
            design="within",
            paired=True,
        )

        effect_sizes = recommend_effect_sizes(ctx, top_k=3)

        assert "cohens_d_paired" in effect_sizes


class TestRecommendPosthoc:
    """Tests for recommend_posthoc() function."""

    def test_returns_list_of_posthoc_tests(self):
        """Test that recommend_posthoc returns post-hoc test names."""
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=True,
        )

        posthoc = recommend_posthoc(ctx, top_k=2)

        assert isinstance(posthoc, list)
        assert all(isinstance(name, str) for name in posthoc)
        assert all(TEST_RULES[name].family == "posthoc" for name in posthoc)

    def test_recommend_tukey_for_three_groups(self):
        """Test that Tukey HSD is recommended for 3+ groups."""
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=True,
        )

        posthoc = recommend_posthoc(ctx, top_k=3)

        assert any(name in ["tukey_hsd", "games_howell"] for name in posthoc)

    def test_recommend_games_howell_without_equal_variance(self):
        """Test that Games-Howell is recommended without equal variance."""
        ctx = StatContext(
            n_groups=3,
            sample_sizes=[20, 20, 20],
            outcome_type="continuous",
            design="between",
            normality_ok=True,
            variance_homogeneity_ok=False,
        )

        posthoc = recommend_posthoc(ctx, top_k=3)

        assert "games_howell" in posthoc
        assert "tukey_hsd" not in posthoc

    def test_no_posthoc_for_two_groups(self):
        """Test that no post-hoc tests are recommended for 2 groups."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        posthoc = recommend_posthoc(ctx, top_k=3)

        assert len(posthoc) == 0


class TestTooltipsAndReasons:
    """Tests for tooltip generation and reason messages."""

    def test_tooltip_contains_specific_reason(self):
        """Test that tooltip contains specific reason for rejection."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
            normality_ok=False,
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        tooltip = "; ".join(reasons)
        assert "normality" in tooltip.lower()

    def test_tooltip_multiple_reasons_joined(self):
        """Test that multiple reasons are joined with semicolons."""
        ctx = StatContext(
            n_groups=1,
            sample_sizes=[5],
            outcome_type="categorical",
            design="between",
        )
        rule = TEST_RULES["ttest_ind"]

        ok, reasons = check_applicable(rule, ctx)

        assert ok is False
        assert len(reasons) >= 2
        tooltip = "; ".join(reasons)
        assert ";" in tooltip

    def test_no_tooltip_when_enabled(self):
        """Test that tooltip is None when test is enabled."""
        ctx = StatContext(
            n_groups=2,
            sample_sizes=[30, 32],
            outcome_type="continuous",
            design="between",
        )

        items = get_menu_items(ctx)
        bm_item = next(i for i in items if i["id"] == "brunner_munzel")

        assert bm_item["enabled"] is True
        assert bm_item["tooltip"] is None

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_selector.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_selector.py
# 
# """
# Test Selector - Automatic statistical test selection engine.
# 
# This module provides the core logic for determining which statistical tests
# are applicable to a given context, generating UI menu items, and recommending
# tests based on priority.
# 
# Key Functions:
# - check_applicable(): Check if a test is applicable to a context
# - get_menu_items(): Generate UI menu items for right-click menus
# - recommend_tests(): Get recommended tests sorted by priority
# - run_all_applicable_tests(): Run all applicable tests in parallel
# 
# The Brunner-Munzel test is the recommended default for 2-group comparisons
# due to its robustness (no normality or equal variance assumptions).
# """
# 
# from __future__ import annotations
# 
# from concurrent.futures import ThreadPoolExecutor
# from typing import Dict, List, Optional, Tuple, Callable, Any
# 
# from ._context import StatContext
# from ._rules import TestRule, TEST_RULES, TestFamily
# 
# 
# # =============================================================================
# # Pretty Labels for UI
# # =============================================================================
# 
# _PRETTY_LABELS: Dict[str, str] = {
#     # Parametric
#     "ttest_ind": "t-test (independent)",
#     "ttest_rel": "t-test (paired)",
#     "anova_oneway": "One-way ANOVA",
#     "anova_rm_oneway": "Repeated-measures ANOVA",
#     "anova_twoway": "Two-way ANOVA",
#     "anova_twoway_mixed": "Mixed-design ANOVA",
#     "welch_anova": "Welch's ANOVA",
# 
#     # Nonparametric
#     "brunner_munzel": "Brunner-Munzel test (recommended)",
#     "mannwhitneyu": "Mann-Whitney U",
#     "wilcoxon": "Wilcoxon signed-rank",
#     "kruskal": "Kruskal-Wallis",
#     "friedman": "Friedman test",
# 
#     # Categorical
#     "chi2_independence": "Chi-square test",
#     "fisher_exact": "Fisher's exact test",
#     "mcnemar": "McNemar's test",
# 
#     # Correlation
#     "pearsonr": "Pearson correlation",
#     "spearmanr": "Spearman correlation",
# 
#     # Normality/Other
#     "shapiro": "Shapiro-Wilk (normality)",
#     "levene": "Levene's test (variance)",
# 
#     # Posthoc
#     "tukey_hsd": "Tukey HSD",
#     "dunnett": "Dunnett's test (vs control)",
#     "games_howell": "Games-Howell",
# 
#     # Effect sizes
#     "cohens_d_ind": "Cohen's d (independent)",
#     "cohens_d_paired": "Cohen's d (paired)",
#     "hedges_g": "Hedges' g",
#     "cliffs_delta": "Cliff's delta",
#     "eta_squared": "Eta-squared (eta^2)",
#     "partial_eta_squared": "Partial eta-squared",
#     "effect_size_r": "Effect size r",
#     "odds_ratio": "Odds ratio",
#     "risk_ratio": "Risk ratio",
#     "prob_superiority": "P(X>Y) superiority",
# }
# 
# 
# def _pretty_label(name: str) -> str:
#     """Get human-readable label for a test name."""
#     return _PRETTY_LABELS.get(name, name)
# 
# 
# # =============================================================================
# # Core Applicability Check
# # =============================================================================
# 
# 
# def check_applicable(
#     rule: TestRule,
#     ctx: StatContext,
# ) -> Tuple[bool, List[str]]:
#     """
#     Check whether a given statistical test is applicable to the context.
# 
#     This function evaluates all conditions in the TestRule against the
#     StatContext and returns both the result and human-readable reasons
#     for any failures (suitable for tooltips).
# 
#     Parameters
#     ----------
#     rule : TestRule
#         The rule definition for a specific test.
#     ctx : StatContext
#         The context inferred from the figure and data.
# 
#     Returns
#     -------
#     ok : bool
#         True if applicable, False otherwise.
#     reasons : list of str
#         If not applicable, human-readable reasons for tooltips.
# 
#     Examples
#     --------
#     >>> from scitex.stats.auto import StatContext, TEST_RULES, check_applicable
#     >>> ctx = StatContext(
#     ...     n_groups=2,
#     ...     sample_sizes=[30, 32],
#     ...     outcome_type="continuous",
#     ...     design="between",
#     ...     paired=False,
#     ...     has_control_group=False,
#     ...     n_factors=1
#     ... )
#     >>> rule = TEST_RULES["ttest_ind"]
#     >>> ok, reasons = check_applicable(rule, ctx)
#     >>> ok
#     True
# 
#     >>> ctx.normality_ok = False
#     >>> ok, reasons = check_applicable(rule, ctx)
#     >>> ok
#     False
#     >>> "normality" in reasons[0].lower()
#     True
#     """
#     reasons: List[str] = []
# 
#     # Number of groups
#     if ctx.n_groups < rule.min_groups:
#         reasons.append(
#             f"Requires at least {rule.min_groups} groups "
#             f"(current: {ctx.n_groups})"
#         )
#     if rule.max_groups is not None and ctx.n_groups > rule.max_groups:
#         reasons.append(
#             f"Maximum {rule.max_groups} groups allowed "
#             f"(current: {ctx.n_groups})"
#         )
# 
#     # Outcome type
#     if ctx.outcome_type not in rule.outcome_types:
#         allowed = ", ".join(sorted(rule.outcome_types))
#         reasons.append(
#             f"This test is for {allowed} data "
#             f"(current: {ctx.outcome_type})"
#         )
# 
#     # Paired / unpaired
#     effective_paired = ctx.effective_paired
#     if effective_paired is True and not rule.supports_paired:
#         reasons.append("This test does not support paired/repeated measures")
#     if effective_paired is False and not rule.supports_unpaired:
#         reasons.append("This test does not support independent groups")
# 
#     # Design
#     if ctx.design not in rule.design_allowed:
#         allowed = ", ".join(sorted(rule.design_allowed))
#         reasons.append(
#             f"Design '{ctx.design}' not supported "
#             f"(allowed: {allowed})"
#         )
# 
#     # Sample sizes
#     if rule.min_n_total is not None:
#         n_total = ctx.n_total
#         if n_total < rule.min_n_total:
#             reasons.append(
#                 f"Sample size too small (need >= {rule.min_n_total}, "
#                 f"current: {n_total})"
#             )
# 
#     if rule.min_n_per_group is not None:
#         min_n = ctx.min_n_per_group
#         if min_n < rule.min_n_per_group:
#             reasons.append(
#                 f"Each group needs n >= {rule.min_n_per_group} "
#                 f"(smallest group: {min_n})"
#             )
# 
#     # Normality assumption
#     if rule.needs_normality and ctx.normality_ok is False:
#         reasons.append(
#             "Normality assumption not met (consider nonparametric test)"
#         )
# 
#     # Equal variance assumption
#     if rule.needs_equal_variance and ctx.variance_homogeneity_ok is False:
#         reasons.append(
#             "Equal variance assumption not met (consider Welch or nonparametric)"
#         )
# 
#     # Control group requirement
#     if rule.requires_control_group and not ctx.has_control_group:
#         reasons.append("This test requires a designated control group")
# 
#     # Factor constraints
#     if rule.min_factors is not None and ctx.n_factors < rule.min_factors:
#         reasons.append(
#             f"Requires at least {rule.min_factors} factor(s) "
#             f"(current: {ctx.n_factors})"
#         )
#     if rule.max_factors is not None and ctx.n_factors > rule.max_factors:
#         reasons.append(
#             f"Maximum {rule.max_factors} factor(s) allowed "
#             f"(current: {ctx.n_factors})"
#         )
# 
#     ok = len(reasons) == 0
#     return ok, reasons
# 
# 
# # =============================================================================
# # Menu Generation
# # =============================================================================
# 
# 
# def get_menu_items(
#     ctx: StatContext,
#     include_families: Optional[List[TestFamily]] = None,
#     exclude_families: Optional[List[TestFamily]] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Build UI menu items for the given statistical context.
# 
#     Returns a list of menu item dictionaries suitable for right-click
#     context menus. Enabled items are sorted to the top, then by priority.
# 
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     include_families : list of TestFamily or None
#         If provided, only tests whose family is in this list will be considered.
#     exclude_families : list of TestFamily or None
#         If provided, tests whose family is in this list will be skipped.
# 
#     Returns
#     -------
#     items : list of dict
#         Each item has:
#         - id (str): internal test name
#         - label (str): human-readable label
#         - family (str): test family
#         - enabled (bool): whether this test is applicable
#         - tooltip (str or None): reason why disabled (if any)
#         - priority (int): for sorting/recommendation
# 
#     Examples
#     --------
#     >>> ctx = StatContext(
#     ...     n_groups=2,
#     ...     sample_sizes=[30, 32],
#     ...     outcome_type="continuous",
#     ...     design="between",
#     ...     paired=False,
#     ...     has_control_group=False,
#     ...     n_factors=1
#     ... )
#     >>> items = get_menu_items(ctx)
#     >>> enabled_items = [i for i in items if i["enabled"]]
#     >>> len(enabled_items) > 0
#     True
#     """
#     items: List[Dict[str, Any]] = []
#     include_set = set(include_families or [])
#     exclude_set = set(exclude_families or [])
# 
#     for name, rule in TEST_RULES.items():
#         # Family-based filtering
#         if include_set and rule.family not in include_set:
#             continue
#         if rule.family in exclude_set:
#             continue
# 
#         ok, reasons = check_applicable(rule, ctx)
#         tooltip = None if ok else "; ".join(reasons)
# 
#         items.append({
#             "id": name,
#             "label": _pretty_label(name),
#             "family": rule.family,
#             "enabled": ok,
#             "tooltip": tooltip,
#             "priority": rule.priority,
#         })
# 
#     # Sort: enabled first, then by priority (desc), then label
#     items.sort(
#         key=lambda d: (
#             not d["enabled"],      # False (enabled) -> 0 -> top
#             -int(d["priority"]),
#             d["label"],
#         )
#     )
#     return items
# 
# 
# # =============================================================================
# # Test Recommendation
# # =============================================================================
# 
# 
# def recommend_tests(
#     ctx: StatContext,
#     top_k: int = 3,
#     families: Optional[List[TestFamily]] = None,
# ) -> List[str]:
#     """
#     Recommend tests for the given context.
# 
#     Returns test names sorted by priority. Brunner-Munzel is the
#     recommended default for 2-group comparisons (priority 110).
# 
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     top_k : int
#         Number of top tests to return.
#     families : list of TestFamily or None
#         Families to consider. If None, uses standard test families
#         (parametric, nonparametric, categorical, correlation).
# 
#     Returns
#     -------
#     test_names : list of str
#         Internal names of recommended tests, sorted by priority.
# 
#     Examples
#     --------
#     >>> ctx = StatContext(
#     ...     n_groups=2,
#     ...     sample_sizes=[30, 32],
#     ...     outcome_type="continuous",
#     ...     design="between",
#     ...     paired=False,
#     ...     has_control_group=False,
#     ...     n_factors=1
#     ... )
#     >>> recommended = recommend_tests(ctx, top_k=3)
#     >>> "brunner_munzel" in recommended
#     True
#     """
#     if families is None:
#         families = [
#             "parametric",
#             "nonparametric",
#             "categorical",
#             "correlation",
#         ]
# 
#     families_set = set(families)
#     candidates: List[Tuple[int, str]] = []
# 
#     for name, rule in TEST_RULES.items():
#         if rule.family not in families_set:
#             continue
# 
#         ok, _ = check_applicable(rule, ctx)
#         if not ok:
#             continue
# 
#         candidates.append((rule.priority, name))
# 
#     # Sort by priority (high -> first)
#     candidates.sort(reverse=True)
# 
#     return [name for _, name in candidates[:top_k]]
# 
# 
# def recommend_effect_sizes(
#     ctx: StatContext,
#     top_k: int = 3,
# ) -> List[str]:
#     """
#     Recommend effect size measures for the given context.
# 
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     top_k : int
#         Number of top effect sizes to return.
# 
#     Returns
#     -------
#     effect_names : list of str
#         Internal names of recommended effect sizes.
#     """
#     return recommend_tests(ctx, top_k=top_k, families=["effect_size"])
# 
# 
# def recommend_posthoc(
#     ctx: StatContext,
#     top_k: int = 2,
# ) -> List[str]:
#     """
#     Recommend post-hoc tests for the given context.
# 
#     Parameters
#     ----------
#     ctx : StatContext
#         Context inferred from figure/data.
#     top_k : int
#         Number of top post-hoc tests to return.
# 
#     Returns
#     -------
#     posthoc_names : list of str
#         Internal names of recommended post-hoc tests.
#     """
#     return recommend_tests(ctx, top_k=top_k, families=["posthoc"])
# 
# 
# # =============================================================================
# # Parallel Test Execution
# # =============================================================================
# 
# 
# def run_all_applicable_tests(
#     ctx: StatContext,
#     data: Any,
#     test_backend: Dict[str, Callable],
#     families: Optional[List[TestFamily]] = None,
#     max_workers: Optional[int] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Run all applicable statistical tests in parallel.
# 
#     Executes all tests that pass check_applicable() using a thread pool,
#     and returns results sorted by priority.
# 
#     Parameters
#     ----------
#     ctx : StatContext
#         Statistical context.
#     data : Any
#         Data to pass to test functions (typically StatData or similar).
#     test_backend : dict
#         Dictionary mapping test names to callable functions.
#         Each function should accept data and return a result dict.
#     families : list of TestFamily or None
#         Families to include. Defaults to standard test families.
#     max_workers : int or None
#         Maximum number of parallel workers. None uses default.
# 
#     Returns
#     -------
#     results : list of dict
#         Test results sorted by priority (highest first).
#         Each result includes at least 'test_name' key.
# 
#     Examples
#     --------
#     >>> # Define test backends
#     >>> def ttest_backend(data):
#     ...     from scipy import stats
#     ...     stat, p = stats.ttest_ind(data.group1, data.group2)
#     ...     return {"test_name": "ttest_ind", "stat": stat, "p_raw": p}
#     >>>
#     >>> backends = {"ttest_ind": ttest_backend}
#     >>> # results = run_all_applicable_tests(ctx, data, backends)
#     """
#     if families is None:
#         families = [
#             "parametric",
#             "nonparametric",
#             "categorical",
#             "correlation",
#         ]
# 
#     families_set = set(families)
#     tasks: List[Tuple[str, int]] = []
# 
#     # Find applicable tests
#     for name, rule in TEST_RULES.items():
#         if rule.family not in families_set:
#             continue
# 
#         ok, _ = check_applicable(rule, ctx)
#         if not ok:
#             continue
# 
#         if name not in test_backend:
#             continue
# 
#         tasks.append((name, rule.priority))
# 
#     results: List[Dict[str, Any]] = []
# 
#     def run_single(name: str) -> Dict[str, Any]:
#         """Run a single test and handle errors."""
#         try:
#             return test_backend[name](data)
#         except Exception as e:
#             return {
#                 "test_name": name,
#                 "p_raw": None,
#                 "stat": None,
#                 "error": str(e),
#             }
# 
#     # Run in parallel
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(run_single, name): (name, priority)
#             for name, priority in tasks
#         }
#         for future in futures:
#             result = future.result()
#             results.append(result)
# 
#     # Sort by priority (high -> first)
#     def get_priority(r: Dict) -> int:
#         test_name = r.get("test_name", "")
#         rule = TEST_RULES.get(test_name)
#         return rule.priority if rule else 0
# 
#     results.sort(key=get_priority, reverse=True)
#     return results
# 
# 
# # =============================================================================
# # Public API
# # =============================================================================
# 
# __all__ = [
#     "check_applicable",
#     "get_menu_items",
#     "recommend_tests",
#     "recommend_effect_sizes",
#     "recommend_posthoc",
#     "run_all_applicable_tests",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_selector.py
# --------------------------------------------------------------------------------
