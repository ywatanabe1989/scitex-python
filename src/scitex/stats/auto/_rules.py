#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_rules.py

"""
Test Rules - Applicability rules for statistical tests.

This module defines TestRule dataclass and the TEST_RULES registry that
maps test names to their applicability conditions. Used by check_applicable()
to determine which tests can be applied to a given StatContext.

The priority field is used for test recommendation - higher priority tests
are recommended first when multiple tests are applicable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Set

# =============================================================================
# Type Aliases
# =============================================================================

TestFamily = Literal[
    "parametric",
    "nonparametric",
    "categorical",
    "correlation",
    "normality",
    "effect_size",
    "posthoc",
    "other",
]


# =============================================================================
# TestRule
# =============================================================================


@dataclass
class TestRule:
    """
    Applicability rule for a specific statistical test.

    Each TestRule defines the conditions under which a test is applicable.
    The check_applicable() function uses these rules to filter tests
    for a given StatContext.

    Parameters
    ----------
    name : str
        Internal name of the test (e.g., "ttest_ind", "brunner_munzel").
    family : TestFamily
        High-level family of the test:
        - "parametric": t-test, ANOVA, etc.
        - "nonparametric": Mann-Whitney, Kruskal-Wallis, etc.
        - "categorical": Chi-square, Fisher's exact, etc.
        - "correlation": Pearson, Spearman, etc.
        - "normality": Shapiro-Wilk, etc.
        - "effect_size": Cohen's d, eta-squared, etc.
        - "posthoc": Tukey, Dunnett, etc.
        - "other": Other tests (Levene, etc.)
    min_groups : int
        Minimum required number of groups.
    max_groups : int or None
        Maximum allowed number of groups. None means no upper bound.
    outcome_types : set of str
        Allowed outcome types for this test.
    supports_paired : bool
        Whether the test supports paired/repeated measures.
    supports_unpaired : bool
        Whether the test supports independent groups.
    design_allowed : set of str
        Allowed designs, e.g., {"between", "within"}.
    requires_control_group : bool
        Whether a dedicated control group is required (e.g., Dunnett).
    min_n_total : int or None
        Minimum total sample size. None means no constraint.
    min_n_per_group : int or None
        Minimum sample size per group.
    needs_normality : bool
        Whether test assumes normality (check normality_ok).
    needs_equal_variance : bool
        Whether test assumes equal variances (check variance_homogeneity_ok).
    min_factors : int or None
        Minimum number of factors.
    max_factors : int or None
        Maximum number of factors.
    priority : int
        Priority score for recommendation. Higher = more recommended.
        Brunner-Munzel has priority 110 as the recommended default for 2 groups.
    description : str
        Human-readable description for tooltips.

    Examples
    --------
    >>> rule = TestRule(
    ...     name="ttest_ind",
    ...     family="parametric",
    ...     min_groups=2,
    ...     max_groups=2,
    ...     outcome_types={"continuous"},
    ...     supports_paired=False,
    ...     supports_unpaired=True,
    ...     design_allowed={"between"},
    ...     requires_control_group=False,
    ...     min_n_total=4,
    ...     min_n_per_group=2,
    ...     needs_normality=True,
    ...     needs_equal_variance=False,
    ...     min_factors=1,
    ...     max_factors=1,
    ...     priority=90,
    ...     description="Independent samples t-test (Welch)"
    ... )
    """

    name: str
    family: TestFamily
    min_groups: int
    max_groups: Optional[int]
    outcome_types: Set[str]
    supports_paired: bool
    supports_unpaired: bool
    design_allowed: Set[str]
    requires_control_group: bool
    min_n_total: Optional[int]
    min_n_per_group: Optional[int]
    needs_normality: bool
    needs_equal_variance: bool
    min_factors: Optional[int]
    max_factors: Optional[int]
    priority: int = 0
    description: str = ""


# =============================================================================
# TEST_RULES Registry
# =============================================================================

TEST_RULES: Dict[str, TestRule] = {
    # =========================================================================
    # Parametric Tests - Mean Comparisons
    # =========================================================================

    # Independent 2-sample t-test (Welch)
    "ttest_ind": TestRule(
        name="ttest_ind",
        family="parametric",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=False,  # Welch doesn't require equal variance
        min_factors=1,
        max_factors=1,
        priority=90,
        description="Independent samples t-test (Welch)",
    ),

    # Paired t-test
    "ttest_rel": TestRule(
        name="ttest_rel",
        family="parametric",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=False,
        design_allowed={"within"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=95,
        description="Paired samples t-test",
    ),

    # One-way ANOVA (between)
    "anova_oneway": TestRule(
        name="anova_oneway",
        family="parametric",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=True,
        min_factors=1,
        max_factors=1,
        priority=80,
        description="One-way ANOVA (between subjects)",
    ),

    # Repeated-measures one-way ANOVA
    "anova_rm_oneway": TestRule(
        name="anova_rm_oneway",
        family="parametric",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=False,
        design_allowed={"within"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=True,
        min_factors=1,
        max_factors=1,
        priority=85,
        description="Repeated-measures one-way ANOVA",
    ),

    # Welch ANOVA (unequal variances)
    "welch_anova": TestRule(
        name="welch_anova",
        family="parametric",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=False,  # Welch doesn't require equal variance
        min_factors=1,
        max_factors=1,
        priority=82,
        description="Welch's ANOVA (heterogeneous variances)",
    ),

    # Two-way ANOVA (between)
    "anova_twoway": TestRule(
        name="anova_twoway",
        family="parametric",
        min_groups=2,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=8,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=True,
        min_factors=2,
        max_factors=2,
        priority=78,
        description="Two-way ANOVA (between subjects)",
    ),

    # Two-way ANOVA (mixed)
    "anova_twoway_mixed": TestRule(
        name="anova_twoway_mixed",
        family="parametric",
        min_groups=2,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=True,
        design_allowed={"mixed", "within"},
        requires_control_group=False,
        min_n_total=8,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=True,
        min_factors=2,
        max_factors=2,
        priority=80,
        description="Two-way mixed-design ANOVA",
    ),

    # =========================================================================
    # Nonparametric Tests - Rank Comparisons
    # =========================================================================

    # Brunner-Munzel test (RECOMMENDED DEFAULT for 2 groups)
    "brunner_munzel": TestRule(
        name="brunner_munzel",
        family="nonparametric",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous", "ordinal"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=3,
        needs_normality=False,
        needs_equal_variance=False,  # Most robust - no assumptions
        min_factors=1,
        max_factors=1,
        priority=110,  # HIGHEST PRIORITY - recommended default
        description="Brunner-Munzel test (most robust, recommended)",
    ),

    # Mann-Whitney U test
    "mannwhitneyu": TestRule(
        name="mannwhitneyu",
        family="nonparametric",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous", "ordinal"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=85,
        description="Mann-Whitney U test (rank-sum)",
    ),

    # Wilcoxon signed-rank test (paired)
    "wilcoxon": TestRule(
        name="wilcoxon",
        family="nonparametric",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous", "ordinal"},
        supports_paired=True,
        supports_unpaired=False,
        design_allowed={"within"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=90,
        description="Wilcoxon signed-rank test (paired)",
    ),

    # Kruskal-Wallis (3+ groups, between)
    "kruskal": TestRule(
        name="kruskal",
        family="nonparametric",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous", "ordinal"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=75,
        description="Kruskal-Wallis H test",
    ),

    # Friedman test (3+ groups, within)
    "friedman": TestRule(
        name="friedman",
        family="nonparametric",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous", "ordinal"},
        supports_paired=True,
        supports_unpaired=False,
        design_allowed={"within"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=80,
        description="Friedman test (repeated measures)",
    ),

    # =========================================================================
    # Categorical Tests
    # =========================================================================

    # Chi-square test of independence
    "chi2_independence": TestRule(
        name="chi2_independence",
        family="categorical",
        min_groups=2,
        max_groups=None,
        outcome_types={"binary", "categorical"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=10,
        min_n_per_group=None,  # Uses expected counts
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=None,
        priority=80,
        description="Chi-square test of independence",
    ),

    # Fisher's exact test (2x2)
    "fisher_exact": TestRule(
        name="fisher_exact",
        family="categorical",
        min_groups=2,
        max_groups=2,
        outcome_types={"binary"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=1,
        min_n_per_group=1,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=90,
        description="Fisher's exact test (2x2)",
    ),

    # McNemar's test (paired binary)
    "mcnemar": TestRule(
        name="mcnemar",
        family="categorical",
        min_groups=2,
        max_groups=2,
        outcome_types={"binary"},
        supports_paired=True,
        supports_unpaired=False,
        design_allowed={"within"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=85,
        description="McNemar's test (paired binary)",
    ),

    # =========================================================================
    # Correlation Tests
    # =========================================================================

    # Pearson correlation
    "pearsonr": TestRule(
        name="pearsonr",
        family="correlation",
        min_groups=1,
        max_groups=1,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=True,
        design_allowed={"between", "within", "mixed"},
        requires_control_group=False,
        min_n_total=3,
        min_n_per_group=None,
        needs_normality=True,
        needs_equal_variance=False,
        min_factors=None,
        max_factors=None,
        priority=80,
        description="Pearson correlation coefficient",
    ),

    # Spearman correlation
    "spearmanr": TestRule(
        name="spearmanr",
        family="correlation",
        min_groups=1,
        max_groups=1,
        outcome_types={"continuous", "ordinal"},
        supports_paired=True,
        supports_unpaired=True,
        design_allowed={"between", "within", "mixed"},
        requires_control_group=False,
        min_n_total=3,
        min_n_per_group=None,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=None,
        max_factors=None,
        priority=85,
        description="Spearman rank correlation",
    ),

    # =========================================================================
    # Normality Tests
    # =========================================================================

    # Shapiro-Wilk test
    "shapiro": TestRule(
        name="shapiro",
        family="normality",
        min_groups=1,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=True,
        design_allowed={"between", "within", "mixed"},
        requires_control_group=False,
        min_n_total=3,
        min_n_per_group=None,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=None,
        max_factors=None,
        priority=60,
        description="Shapiro-Wilk normality test",
    ),

    # Levene's test for homogeneity of variance
    "levene": TestRule(
        name="levene",
        family="other",
        min_groups=2,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=None,
        priority=70,
        description="Levene's test for homogeneity of variance",
    ),

    # =========================================================================
    # Post-hoc Tests
    # =========================================================================

    # Tukey HSD
    "tukey_hsd": TestRule(
        name="tukey_hsd",
        family="posthoc",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=True,
        min_factors=1,
        max_factors=1,
        priority=88,
        description="Tukey HSD post-hoc test",
    ),

    # Dunnett (control vs treatments)
    "dunnett": TestRule(
        name="dunnett",
        family="posthoc",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=True,  # REQUIRES control group
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=True,
        min_factors=1,
        max_factors=1,
        priority=86,
        description="Dunnett's test (control vs treatments)",
    ),

    # Games-Howell (unequal variances)
    "games_howell": TestRule(
        name="games_howell",
        family="posthoc",
        min_groups=3,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=6,
        min_n_per_group=2,
        needs_normality=True,
        needs_equal_variance=False,  # Does NOT require equal variance
        min_factors=1,
        max_factors=1,
        priority=89,
        description="Games-Howell post-hoc (unequal variances)",
    ),

    # =========================================================================
    # Effect Size Measures
    # =========================================================================

    # Cohen's d (independent)
    "cohens_d_ind": TestRule(
        name="cohens_d_ind",
        family="effect_size",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=90,
        description="Cohen's d (independent samples)",
    ),

    # Cohen's d (paired)
    "cohens_d_paired": TestRule(
        name="cohens_d_paired",
        family="effect_size",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=False,
        design_allowed={"within"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=92,
        description="Cohen's d (paired samples)",
    ),

    # Hedges' g
    "hedges_g": TestRule(
        name="hedges_g",
        family="effect_size",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=88,
        description="Hedges' g (bias-corrected effect size)",
    ),

    # Cliff's delta
    "cliffs_delta": TestRule(
        name="cliffs_delta",
        family="effect_size",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous", "ordinal"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=86,
        description="Cliff's delta (nonparametric effect size)",
    ),

    # Eta-squared
    "eta_squared": TestRule(
        name="eta_squared",
        family="effect_size",
        min_groups=2,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=True,
        design_allowed={"between", "within"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=80,
        description="Eta-squared (variance explained)",
    ),

    # Partial eta-squared
    "partial_eta_squared": TestRule(
        name="partial_eta_squared",
        family="effect_size",
        min_groups=2,
        max_groups=None,
        outcome_types={"continuous"},
        supports_paired=True,
        supports_unpaired=True,
        design_allowed={"between", "within", "mixed"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=None,
        priority=85,
        description="Partial eta-squared (multi-factor designs)",
    ),

    # Effect size r (for correlations)
    "effect_size_r": TestRule(
        name="effect_size_r",
        family="effect_size",
        min_groups=1,
        max_groups=1,
        outcome_types={"continuous", "ordinal"},
        supports_paired=True,
        supports_unpaired=True,
        design_allowed={"between", "within", "mixed"},
        requires_control_group=False,
        min_n_total=3,
        min_n_per_group=None,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=None,
        max_factors=None,
        priority=80,
        description="Effect size r (correlation)",
    ),

    # Odds ratio
    "odds_ratio": TestRule(
        name="odds_ratio",
        family="effect_size",
        min_groups=2,
        max_groups=2,
        outcome_types={"binary"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=1,
        min_n_per_group=1,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=88,
        description="Odds ratio (2x2 table)",
    ),

    # Risk ratio
    "risk_ratio": TestRule(
        name="risk_ratio",
        family="effect_size",
        min_groups=2,
        max_groups=2,
        outcome_types={"binary"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=1,
        min_n_per_group=1,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=86,
        description="Risk ratio (relative risk)",
    ),

    # Probability of superiority P(X>Y)
    "prob_superiority": TestRule(
        name="prob_superiority",
        family="effect_size",
        min_groups=2,
        max_groups=2,
        outcome_types={"continuous", "ordinal"},
        supports_paired=False,
        supports_unpaired=True,
        design_allowed={"between"},
        requires_control_group=False,
        min_n_total=4,
        min_n_per_group=2,
        needs_normality=False,
        needs_equal_variance=False,
        min_factors=1,
        max_factors=1,
        priority=84,
        description="Probability of superiority P(X>Y)",
    ),
}


# =============================================================================
# Utility Functions
# =============================================================================


def get_test_rule(name: str) -> Optional[TestRule]:
    """
    Get a TestRule by name.

    Parameters
    ----------
    name : str
        Test name (e.g., "ttest_ind", "brunner_munzel").

    Returns
    -------
    TestRule or None
        The TestRule if found, else None.
    """
    return TEST_RULES.get(name)


def list_tests_by_family(family: TestFamily) -> Dict[str, TestRule]:
    """
    Get all tests in a specific family.

    Parameters
    ----------
    family : TestFamily
        Test family to filter by.

    Returns
    -------
    dict
        Dictionary of test name -> TestRule for the family.
    """
    return {
        name: rule for name, rule in TEST_RULES.items()
        if rule.family == family
    }


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "TestRule",
    "TestFamily",
    "TEST_RULES",
    "get_test_rule",
    "list_tests_by_family",
]

# EOF
