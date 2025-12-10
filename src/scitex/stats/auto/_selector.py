#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_selector.py

"""
Test Selector - Automatic statistical test selection engine.

This module provides the core logic for determining which statistical tests
are applicable to a given context, generating UI menu items, and recommending
tests based on priority.

Key Functions:
- check_applicable(): Check if a test is applicable to a context
- get_menu_items(): Generate UI menu items for right-click menus
- recommend_tests(): Get recommended tests sorted by priority
- run_all_applicable_tests(): Run all applicable tests in parallel

The Brunner-Munzel test is the recommended default for 2-group comparisons
due to its robustness (no normality or equal variance assumptions).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Callable, Any

from ._context import StatContext
from ._rules import TestRule, TEST_RULES, TestFamily


# =============================================================================
# Pretty Labels for UI
# =============================================================================

_PRETTY_LABELS: Dict[str, str] = {
    # Parametric
    "ttest_ind": "t-test (independent)",
    "ttest_rel": "t-test (paired)",
    "anova_oneway": "One-way ANOVA",
    "anova_rm_oneway": "Repeated-measures ANOVA",
    "anova_twoway": "Two-way ANOVA",
    "anova_twoway_mixed": "Mixed-design ANOVA",
    "welch_anova": "Welch's ANOVA",

    # Nonparametric
    "brunner_munzel": "Brunner-Munzel test (recommended)",
    "mannwhitneyu": "Mann-Whitney U",
    "wilcoxon": "Wilcoxon signed-rank",
    "kruskal": "Kruskal-Wallis",
    "friedman": "Friedman test",

    # Categorical
    "chi2_independence": "Chi-square test",
    "fisher_exact": "Fisher's exact test",
    "mcnemar": "McNemar's test",

    # Correlation
    "pearsonr": "Pearson correlation",
    "spearmanr": "Spearman correlation",

    # Normality/Other
    "shapiro": "Shapiro-Wilk (normality)",
    "levene": "Levene's test (variance)",

    # Posthoc
    "tukey_hsd": "Tukey HSD",
    "dunnett": "Dunnett's test (vs control)",
    "games_howell": "Games-Howell",

    # Effect sizes
    "cohens_d_ind": "Cohen's d (independent)",
    "cohens_d_paired": "Cohen's d (paired)",
    "hedges_g": "Hedges' g",
    "cliffs_delta": "Cliff's delta",
    "eta_squared": "Eta-squared (eta^2)",
    "partial_eta_squared": "Partial eta-squared",
    "effect_size_r": "Effect size r",
    "odds_ratio": "Odds ratio",
    "risk_ratio": "Risk ratio",
    "prob_superiority": "P(X>Y) superiority",
}


def _pretty_label(name: str) -> str:
    """Get human-readable label for a test name."""
    return _PRETTY_LABELS.get(name, name)


# =============================================================================
# Core Applicability Check
# =============================================================================


def check_applicable(
    rule: TestRule,
    ctx: StatContext,
) -> Tuple[bool, List[str]]:
    """
    Check whether a given statistical test is applicable to the context.

    This function evaluates all conditions in the TestRule against the
    StatContext and returns both the result and human-readable reasons
    for any failures (suitable for tooltips).

    Parameters
    ----------
    rule : TestRule
        The rule definition for a specific test.
    ctx : StatContext
        The context inferred from the figure and data.

    Returns
    -------
    ok : bool
        True if applicable, False otherwise.
    reasons : list of str
        If not applicable, human-readable reasons for tooltips.

    Examples
    --------
    >>> from scitex.stats.auto import StatContext, TEST_RULES, check_applicable
    >>> ctx = StatContext(
    ...     n_groups=2,
    ...     sample_sizes=[30, 32],
    ...     outcome_type="continuous",
    ...     design="between",
    ...     paired=False,
    ...     has_control_group=False,
    ...     n_factors=1
    ... )
    >>> rule = TEST_RULES["ttest_ind"]
    >>> ok, reasons = check_applicable(rule, ctx)
    >>> ok
    True

    >>> ctx.normality_ok = False
    >>> ok, reasons = check_applicable(rule, ctx)
    >>> ok
    False
    >>> "normality" in reasons[0].lower()
    True
    """
    reasons: List[str] = []

    # Number of groups
    if ctx.n_groups < rule.min_groups:
        reasons.append(
            f"Requires at least {rule.min_groups} groups "
            f"(current: {ctx.n_groups})"
        )
    if rule.max_groups is not None and ctx.n_groups > rule.max_groups:
        reasons.append(
            f"Maximum {rule.max_groups} groups allowed "
            f"(current: {ctx.n_groups})"
        )

    # Outcome type
    if ctx.outcome_type not in rule.outcome_types:
        allowed = ", ".join(sorted(rule.outcome_types))
        reasons.append(
            f"This test is for {allowed} data "
            f"(current: {ctx.outcome_type})"
        )

    # Paired / unpaired
    effective_paired = ctx.effective_paired
    if effective_paired is True and not rule.supports_paired:
        reasons.append("This test does not support paired/repeated measures")
    if effective_paired is False and not rule.supports_unpaired:
        reasons.append("This test does not support independent groups")

    # Design
    if ctx.design not in rule.design_allowed:
        allowed = ", ".join(sorted(rule.design_allowed))
        reasons.append(
            f"Design '{ctx.design}' not supported "
            f"(allowed: {allowed})"
        )

    # Sample sizes
    if rule.min_n_total is not None:
        n_total = ctx.n_total
        if n_total < rule.min_n_total:
            reasons.append(
                f"Sample size too small (need >= {rule.min_n_total}, "
                f"current: {n_total})"
            )

    if rule.min_n_per_group is not None:
        min_n = ctx.min_n_per_group
        if min_n < rule.min_n_per_group:
            reasons.append(
                f"Each group needs n >= {rule.min_n_per_group} "
                f"(smallest group: {min_n})"
            )

    # Normality assumption
    if rule.needs_normality and ctx.normality_ok is False:
        reasons.append(
            "Normality assumption not met (consider nonparametric test)"
        )

    # Equal variance assumption
    if rule.needs_equal_variance and ctx.variance_homogeneity_ok is False:
        reasons.append(
            "Equal variance assumption not met (consider Welch or nonparametric)"
        )

    # Control group requirement
    if rule.requires_control_group and not ctx.has_control_group:
        reasons.append("This test requires a designated control group")

    # Factor constraints
    if rule.min_factors is not None and ctx.n_factors < rule.min_factors:
        reasons.append(
            f"Requires at least {rule.min_factors} factor(s) "
            f"(current: {ctx.n_factors})"
        )
    if rule.max_factors is not None and ctx.n_factors > rule.max_factors:
        reasons.append(
            f"Maximum {rule.max_factors} factor(s) allowed "
            f"(current: {ctx.n_factors})"
        )

    ok = len(reasons) == 0
    return ok, reasons


# =============================================================================
# Menu Generation
# =============================================================================


def get_menu_items(
    ctx: StatContext,
    include_families: Optional[List[TestFamily]] = None,
    exclude_families: Optional[List[TestFamily]] = None,
) -> List[Dict[str, Any]]:
    """
    Build UI menu items for the given statistical context.

    Returns a list of menu item dictionaries suitable for right-click
    context menus. Enabled items are sorted to the top, then by priority.

    Parameters
    ----------
    ctx : StatContext
        Context inferred from figure/data.
    include_families : list of TestFamily or None
        If provided, only tests whose family is in this list will be considered.
    exclude_families : list of TestFamily or None
        If provided, tests whose family is in this list will be skipped.

    Returns
    -------
    items : list of dict
        Each item has:
        - id (str): internal test name
        - label (str): human-readable label
        - family (str): test family
        - enabled (bool): whether this test is applicable
        - tooltip (str or None): reason why disabled (if any)
        - priority (int): for sorting/recommendation

    Examples
    --------
    >>> ctx = StatContext(
    ...     n_groups=2,
    ...     sample_sizes=[30, 32],
    ...     outcome_type="continuous",
    ...     design="between",
    ...     paired=False,
    ...     has_control_group=False,
    ...     n_factors=1
    ... )
    >>> items = get_menu_items(ctx)
    >>> enabled_items = [i for i in items if i["enabled"]]
    >>> len(enabled_items) > 0
    True
    """
    items: List[Dict[str, Any]] = []
    include_set = set(include_families or [])
    exclude_set = set(exclude_families or [])

    for name, rule in TEST_RULES.items():
        # Family-based filtering
        if include_set and rule.family not in include_set:
            continue
        if rule.family in exclude_set:
            continue

        ok, reasons = check_applicable(rule, ctx)
        tooltip = None if ok else "; ".join(reasons)

        items.append({
            "id": name,
            "label": _pretty_label(name),
            "family": rule.family,
            "enabled": ok,
            "tooltip": tooltip,
            "priority": rule.priority,
        })

    # Sort: enabled first, then by priority (desc), then label
    items.sort(
        key=lambda d: (
            not d["enabled"],      # False (enabled) -> 0 -> top
            -int(d["priority"]),
            d["label"],
        )
    )
    return items


# =============================================================================
# Test Recommendation
# =============================================================================


def recommend_tests(
    ctx: StatContext,
    top_k: int = 3,
    families: Optional[List[TestFamily]] = None,
) -> List[str]:
    """
    Recommend tests for the given context.

    Returns test names sorted by priority. Brunner-Munzel is the
    recommended default for 2-group comparisons (priority 110).

    Parameters
    ----------
    ctx : StatContext
        Context inferred from figure/data.
    top_k : int
        Number of top tests to return.
    families : list of TestFamily or None
        Families to consider. If None, uses standard test families
        (parametric, nonparametric, categorical, correlation).

    Returns
    -------
    test_names : list of str
        Internal names of recommended tests, sorted by priority.

    Examples
    --------
    >>> ctx = StatContext(
    ...     n_groups=2,
    ...     sample_sizes=[30, 32],
    ...     outcome_type="continuous",
    ...     design="between",
    ...     paired=False,
    ...     has_control_group=False,
    ...     n_factors=1
    ... )
    >>> recommended = recommend_tests(ctx, top_k=3)
    >>> "brunner_munzel" in recommended
    True
    """
    if families is None:
        families = [
            "parametric",
            "nonparametric",
            "categorical",
            "correlation",
        ]

    families_set = set(families)
    candidates: List[Tuple[int, str]] = []

    for name, rule in TEST_RULES.items():
        if rule.family not in families_set:
            continue

        ok, _ = check_applicable(rule, ctx)
        if not ok:
            continue

        candidates.append((rule.priority, name))

    # Sort by priority (high -> first)
    candidates.sort(reverse=True)

    return [name for _, name in candidates[:top_k]]


def recommend_effect_sizes(
    ctx: StatContext,
    top_k: int = 3,
) -> List[str]:
    """
    Recommend effect size measures for the given context.

    Parameters
    ----------
    ctx : StatContext
        Context inferred from figure/data.
    top_k : int
        Number of top effect sizes to return.

    Returns
    -------
    effect_names : list of str
        Internal names of recommended effect sizes.
    """
    return recommend_tests(ctx, top_k=top_k, families=["effect_size"])


def recommend_posthoc(
    ctx: StatContext,
    top_k: int = 2,
) -> List[str]:
    """
    Recommend post-hoc tests for the given context.

    Parameters
    ----------
    ctx : StatContext
        Context inferred from figure/data.
    top_k : int
        Number of top post-hoc tests to return.

    Returns
    -------
    posthoc_names : list of str
        Internal names of recommended post-hoc tests.
    """
    return recommend_tests(ctx, top_k=top_k, families=["posthoc"])


# =============================================================================
# Parallel Test Execution
# =============================================================================


def run_all_applicable_tests(
    ctx: StatContext,
    data: Any,
    test_backend: Dict[str, Callable],
    families: Optional[List[TestFamily]] = None,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run all applicable statistical tests in parallel.

    Executes all tests that pass check_applicable() using a thread pool,
    and returns results sorted by priority.

    Parameters
    ----------
    ctx : StatContext
        Statistical context.
    data : Any
        Data to pass to test functions (typically StatData or similar).
    test_backend : dict
        Dictionary mapping test names to callable functions.
        Each function should accept data and return a result dict.
    families : list of TestFamily or None
        Families to include. Defaults to standard test families.
    max_workers : int or None
        Maximum number of parallel workers. None uses default.

    Returns
    -------
    results : list of dict
        Test results sorted by priority (highest first).
        Each result includes at least 'test_name' key.

    Examples
    --------
    >>> # Define test backends
    >>> def ttest_backend(data):
    ...     from scipy import stats
    ...     stat, p = stats.ttest_ind(data.group1, data.group2)
    ...     return {"test_name": "ttest_ind", "stat": stat, "p_raw": p}
    >>>
    >>> backends = {"ttest_ind": ttest_backend}
    >>> # results = run_all_applicable_tests(ctx, data, backends)
    """
    if families is None:
        families = [
            "parametric",
            "nonparametric",
            "categorical",
            "correlation",
        ]

    families_set = set(families)
    tasks: List[Tuple[str, int]] = []

    # Find applicable tests
    for name, rule in TEST_RULES.items():
        if rule.family not in families_set:
            continue

        ok, _ = check_applicable(rule, ctx)
        if not ok:
            continue

        if name not in test_backend:
            continue

        tasks.append((name, rule.priority))

    results: List[Dict[str, Any]] = []

    def run_single(name: str) -> Dict[str, Any]:
        """Run a single test and handle errors."""
        try:
            return test_backend[name](data)
        except Exception as e:
            return {
                "test_name": name,
                "p_raw": None,
                "stat": None,
                "error": str(e),
            }

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single, name): (name, priority)
            for name, priority in tasks
        }
        for future in futures:
            result = future.result()
            results.append(result)

    # Sort by priority (high -> first)
    def get_priority(r: Dict) -> int:
        test_name = r.get("test_name", "")
        rule = TEST_RULES.get(test_name)
        return rule.priority if rule else 0

    results.sort(key=get_priority, reverse=True)
    return results


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "check_applicable",
    "get_menu_items",
    "recommend_tests",
    "recommend_effect_sizes",
    "recommend_posthoc",
    "run_all_applicable_tests",
]

# EOF
