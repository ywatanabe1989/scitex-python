#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive tests for TestRule dataclass and TEST_RULES registry.

Tests cover:
- TestRule dataclass creation and field validation
- TEST_RULES registry completeness and consistency
- get_test_rule() function
- list_tests_by_family() function
- Priority ordering (brunner_munzel should be highest at 110)
- Family categorization and test coverage
- Test rule attributes for specific tests
"""

import pytest
from scitex.stats.auto._rules import (
    TestRule,
    TEST_RULES,
    get_test_rule,
    list_tests_by_family,
)


class TestTestRuleDataclass:
    """Tests for TestRule dataclass."""

    def test_testrule_creation(self):
        """Test basic TestRule instantiation."""
        rule = TestRule(
            name="test_example",
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
            needs_equal_variance=False,
            min_factors=1,
            max_factors=1,
            priority=90,
            description="Example test rule",
        )

        assert rule.name == "test_example"
        assert rule.family == "parametric"
        assert rule.min_groups == 2
        assert rule.max_groups == 2
        assert rule.outcome_types == {"continuous"}
        assert rule.supports_paired is False
        assert rule.supports_unpaired is True
        assert rule.design_allowed == {"between"}
        assert rule.requires_control_group is False
        assert rule.min_n_total == 4
        assert rule.min_n_per_group == 2
        assert rule.needs_normality is True
        assert rule.needs_equal_variance is False
        assert rule.min_factors == 1
        assert rule.max_factors == 1
        assert rule.priority == 90
        assert rule.description == "Example test rule"

    def test_testrule_defaults(self):
        """Test TestRule with default values."""
        rule = TestRule(
            name="minimal",
            family="parametric",
            min_groups=2,
            max_groups=None,
            outcome_types={"continuous"},
            supports_paired=False,
            supports_unpaired=True,
            design_allowed={"between"},
            requires_control_group=False,
            min_n_total=None,
            min_n_per_group=None,
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )

        assert rule.priority == 0
        assert rule.description == ""

    def test_testrule_with_multiple_outcome_types(self):
        """Test TestRule accepting multiple outcome types."""
        rule = TestRule(
            name="multi_outcome",
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
        )

        assert "continuous" in rule.outcome_types
        assert "ordinal" in rule.outcome_types
        assert len(rule.outcome_types) == 2

    def test_testrule_with_multiple_designs(self):
        """Test TestRule with multiple allowed designs."""
        rule = TestRule(
            name="multi_design",
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
            needs_normality=False,
            needs_equal_variance=False,
            min_factors=None,
            max_factors=None,
        )

        assert len(rule.design_allowed) == 3
        assert "between" in rule.design_allowed
        assert "within" in rule.design_allowed
        assert "mixed" in rule.design_allowed


class TestTESTRULESRegistry:
    """Tests for the TEST_RULES registry."""

    def test_registry_is_dict(self):
        """Test that TEST_RULES is a dictionary."""
        assert isinstance(TEST_RULES, dict)

    def test_registry_not_empty(self):
        """Test that TEST_RULES contains tests."""
        assert len(TEST_RULES) > 0

    def test_registry_has_expected_tests(self):
        """Test that registry contains expected test names."""
        expected_tests = [
            "ttest_ind",
            "ttest_rel",
            "brunner_munzel",
            "mannwhitneyu",
            "wilcoxon",
            "anova_oneway",
            "kruskal",
            "friedman",
            "chi2_independence",
            "fisher_exact",
            "pearsonr",
            "spearmanr",
        ]

        for test_name in expected_tests:
            assert test_name in TEST_RULES, f"{test_name} should be in TEST_RULES"

    def test_all_rules_are_testrule_instances(self):
        """Test that all registry values are TestRule instances."""
        for name, rule in TEST_RULES.items():
            assert isinstance(rule, TestRule), f"{name} should be a TestRule instance"

    def test_all_rules_have_matching_names(self):
        """Test that rule names match their registry keys."""
        for key, rule in TEST_RULES.items():
            assert rule.name == key, f"Rule name '{rule.name}' should match key '{key}'"

    def test_all_rules_have_valid_families(self):
        """Test that all rules have valid family types."""
        valid_families = {
            "parametric",
            "nonparametric",
            "categorical",
            "correlation",
            "normality",
            "effect_size",
            "posthoc",
            "other",
        }

        for name, rule in TEST_RULES.items():
            assert rule.family in valid_families, f"{name} has invalid family: {rule.family}"

    def test_all_rules_have_descriptions(self):
        """Test that all rules have descriptions."""
        for name, rule in TEST_RULES.items():
            assert rule.description, f"{name} should have a description"
            assert isinstance(rule.description, str)

    def test_all_rules_have_priorities(self):
        """Test that all rules have priority values."""
        for name, rule in TEST_RULES.items():
            assert isinstance(rule.priority, int), f"{name} priority should be int"

    def test_parametric_tests_present(self):
        """Test that parametric tests are in the registry."""
        parametric_tests = ["ttest_ind", "ttest_rel", "anova_oneway", "welch_anova"]
        for test in parametric_tests:
            assert test in TEST_RULES
            assert TEST_RULES[test].family == "parametric"

    def test_nonparametric_tests_present(self):
        """Test that nonparametric tests are in the registry."""
        nonparametric_tests = ["brunner_munzel", "mannwhitneyu", "wilcoxon", "kruskal", "friedman"]
        for test in nonparametric_tests:
            assert test in TEST_RULES
            assert TEST_RULES[test].family == "nonparametric"

    def test_categorical_tests_present(self):
        """Test that categorical tests are in the registry."""
        categorical_tests = ["chi2_independence", "fisher_exact", "mcnemar"]
        for test in categorical_tests:
            assert test in TEST_RULES
            assert TEST_RULES[test].family == "categorical"

    def test_correlation_tests_present(self):
        """Test that correlation tests are in the registry."""
        correlation_tests = ["pearsonr", "spearmanr"]
        for test in correlation_tests:
            assert test in TEST_RULES
            assert TEST_RULES[test].family == "correlation"

    def test_posthoc_tests_present(self):
        """Test that post-hoc tests are in the registry."""
        posthoc_tests = ["tukey_hsd", "dunnett", "games_howell"]
        for test in posthoc_tests:
            assert test in TEST_RULES
            assert TEST_RULES[test].family == "posthoc"

    def test_effect_size_tests_present(self):
        """Test that effect size measures are in the registry."""
        effect_size_tests = [
            "cohens_d_ind",
            "cohens_d_paired",
            "hedges_g",
            "cliffs_delta",
            "eta_squared",
        ]
        for test in effect_size_tests:
            assert test in TEST_RULES
            assert TEST_RULES[test].family == "effect_size"


class TestBrunnerMunzelPriority:
    """Tests for Brunner-Munzel test as the recommended default."""

    def test_brunner_munzel_exists(self):
        """Test that brunner_munzel is in the registry."""
        assert "brunner_munzel" in TEST_RULES

    def test_brunner_munzel_priority_is_110(self):
        """Test that Brunner-Munzel has priority 110."""
        assert TEST_RULES["brunner_munzel"].priority == 110

    def test_brunner_munzel_has_highest_priority(self):
        """Test that Brunner-Munzel has the highest priority overall."""
        max_priority = max(rule.priority for rule in TEST_RULES.values())
        assert max_priority == 110
        assert TEST_RULES["brunner_munzel"].priority == max_priority

    def test_brunner_munzel_is_nonparametric(self):
        """Test that Brunner-Munzel is nonparametric."""
        assert TEST_RULES["brunner_munzel"].family == "nonparametric"

    def test_brunner_munzel_is_for_two_groups(self):
        """Test that Brunner-Munzel is for 2-group comparisons."""
        rule = TEST_RULES["brunner_munzel"]
        assert rule.min_groups == 2
        assert rule.max_groups == 2

    def test_brunner_munzel_no_assumptions(self):
        """Test that Brunner-Munzel requires no assumptions."""
        rule = TEST_RULES["brunner_munzel"]
        assert rule.needs_normality is False
        assert rule.needs_equal_variance is False

    def test_brunner_munzel_between_design(self):
        """Test that Brunner-Munzel is for between-subjects design."""
        rule = TEST_RULES["brunner_munzel"]
        assert "between" in rule.design_allowed
        assert rule.supports_unpaired is True
        assert rule.supports_paired is False


class TestGetTestRule:
    """Tests for get_test_rule() function."""

    def test_get_existing_test(self):
        """Test retrieving an existing test rule."""
        rule = get_test_rule("ttest_ind")
        assert rule is not None
        assert rule.name == "ttest_ind"

    def test_get_nonexistent_test(self):
        """Test retrieving a nonexistent test returns None."""
        rule = get_test_rule("nonexistent_test")
        assert rule is None

    def test_get_all_expected_tests(self):
        """Test retrieving all expected tests."""
        test_names = [
            "ttest_ind",
            "brunner_munzel",
            "anova_oneway",
            "kruskal",
            "chi2_independence",
            "pearsonr",
        ]

        for name in test_names:
            rule = get_test_rule(name)
            assert rule is not None
            assert rule.name == name

    def test_get_rule_returns_correct_type(self):
        """Test that get_test_rule returns TestRule instance."""
        rule = get_test_rule("ttest_ind")
        assert isinstance(rule, TestRule)


class TestListTestsByFamily:
    """Tests for list_tests_by_family() function."""

    def test_list_parametric_tests(self):
        """Test listing parametric tests."""
        parametric = list_tests_by_family("parametric")

        assert isinstance(parametric, dict)
        assert len(parametric) > 0
        assert "ttest_ind" in parametric
        assert "anova_oneway" in parametric
        assert all(rule.family == "parametric" for rule in parametric.values())

    def test_list_nonparametric_tests(self):
        """Test listing nonparametric tests."""
        nonparametric = list_tests_by_family("nonparametric")

        assert isinstance(nonparametric, dict)
        assert len(nonparametric) > 0
        assert "brunner_munzel" in nonparametric
        assert "mannwhitneyu" in nonparametric
        assert "kruskal" in nonparametric
        assert all(rule.family == "nonparametric" for rule in nonparametric.values())

    def test_list_categorical_tests(self):
        """Test listing categorical tests."""
        categorical = list_tests_by_family("categorical")

        assert isinstance(categorical, dict)
        assert len(categorical) > 0
        assert "chi2_independence" in categorical
        assert "fisher_exact" in categorical
        assert all(rule.family == "categorical" for rule in categorical.values())

    def test_list_correlation_tests(self):
        """Test listing correlation tests."""
        correlation = list_tests_by_family("correlation")

        assert isinstance(correlation, dict)
        assert len(correlation) > 0
        assert "pearsonr" in correlation
        assert "spearmanr" in correlation
        assert all(rule.family == "correlation" for rule in correlation.values())

    def test_list_posthoc_tests(self):
        """Test listing post-hoc tests."""
        posthoc = list_tests_by_family("posthoc")

        assert isinstance(posthoc, dict)
        assert len(posthoc) > 0
        assert "tukey_hsd" in posthoc
        assert "dunnett" in posthoc
        assert all(rule.family == "posthoc" for rule in posthoc.values())

    def test_list_effect_size_tests(self):
        """Test listing effect size measures."""
        effect_size = list_tests_by_family("effect_size")

        assert isinstance(effect_size, dict)
        assert len(effect_size) > 0
        assert "cohens_d_ind" in effect_size
        assert "eta_squared" in effect_size
        assert all(rule.family == "effect_size" for rule in effect_size.values())

    def test_list_normality_tests(self):
        """Test listing normality tests."""
        normality = list_tests_by_family("normality")

        assert isinstance(normality, dict)
        assert "shapiro" in normality
        assert all(rule.family == "normality" for rule in normality.values())

    def test_list_other_tests(self):
        """Test listing other tests."""
        other = list_tests_by_family("other")

        assert isinstance(other, dict)
        assert "levene" in other
        assert all(rule.family == "other" for rule in other.values())

    def test_list_tests_empty_family(self):
        """Test listing tests from nonexistent family returns empty dict."""
        result = list_tests_by_family("nonexistent_family")
        assert result == {}


class TestSpecificTestRules:
    """Tests for specific test rule attributes."""

    def test_ttest_ind_rule(self):
        """Test t-test (independent) rule attributes."""
        rule = TEST_RULES["ttest_ind"]

        assert rule.family == "parametric"
        assert rule.min_groups == 2
        assert rule.max_groups == 2
        assert "continuous" in rule.outcome_types
        assert rule.supports_unpaired is True
        assert rule.supports_paired is False
        assert "between" in rule.design_allowed
        assert rule.needs_normality is True
        assert rule.needs_equal_variance is False

    def test_ttest_rel_rule(self):
        """Test paired t-test rule attributes."""
        rule = TEST_RULES["ttest_rel"]

        assert rule.family == "parametric"
        assert rule.min_groups == 2
        assert rule.max_groups == 2
        assert rule.supports_paired is True
        assert rule.supports_unpaired is False
        assert "within" in rule.design_allowed
        assert rule.needs_normality is True

    def test_anova_oneway_rule(self):
        """Test one-way ANOVA rule attributes."""
        rule = TEST_RULES["anova_oneway"]

        assert rule.family == "parametric"
        assert rule.min_groups == 3
        assert rule.max_groups is None
        assert rule.supports_unpaired is True
        assert "between" in rule.design_allowed
        assert rule.needs_normality is True
        assert rule.needs_equal_variance is True

    def test_mannwhitneyu_rule(self):
        """Test Mann-Whitney U rule attributes."""
        rule = TEST_RULES["mannwhitneyu"]

        assert rule.family == "nonparametric"
        assert rule.min_groups == 2
        assert rule.max_groups == 2
        assert "continuous" in rule.outcome_types
        assert "ordinal" in rule.outcome_types
        assert rule.needs_normality is False
        assert rule.needs_equal_variance is False

    def test_wilcoxon_rule(self):
        """Test Wilcoxon signed-rank rule attributes."""
        rule = TEST_RULES["wilcoxon"]

        assert rule.family == "nonparametric"
        assert rule.min_groups == 2
        assert rule.max_groups == 2
        assert rule.supports_paired is True
        assert rule.supports_unpaired is False
        assert "within" in rule.design_allowed

    def test_kruskal_rule(self):
        """Test Kruskal-Wallis rule attributes."""
        rule = TEST_RULES["kruskal"]

        assert rule.family == "nonparametric"
        assert rule.min_groups == 3
        assert rule.max_groups is None
        assert rule.supports_unpaired is True
        assert "between" in rule.design_allowed

    def test_friedman_rule(self):
        """Test Friedman rule attributes."""
        rule = TEST_RULES["friedman"]

        assert rule.family == "nonparametric"
        assert rule.min_groups == 3
        assert rule.max_groups is None
        assert rule.supports_paired is True
        assert "within" in rule.design_allowed

    def test_fisher_exact_rule(self):
        """Test Fisher's exact test rule attributes."""
        rule = TEST_RULES["fisher_exact"]

        assert rule.family == "categorical"
        assert rule.min_groups == 2
        assert rule.max_groups == 2
        assert "binary" in rule.outcome_types
        assert rule.needs_normality is False

    def test_dunnett_rule(self):
        """Test Dunnett's test rule attributes."""
        rule = TEST_RULES["dunnett"]

        assert rule.family == "posthoc"
        assert rule.min_groups == 3
        assert rule.requires_control_group is True
        assert rule.needs_normality is True
        assert rule.needs_equal_variance is True

    def test_games_howell_rule(self):
        """Test Games-Howell rule attributes."""
        rule = TEST_RULES["games_howell"]

        assert rule.family == "posthoc"
        assert rule.min_groups == 3
        assert rule.requires_control_group is False
        assert rule.needs_normality is True
        assert rule.needs_equal_variance is False


class TestPriorityOrdering:
    """Tests for priority ordering across tests."""

    def test_all_priorities_are_integers(self):
        """Test that all priorities are integers."""
        for name, rule in TEST_RULES.items():
            assert isinstance(rule.priority, int), f"{name} priority should be int"

    def test_priorities_are_reasonable(self):
        """Test that priorities are in a reasonable range."""
        for name, rule in TEST_RULES.items():
            assert 0 <= rule.priority <= 150, f"{name} priority {rule.priority} out of range"

    def test_brunner_munzel_highest_priority_for_two_groups(self):
        """Test that Brunner-Munzel has highest priority among 2-group tests."""
        two_group_tests = {
            name: rule for name, rule in TEST_RULES.items()
            if rule.min_groups == 2 and rule.max_groups == 2
        }

        priorities = {name: rule.priority for name, rule in two_group_tests.items()}
        max_priority_test = max(priorities, key=priorities.get)

        assert max_priority_test == "brunner_munzel"

    def test_brunner_munzel_has_highest_priority(self):
        """Test that Brunner-Munzel has the highest priority among 2-group tests.

        Brunner-Munzel is the recommended default due to its robustness
        (no normality or equal variance assumptions). Other tests like
        Mann-Whitney and t-test have lower priorities.
        """
        brunner_munzel_priority = TEST_RULES["brunner_munzel"].priority
        mannwhitneyu_priority = TEST_RULES["mannwhitneyu"].priority
        ttest_ind_priority = TEST_RULES["ttest_ind"].priority

        # Brunner-Munzel should be highest (110)
        assert brunner_munzel_priority > ttest_ind_priority
        assert brunner_munzel_priority > mannwhitneyu_priority
        # t-test is slightly higher than Mann-Whitney (90 vs 85)
        assert ttest_ind_priority > mannwhitneyu_priority

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_rules.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-10 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_rules.py
# 
# """
# Test Rules - Applicability rules for statistical tests.
# 
# This module defines TestRule dataclass and the TEST_RULES registry that
# maps test names to their applicability conditions. Used by check_applicable()
# to determine which tests can be applied to a given StatContext.
# 
# The priority field is used for test recommendation - higher priority tests
# are recommended first when multiple tests are applicable.
# """
# 
# from __future__ import annotations
# 
# from dataclasses import dataclass, field
# from typing import Dict, Literal, Optional, Set
# 
# # =============================================================================
# # Type Aliases
# # =============================================================================
# 
# TestFamily = Literal[
#     "parametric",
#     "nonparametric",
#     "categorical",
#     "correlation",
#     "normality",
#     "effect_size",
#     "posthoc",
#     "other",
# ]
# 
# 
# # =============================================================================
# # TestRule
# # =============================================================================
# 
# 
# @dataclass
# class TestRule:
#     """
#     Applicability rule for a specific statistical test.
# 
#     Each TestRule defines the conditions under which a test is applicable.
#     The check_applicable() function uses these rules to filter tests
#     for a given StatContext.
# 
#     Parameters
#     ----------
#     name : str
#         Internal name of the test (e.g., "ttest_ind", "brunner_munzel").
#     family : TestFamily
#         High-level family of the test:
#         - "parametric": t-test, ANOVA, etc.
#         - "nonparametric": Mann-Whitney, Kruskal-Wallis, etc.
#         - "categorical": Chi-square, Fisher's exact, etc.
#         - "correlation": Pearson, Spearman, etc.
#         - "normality": Shapiro-Wilk, etc.
#         - "effect_size": Cohen's d, eta-squared, etc.
#         - "posthoc": Tukey, Dunnett, etc.
#         - "other": Other tests (Levene, etc.)
#     min_groups : int
#         Minimum required number of groups.
#     max_groups : int or None
#         Maximum allowed number of groups. None means no upper bound.
#     outcome_types : set of str
#         Allowed outcome types for this test.
#     supports_paired : bool
#         Whether the test supports paired/repeated measures.
#     supports_unpaired : bool
#         Whether the test supports independent groups.
#     design_allowed : set of str
#         Allowed designs, e.g., {"between", "within"}.
#     requires_control_group : bool
#         Whether a dedicated control group is required (e.g., Dunnett).
#     min_n_total : int or None
#         Minimum total sample size. None means no constraint.
#     min_n_per_group : int or None
#         Minimum sample size per group.
#     needs_normality : bool
#         Whether test assumes normality (check normality_ok).
#     needs_equal_variance : bool
#         Whether test assumes equal variances (check variance_homogeneity_ok).
#     min_factors : int or None
#         Minimum number of factors.
#     max_factors : int or None
#         Maximum number of factors.
#     priority : int
#         Priority score for recommendation. Higher = more recommended.
#         Brunner-Munzel has priority 110 as the recommended default for 2 groups.
#     description : str
#         Human-readable description for tooltips.
# 
#     Examples
#     --------
#     >>> rule = TestRule(
#     ...     name="ttest_ind",
#     ...     family="parametric",
#     ...     min_groups=2,
#     ...     max_groups=2,
#     ...     outcome_types={"continuous"},
#     ...     supports_paired=False,
#     ...     supports_unpaired=True,
#     ...     design_allowed={"between"},
#     ...     requires_control_group=False,
#     ...     min_n_total=4,
#     ...     min_n_per_group=2,
#     ...     needs_normality=True,
#     ...     needs_equal_variance=False,
#     ...     min_factors=1,
#     ...     max_factors=1,
#     ...     priority=90,
#     ...     description="Independent samples t-test (Welch)"
#     ... )
#     """
# 
#     name: str
#     family: TestFamily
#     min_groups: int
#     max_groups: Optional[int]
#     outcome_types: Set[str]
#     supports_paired: bool
#     supports_unpaired: bool
#     design_allowed: Set[str]
#     requires_control_group: bool
#     min_n_total: Optional[int]
#     min_n_per_group: Optional[int]
#     needs_normality: bool
#     needs_equal_variance: bool
#     min_factors: Optional[int]
#     max_factors: Optional[int]
#     priority: int = 0
#     description: str = ""
# 
# 
# # =============================================================================
# # TEST_RULES Registry
# # =============================================================================
# 
# TEST_RULES: Dict[str, TestRule] = {
#     # =========================================================================
#     # Parametric Tests - Mean Comparisons
#     # =========================================================================
# 
#     # Independent 2-sample t-test (Welch)
#     "ttest_ind": TestRule(
#         name="ttest_ind",
#         family="parametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,  # Welch doesn't require equal variance
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Independent samples t-test (Welch)",
#     ),
# 
#     # Paired t-test
#     "ttest_rel": TestRule(
#         name="ttest_rel",
#         family="parametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=95,
#         description="Paired samples t-test",
#     ),
# 
#     # One-way ANOVA (between)
#     "anova_oneway": TestRule(
#         name="anova_oneway",
#         family="parametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=80,
#         description="One-way ANOVA (between subjects)",
#     ),
# 
#     # Repeated-measures one-way ANOVA
#     "anova_rm_oneway": TestRule(
#         name="anova_rm_oneway",
#         family="parametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=85,
#         description="Repeated-measures one-way ANOVA",
#     ),
# 
#     # Welch ANOVA (unequal variances)
#     "welch_anova": TestRule(
#         name="welch_anova",
#         family="parametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,  # Welch doesn't require equal variance
#         min_factors=1,
#         max_factors=1,
#         priority=82,
#         description="Welch's ANOVA (heterogeneous variances)",
#     ),
# 
#     # Two-way ANOVA (between)
#     "anova_twoway": TestRule(
#         name="anova_twoway",
#         family="parametric",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=8,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=2,
#         max_factors=2,
#         priority=78,
#         description="Two-way ANOVA (between subjects)",
#     ),
# 
#     # Two-way ANOVA (mixed)
#     "anova_twoway_mixed": TestRule(
#         name="anova_twoway_mixed",
#         family="parametric",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"mixed", "within"},
#         requires_control_group=False,
#         min_n_total=8,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=2,
#         max_factors=2,
#         priority=80,
#         description="Two-way mixed-design ANOVA",
#     ),
# 
#     # =========================================================================
#     # Nonparametric Tests - Rank Comparisons
#     # =========================================================================
# 
#     # Brunner-Munzel test (RECOMMENDED DEFAULT for 2 groups)
#     "brunner_munzel": TestRule(
#         name="brunner_munzel",
#         family="nonparametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=3,
#         needs_normality=False,
#         needs_equal_variance=False,  # Most robust - no assumptions
#         min_factors=1,
#         max_factors=1,
#         priority=110,  # HIGHEST PRIORITY - recommended default
#         description="Brunner-Munzel test (most robust, recommended)",
#     ),
# 
#     # Mann-Whitney U test
#     "mannwhitneyu": TestRule(
#         name="mannwhitneyu",
#         family="nonparametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=85,
#         description="Mann-Whitney U test (rank-sum)",
#     ),
# 
#     # Wilcoxon signed-rank test (paired)
#     "wilcoxon": TestRule(
#         name="wilcoxon",
#         family="nonparametric",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Wilcoxon signed-rank test (paired)",
#     ),
# 
#     # Kruskal-Wallis (3+ groups, between)
#     "kruskal": TestRule(
#         name="kruskal",
#         family="nonparametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=75,
#         description="Kruskal-Wallis H test",
#     ),
# 
#     # Friedman test (3+ groups, within)
#     "friedman": TestRule(
#         name="friedman",
#         family="nonparametric",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=80,
#         description="Friedman test (repeated measures)",
#     ),
# 
#     # =========================================================================
#     # Categorical Tests
#     # =========================================================================
# 
#     # Chi-square test of independence
#     "chi2_independence": TestRule(
#         name="chi2_independence",
#         family="categorical",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"binary", "categorical"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=10,
#         min_n_per_group=None,  # Uses expected counts
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=None,
#         priority=80,
#         description="Chi-square test of independence",
#     ),
# 
#     # Fisher's exact test (2x2)
#     "fisher_exact": TestRule(
#         name="fisher_exact",
#         family="categorical",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=1,
#         min_n_per_group=1,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Fisher's exact test (2x2)",
#     ),
# 
#     # McNemar's test (paired binary)
#     "mcnemar": TestRule(
#         name="mcnemar",
#         family="categorical",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=85,
#         description="McNemar's test (paired binary)",
#     ),
# 
#     # =========================================================================
#     # Correlation Tests
#     # =========================================================================
# 
#     # Pearson correlation
#     "pearsonr": TestRule(
#         name="pearsonr",
#         family="correlation",
#         min_groups=1,
#         max_groups=1,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=True,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=80,
#         description="Pearson correlation coefficient",
#     ),
# 
#     # Spearman correlation
#     "spearmanr": TestRule(
#         name="spearmanr",
#         family="correlation",
#         min_groups=1,
#         max_groups=1,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=85,
#         description="Spearman rank correlation",
#     ),
# 
#     # =========================================================================
#     # Normality Tests
#     # =========================================================================
# 
#     # Shapiro-Wilk test
#     "shapiro": TestRule(
#         name="shapiro",
#         family="normality",
#         min_groups=1,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=60,
#         description="Shapiro-Wilk normality test",
#     ),
# 
#     # Levene's test for homogeneity of variance
#     "levene": TestRule(
#         name="levene",
#         family="other",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=None,
#         priority=70,
#         description="Levene's test for homogeneity of variance",
#     ),
# 
#     # =========================================================================
#     # Post-hoc Tests
#     # =========================================================================
# 
#     # Tukey HSD
#     "tukey_hsd": TestRule(
#         name="tukey_hsd",
#         family="posthoc",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=88,
#         description="Tukey HSD post-hoc test",
#     ),
# 
#     # Dunnett (control vs treatments)
#     "dunnett": TestRule(
#         name="dunnett",
#         family="posthoc",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=True,  # REQUIRES control group
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=True,
#         min_factors=1,
#         max_factors=1,
#         priority=86,
#         description="Dunnett's test (control vs treatments)",
#     ),
# 
#     # Games-Howell (unequal variances)
#     "games_howell": TestRule(
#         name="games_howell",
#         family="posthoc",
#         min_groups=3,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=6,
#         min_n_per_group=2,
#         needs_normality=True,
#         needs_equal_variance=False,  # Does NOT require equal variance
#         min_factors=1,
#         max_factors=1,
#         priority=89,
#         description="Games-Howell post-hoc (unequal variances)",
#     ),
# 
#     # =========================================================================
#     # Effect Size Measures
#     # =========================================================================
# 
#     # Cohen's d (independent)
#     "cohens_d_ind": TestRule(
#         name="cohens_d_ind",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=90,
#         description="Cohen's d (independent samples)",
#     ),
# 
#     # Cohen's d (paired)
#     "cohens_d_paired": TestRule(
#         name="cohens_d_paired",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=False,
#         design_allowed={"within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=92,
#         description="Cohen's d (paired samples)",
#     ),
# 
#     # Hedges' g
#     "hedges_g": TestRule(
#         name="hedges_g",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=88,
#         description="Hedges' g (bias-corrected effect size)",
#     ),
# 
#     # Cliff's delta
#     "cliffs_delta": TestRule(
#         name="cliffs_delta",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=86,
#         description="Cliff's delta (nonparametric effect size)",
#     ),
# 
#     # Eta-squared
#     "eta_squared": TestRule(
#         name="eta_squared",
#         family="effect_size",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=80,
#         description="Eta-squared (variance explained)",
#     ),
# 
#     # Partial eta-squared
#     "partial_eta_squared": TestRule(
#         name="partial_eta_squared",
#         family="effect_size",
#         min_groups=2,
#         max_groups=None,
#         outcome_types={"continuous"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=None,
#         priority=85,
#         description="Partial eta-squared (multi-factor designs)",
#     ),
# 
#     # Effect size r (for correlations)
#     "effect_size_r": TestRule(
#         name="effect_size_r",
#         family="effect_size",
#         min_groups=1,
#         max_groups=1,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=True,
#         supports_unpaired=True,
#         design_allowed={"between", "within", "mixed"},
#         requires_control_group=False,
#         min_n_total=3,
#         min_n_per_group=None,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=None,
#         max_factors=None,
#         priority=80,
#         description="Effect size r (correlation)",
#     ),
# 
#     # Odds ratio
#     "odds_ratio": TestRule(
#         name="odds_ratio",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=1,
#         min_n_per_group=1,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=88,
#         description="Odds ratio (2x2 table)",
#     ),
# 
#     # Risk ratio
#     "risk_ratio": TestRule(
#         name="risk_ratio",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"binary"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=1,
#         min_n_per_group=1,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=86,
#         description="Risk ratio (relative risk)",
#     ),
# 
#     # Probability of superiority P(X>Y)
#     "prob_superiority": TestRule(
#         name="prob_superiority",
#         family="effect_size",
#         min_groups=2,
#         max_groups=2,
#         outcome_types={"continuous", "ordinal"},
#         supports_paired=False,
#         supports_unpaired=True,
#         design_allowed={"between"},
#         requires_control_group=False,
#         min_n_total=4,
#         min_n_per_group=2,
#         needs_normality=False,
#         needs_equal_variance=False,
#         min_factors=1,
#         max_factors=1,
#         priority=84,
#         description="Probability of superiority P(X>Y)",
#     ),
# }
# 
# 
# # =============================================================================
# # Utility Functions
# # =============================================================================
# 
# 
# def get_test_rule(name: str) -> Optional[TestRule]:
#     """
#     Get a TestRule by name.
# 
#     Parameters
#     ----------
#     name : str
#         Test name (e.g., "ttest_ind", "brunner_munzel").
# 
#     Returns
#     -------
#     TestRule or None
#         The TestRule if found, else None.
#     """
#     return TEST_RULES.get(name)
# 
# 
# def list_tests_by_family(family: TestFamily) -> Dict[str, TestRule]:
#     """
#     Get all tests in a specific family.
# 
#     Parameters
#     ----------
#     family : TestFamily
#         Test family to filter by.
# 
#     Returns
#     -------
#     dict
#         Dictionary of test name -> TestRule for the family.
#     """
#     return {
#         name: rule for name, rule in TEST_RULES.items()
#         if rule.family == family
#     }
# 
# 
# # =============================================================================
# # Public API
# # =============================================================================
# 
# __all__ = [
#     "TestRule",
#     "TestFamily",
#     "TEST_RULES",
#     "get_test_rule",
#     "list_tests_by_family",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_rules.py
# --------------------------------------------------------------------------------
