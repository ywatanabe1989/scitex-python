#!/usr/bin/env python3
"""
Statistical tests module.

Submodules:
- parametric: t-tests, ANOVA (one-way, two-way, repeated measures)
- nonparametric: Brunner-Munzel, Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
- correlation: Pearson, Spearman, Kendall, Theil-Sen
- normality: Shapiro-Wilk, Kolmogorov-Smirnov
- categorical: Chi-square, Fisher exact, McNemar, Cochran's Q

All 23 test functions are also available directly::

    from scitex.stats.tests import test_ttest_ind, test_pearson
"""

from . import categorical, correlation, nonparametric, normality, parametric

# Categorical tests (4)
from .categorical import test_chi2, test_cochran_q, test_fisher, test_mcnemar

# Correlation tests (4)
from .correlation import test_kendall, test_pearson, test_spearman, test_theilsen

# Nonparametric tests (5)
from .nonparametric import (
    test_brunner_munzel,
    test_friedman,
    test_kruskal,
    test_mannwhitneyu,
    test_wilcoxon,
)

# Normality tests (4)
from .normality import test_ks_1samp, test_ks_2samp, test_normality, test_shapiro

# Parametric tests (6)
from .parametric import (
    test_anova,
    test_anova_2way,
    test_anova_rm,
    test_ttest_1samp,
    test_ttest_ind,
    test_ttest_rel,
)

__all__ = [
    # Category modules
    "categorical",
    "correlation",
    "nonparametric",
    "normality",
    "parametric",
    # Parametric tests (6)
    "test_ttest_ind",
    "test_ttest_rel",
    "test_ttest_1samp",
    "test_anova",
    "test_anova_rm",
    "test_anova_2way",
    # Nonparametric tests (5)
    "test_brunner_munzel",
    "test_wilcoxon",
    "test_kruskal",
    "test_mannwhitneyu",
    "test_friedman",
    # Correlation tests (4)
    "test_pearson",
    "test_spearman",
    "test_kendall",
    "test_theilsen",
    # Categorical tests (4)
    "test_chi2",
    "test_fisher",
    "test_mcnemar",
    "test_cochran_q",
    # Normality tests (4)
    "test_shapiro",
    "test_normality",
    "test_ks_1samp",
    "test_ks_2samp",
]
