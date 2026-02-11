#!/usr/bin/env python3
"""
Statistical tests module.

Submodules:
- parametric: t-tests, ANOVA (one-way, two-way, repeated measures)
- nonparametric: Brunner-Munzel, Mann-Whitney U, Wilcoxon, Kruskal-Wallis, Friedman
- correlation: Pearson, Spearman, Kendall, Theil-Sen
- normality: Shapiro-Wilk, Kolmogorov-Smirnov
- categorical: Chi-square, Fisher exact, McNemar, Cochran's Q
"""

from . import categorical, correlation, nonparametric, normality, parametric

__all__ = [
    "categorical",
    "correlation",
    "nonparametric",
    "normality",
    "parametric",
]
