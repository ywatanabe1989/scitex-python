#!/usr/bin/env python3
# Timestamp: "2025-12-10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/stats/auto/_symbols.py

"""
Statistical Symbol Mapping - Test statistic symbol definitions.

This module provides mappings from test names to their statistical symbols
(e.g., 't', 'F', 'χ²', 'r', etc.) for use in formatted output.
"""

from __future__ import annotations

from typing import Dict

# =============================================================================
# Statistic Symbol Mapping
# =============================================================================

_STAT_SYMBOLS: Dict[str, str] = {
    # Parametric
    "ttest_ind": "t",
    "ttest_rel": "t",
    "ttest_1samp": "t",
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
    "chi2": "χ²",
    "fisher_exact": "OR",
    "mcnemar": "chi2",
    # Correlation
    "pearsonr": "r",
    "spearmanr": "r",
    # Normality
    "shapiro": "W",
    "ks_1samp": "D",
    "ks_2samp": "D",
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
# Public API
# =============================================================================

__all__ = [
    "get_stat_symbol",
]

# EOF
