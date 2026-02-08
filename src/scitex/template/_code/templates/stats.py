#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/stats.py
# ----------------------------------------

"""Template for stx.stats statistical analysis module usage."""

TEMPLATE = {
    "name": "Statistics Module",
    "description": "stx.stats usage for publication-ready statistical analysis (23 tests)",
    "filename": "stats_script.py",
    "priority": 3,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.stats - Publication-Ready Statistics
========================================

stx.stats provides:
- 23 statistical tests with proper formatting
- Automatic normality checking
- Effect size calculations (Cohen's d, eta², etc.)
- APA/Nature-style result formatting
- Power analysis and sample size calculation
- Multiple comparison corrections

Usage Patterns
--------------
1. Direct API: stx.stats.test_*(data, return_as="dataframe")
2. CLI: scitex stats recommend --n-groups 3 --outcome continuous
3. MCP: stats_recommend_tests, stats_run_test, stats_format_results

Output Formats
--------------
- return_as="dataframe": Full results as pandas DataFrame
- return_as="dict": Results as dictionary
- return_as="latex": Ready-to-use LaTeX string
- return_as="apa": APA-style formatted string
"""

import numpy as np
import scitex as stx

# ============================================================
# Pattern 1: Two-Group Comparisons
# ============================================================

def two_group_comparison():
    """Compare two independent groups."""
    # Sample data
    group1 = np.random.normal(10, 2, 50)
    group2 = np.random.normal(12, 2, 50)

    # Independent t-test (with normality check)
    result = stx.stats.test_ttest_ind(
        group1, group2,
        return_as="dataframe"  # Options: dataframe, dict, latex, apa
    )
    print("Independent t-test:")
    print(result)
    # Returns: t, p, Cohen's d, CI, normality test, power

    # Mann-Whitney U (non-parametric alternative)
    result_mwu = stx.stats.test_mannwhitneyu(
        group1, group2,
        return_as="dataframe"
    )
    print("\\nMann-Whitney U:")
    print(result_mwu)

    # Paired t-test (for matched samples)
    pre = np.random.normal(10, 2, 30)
    post = pre + np.random.normal(2, 1, 30)  # Improvement

    result_paired = stx.stats.test_ttest_rel(
        pre, post,
        return_as="latex"  # Get LaTeX-formatted output
    )
    print("\\nPaired t-test (LaTeX):")
    print(result_paired)

# ============================================================
# Pattern 2: Multi-Group Comparisons
# ============================================================

def multi_group_comparison():
    """Compare three or more groups."""
    # Three groups
    group_a = np.random.normal(10, 2, 40)
    group_b = np.random.normal(12, 2, 40)
    group_c = np.random.normal(14, 2, 40)

    # One-way ANOVA
    result = stx.stats.test_anova(
        group_a, group_b, group_c,
        return_as="dataframe"
    )
    print("One-way ANOVA:")
    print(result)
    # Returns: F, p, eta², omega², power

    # Kruskal-Wallis (non-parametric)
    result_kw = stx.stats.test_kruskal(
        group_a, group_b, group_c,
        return_as="dataframe"
    )
    print("\\nKruskal-Wallis:")
    print(result_kw)

    # Post-hoc tests (after significant ANOVA)
    result_posthoc = stx.stats.test_tukey_hsd(
        group_a, group_b, group_c,
        return_as="dataframe"
    )
    print("\\nTukey HSD post-hoc:")
    print(result_posthoc)

# ============================================================
# Pattern 3: Correlation Analysis
# ============================================================

def correlation_analysis():
    """Analyze relationships between variables."""
    # Generate correlated data
    x = np.random.normal(0, 1, 100)
    y = 0.7 * x + np.random.normal(0, 0.5, 100)

    # Pearson correlation
    result = stx.stats.test_pearsonr(
        x, y,
        return_as="dataframe"
    )
    print("Pearson correlation:")
    print(result)
    # Returns: r, p, CI, R², power

    # Spearman correlation (non-parametric)
    result_spearman = stx.stats.test_spearmanr(
        x, y,
        return_as="dataframe"
    )
    print("\\nSpearman correlation:")
    print(result_spearman)

# ============================================================
# Pattern 4: Categorical Data Analysis
# ============================================================

def categorical_analysis():
    """Analyze categorical/count data."""
    # Chi-square test
    observed = np.array([[30, 20], [15, 35]])

    result = stx.stats.test_chi2(
        observed,
        return_as="dataframe"
    )
    print("Chi-square test:")
    print(result)
    # Returns: chi2, p, Cramer's V, df

    # Fisher's exact test (for small samples)
    small_table = np.array([[5, 2], [1, 6]])
    result_fisher = stx.stats.test_fisher_exact(
        small_table,
        return_as="dataframe"
    )
    print("\\nFisher's exact test:")
    print(result_fisher)

# ============================================================
# Pattern 5: With @stx.session (Recommended)
# ============================================================

@stx.session
def main(
    n_subjects=50,
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Complete statistical analysis workflow."""

    # Generate experimental data
    control = np.random.normal(100, 15, n_subjects)
    treatment = np.random.normal(115, 15, n_subjects)

    # 1. Run statistical test
    result = stx.stats.test_ttest_ind(
        control, treatment,
        return_as="dataframe"
    )
    logger.info(f"Statistical test completed")

    # 2. Save results
    stx.io.save(result, CONFIG.SDIR_OUT / "stats_results.csv")

    # 3. Create visualization with stats annotation
    fig, ax = plt.subplots()

    # Box plot
    ax.boxplot([control, treatment], labels=["Control", "Treatment"])
    ax.set_xyt("Group", "Score", "Treatment Effect")

    # Add significance annotation
    p_value = result["p_value"].values[0]
    stars = stx.stats.p_to_stars(p_value)
    ax.annotate(
        stars,
        xy=(1.5, max(control.max(), treatment.max()) * 1.05),
        ha="center", fontsize=14
    )

    stx.io.save(fig, CONFIG.SDIR_OUT / "treatment_effect.png")

    # 4. Export for paper
    latex_result = stx.stats.test_ttest_ind(
        control, treatment,
        return_as="latex"
    )
    stx.io.save(latex_result, CONFIG.SDIR_OUT / "stats_latex.txt")

    logger.info(f"Results saved to {{CONFIG.SDIR_OUT}}")
    return 0

# ============================================================
# Pattern 6: Power Analysis & Sample Size
# ============================================================

def power_analysis_example():
    """Calculate required sample size or achieved power."""
    # Calculate required sample size for t-test
    required_n = stx.stats.power_ttest(
        effect_size=0.5,    # Medium effect (Cohen's d)
        alpha=0.05,
        power=0.80,
        alternative="two-sided"
    )
    print(f"Required sample size per group: {{required_n}}")

    # Calculate achieved power
    achieved_power = stx.stats.power_ttest(
        effect_size=0.5,
        alpha=0.05,
        n=30,               # Given sample size
        alternative="two-sided"
    )
    print(f"Achieved power with n=30: {{achieved_power:.3f}}")

# ============================================================
# Available Statistical Tests
# ============================================================

"""
stx.stats Available Tests (23 total):
=====================================

Parametric (Continuous):
------------------------
- test_ttest_ind       : Independent samples t-test
- test_ttest_rel       : Paired samples t-test
- test_ttest_1samp     : One-sample t-test
- test_anova           : One-way ANOVA
- test_anova_rm        : Repeated measures ANOVA
- test_pearsonr        : Pearson correlation

Non-parametric:
---------------
- test_mannwhitneyu    : Mann-Whitney U test
- test_wilcoxon        : Wilcoxon signed-rank test
- test_kruskal         : Kruskal-Wallis H test
- test_friedman        : Friedman test
- test_spearmanr       : Spearman correlation

Categorical:
------------
- test_chi2            : Chi-square test
- test_fisher_exact    : Fisher's exact test
- test_mcnemar         : McNemar's test

Post-hoc:
---------
- test_tukey_hsd       : Tukey HSD
- test_dunn            : Dunn's test (after Kruskal)

Normality:
----------
- test_shapiro         : Shapiro-Wilk test
- test_normaltest      : D'Agostino-Pearson test

Helpers:
--------
- p_to_stars(p)        : Convert p-value to significance stars
- power_ttest(...)     : Power analysis for t-tests
- effect_size_cohens_d : Calculate Cohen's d
- correct_pvalues      : Multiple comparison correction (FDR, Bonferroni)

CLI Commands:
-------------
scitex stats recommend --n-groups 2    # Recommend appropriate test
scitex stats --help                    # Show all options

MCP Tools:
----------
stats_recommend_tests   # Get test recommendations
stats_run_test         # Execute a specific test
stats_format_results   # Format for publication
stats_power_analysis   # Power/sample size calculation
stats_correct_pvalues  # Multiple comparison correction
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
