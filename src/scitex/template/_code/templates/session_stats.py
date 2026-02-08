#!/usr/bin/env python3
"""Statistical analysis session script template."""

TEMPLATE = {
    "name": "Statistical Analysis Session",
    "description": "@stx.session script for statistical testing with publication-ready output",
    "filename": "stats_script.py",
    "usage": """
Usage:
  python stats_script.py
  python stats_script.py --help

Available Tests:
  stx.stats.test_ttest_ind()      # Independent t-test
  stx.stats.test_ttest_paired()   # Paired t-test
  stx.stats.test_anova()          # One-way ANOVA
  stx.stats.test_wilcoxon()       # Wilcoxon signed-rank
  stx.stats.test_mannwhitneyu()   # Mann-Whitney U
""",
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "{timestamp}"
# File: {filepath}

"""
{docstring}

Available Tests (23 total):
  stx.stats.test_ttest_ind()      # Independent t-test with effect size
  stx.stats.test_ttest_paired()   # Paired t-test
  stx.stats.test_anova()          # One-way ANOVA
  stx.stats.test_wilcoxon()       # Wilcoxon signed-rank (non-parametric)
  stx.stats.test_mannwhitneyu()   # Mann-Whitney U (non-parametric)

Output Formats:
  return_as="dataframe"   # pandas DataFrame
  return_as="dict"        # Python dictionary
  return_as="latex"       # LaTeX table
"""

import numpy as np
import scitex as stx


@stx.session
def main(
    n_samples=30,
    effect_size=0.5,
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Run statistical analysis with publication-ready output."""
    # Generate sample data
    rng = np.random.default_rng(42)
    group1 = rng.normal(10, 2, n_samples)
    group2 = rng.normal(10 + effect_size * 2, 2, n_samples)

    # Run t-test with effect size and CI
    result = stx.stats.test_ttest_ind(group1, group2, return_as="dataframe")
    logger.info(f"T-test result:\\n{{result}}")

    # Save results to CONFIG.SDIR_OUT
    stx.io.save(result, "stats_results.csv")
    logger.info(f"Results saved to: {{CONFIG.SDIR_OUT}}/stats_results.csv")

    return 0


if __name__ == "__main__":
    main()

# EOF
''',
}

# EOF
