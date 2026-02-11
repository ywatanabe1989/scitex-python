#!/usr/bin/env python3
# Timestamp: 2026-02-11
# File: src/scitex/stats/_mcp/_handlers/_run_test.py

"""Statistical test execution handler.

Delegates to scitex.stats.tests functions (single source of truth)
and converts their rich output to MCP format.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = ["run_test_handler"]

# =============================================================================
# Test Dispatch Registry - Maps MCP test names to (category, function_name)
# =============================================================================

_TEST_DISPATCH = {
    # Parametric tests
    "ttest_ind": ("parametric", "test_ttest_ind"),
    "ttest_paired": ("parametric", "test_ttest_rel"),
    "ttest_1samp": ("parametric", "test_ttest_1samp"),
    "anova": ("parametric", "test_anova"),
    "anova_rm": ("parametric", "test_anova_rm"),
    "anova_2way": ("parametric", "test_anova_2way"),
    # Nonparametric tests
    "brunner_munzel": ("nonparametric", "test_brunner_munzel"),
    "mannwhitneyu": ("nonparametric", "test_mannwhitneyu"),
    "wilcoxon": ("nonparametric", "test_wilcoxon"),
    "kruskal": ("nonparametric", "test_kruskal"),
    "friedman": ("nonparametric", "test_friedman"),
    # Correlation tests
    "pearson": ("correlation", "test_pearson"),
    "spearman": ("correlation", "test_spearman"),
    "kendall": ("correlation", "test_kendall"),
    "theilsen": ("correlation", "test_theilsen"),
    # Categorical tests
    "chi2": ("categorical", "test_chi2"),
    "fisher_exact": ("categorical", "test_fisher"),
    "mcnemar": ("categorical", "test_mcnemar"),
    "cochran_q": ("categorical", "test_cochran_q"),
    # Normality tests
    "shapiro": ("normality", "test_shapiro"),
    "normality": ("normality", "test_normality"),
    "ks_1samp": ("normality", "test_ks_1samp"),
    "ks_2samp": ("normality", "test_ks_2samp"),
}

# Tests that need special input handling
_TWO_GROUP = {
    "ttest_ind",
    "ttest_paired",
    "brunner_munzel",
    "mannwhitneyu",
    "wilcoxon",
    "pearson",
    "spearman",
    "kendall",
    "theilsen",
    "ks_2samp",
}
_ONE_SAMPLE = {"ttest_1samp", "shapiro", "normality", "ks_1samp"}
_MULTI_GROUP = {"anova", "kruskal"}
_CONTINGENCY = {"chi2", "fisher_exact", "mcnemar"}
_DATAFRAME_REQUIRED = {"anova_rm", "anova_2way", "friedman", "cochran_q"}


async def run_test_handler(  # noqa: C901
    test_name: str,
    data: list[list[float]] | None = None,
    data_file: str | None = None,
    columns: list[str] | None = None,
    alternative: str = "two-sided",
) -> dict:
    """Execute a statistical test by delegating to scitex.stats.tests.

    Parameters
    ----------
    test_name : str
        MCP test name (e.g., "ttest_ind", "pearson", "fisher_exact").
    data : list[list[float]], optional
        Data arrays. Interpretation depends on test type:
        - Two-group tests: [[group1...], [group2...]]
        - One-sample tests: [[sample...]]
        - Multi-group tests: [[g1...], [g2...], [g3...], ...]
        - Contingency: [[row1...], [row2...]] (2D table)
        - RM/factorial: [[condition1...], [condition2...], ...] (wide format)
    data_file : str, optional
        Path to CSV file. Use with ``columns`` instead of ``data``.
    columns : list[str], optional
        Column names to extract from ``data_file``.
    alternative : str
        Alternative hypothesis: "two-sided", "less", or "greater".
    """
    try:
        # Resolve data from CSV if needed
        if data_file and columns:
            import pandas as pd

            df = pd.read_csv(data_file)
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Columns not found: {missing}. Available: {list(df.columns)}"
                )
            data = [df[col].dropna().tolist() for col in columns]
        elif data is None:
            raise ValueError("Provide 'data' or 'data_file'+'columns'")

        from scitex.stats import tests

        loop = asyncio.get_event_loop()

        def do_test():
            if test_name not in _TEST_DISPATCH:
                raise ValueError(
                    f"Unknown test: {test_name}. "
                    f"Available: {', '.join(sorted(_TEST_DISPATCH.keys()))}"
                )

            category, func_name = _TEST_DISPATCH[test_name]
            test_func = getattr(getattr(tests, category), func_name)
            groups = [np.array(g, dtype=float) for g in data]

            # Call the test function with appropriate arguments
            if test_name in _TWO_GROUP:
                result = test_func(groups[0], groups[1], alternative=alternative)
            elif test_name in _ONE_SAMPLE:
                result = test_func(groups[0])
            elif test_name in _MULTI_GROUP:
                result = test_func(groups)
            elif test_name in _CONTINGENCY:
                table = np.array(data)
                if test_name == "fisher_exact":
                    result = test_func(table, alternative=alternative)
                else:
                    result = test_func(table)
            elif test_name in _DATAFRAME_REQUIRED:
                result = _call_dataframe_test(test_func, groups, test_name)
            else:
                result = test_func(*groups, alternative=alternative)

            return _convert_to_mcp_format(result)

        result = await loop.run_in_executor(None, do_test)

        return {
            "success": True,
            "test_name": test_name,
            "alternative": alternative,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def _call_dataframe_test(test_func, groups, test_name):
    """Construct DataFrame for RM/factorial tests from wide-format groups."""
    import pandas as pd

    # Convert list of conditions to long-format DataFrame
    records = []
    for subj_idx in range(len(groups[0])):
        for cond_idx, group in enumerate(groups):
            if subj_idx < len(group):
                records.append(
                    {
                        "subject": subj_idx,
                        "condition": f"cond_{cond_idx}",
                        "value": group[subj_idx],
                    }
                )
    df = pd.DataFrame(records)
    return test_func(
        df,
        subject_col="subject",
        condition_col="condition",
        value_col="value",
    )


def _convert_to_mcp_format(test_result: dict) -> dict:
    """Convert rich test function output to MCP format."""
    mcp_result = {
        "test": test_result.get("test_method", ""),
        "statistic": test_result.get("statistic", 0.0),
        "statistic_name": test_result.get("statistic_name", ""),
        "p_value": test_result.get("pvalue", 1.0),
        "stars": test_result.get("stars", "ns"),
        "significant": test_result.get("significant", False),
        "alpha": test_result.get("alpha", 0.05),
    }

    # Optional fields
    for key in (
        "effect_size",
        "effect_size_metric",
        "effect_size_interpretation",
        "power",
        "H0",
        "r_squared",
        "ci_lower",
        "ci_upper",
        "expected_frequencies",
    ):
        if key in test_result:
            mcp_result[key] = test_result[key]

    # Pass through df_* and n_* fields
    for key, val in test_result.items():
        if key.startswith(("df", "n_")):
            mcp_result[key] = val

    return mcp_result


# EOF
