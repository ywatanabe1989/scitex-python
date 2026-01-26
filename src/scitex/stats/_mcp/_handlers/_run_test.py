#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_run_test.py

"""Statistical test execution handler."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = ["run_test_handler"]


async def run_test_handler(
    test_name: str,
    data: list[list[float]],
    alternative: str = "two-sided",
) -> dict:
    """Execute a statistical test on provided data."""
    try:
        from scipy import stats as scipy_stats

        loop = asyncio.get_event_loop()

        def do_test():
            # Convert data to numpy arrays
            groups = [np.array(g, dtype=float) for g in data]

            result = {}

            # Run the appropriate test
            if test_name == "ttest_ind":
                result = _run_ttest_ind(groups, alternative, scipy_stats)
            elif test_name == "ttest_paired":
                result = _run_ttest_paired(groups, alternative, scipy_stats)
            elif test_name == "ttest_1samp":
                result = _run_ttest_1samp(groups, alternative, scipy_stats)
            elif test_name == "brunner_munzel":
                result = _run_brunner_munzel(groups, alternative, scipy_stats)
            elif test_name == "mannwhitneyu":
                result = _run_mannwhitneyu(groups, alternative, scipy_stats)
            elif test_name == "wilcoxon":
                result = _run_wilcoxon(groups, alternative, scipy_stats)
            elif test_name == "anova":
                result = _run_anova(groups, scipy_stats)
            elif test_name == "kruskal":
                result = _run_kruskal(groups, scipy_stats)
            elif test_name == "chi2":
                result = _run_chi2(data, scipy_stats)
            elif test_name == "fisher_exact":
                result = _run_fisher_exact(data, alternative, scipy_stats)
            elif test_name == "pearson":
                result = _run_pearson(groups, scipy_stats)
            elif test_name == "spearman":
                result = _run_spearman(groups, scipy_stats)
            elif test_name == "kendall":
                result = _run_kendall(groups, scipy_stats)
            else:
                raise ValueError(f"Unknown test: {test_name}")

            # Calculate effect size if applicable
            if test_name in [
                "ttest_ind",
                "ttest_paired",
                "brunner_munzel",
                "mannwhitneyu",
            ]:
                result = _add_effect_size(result, groups)

            # Add significance determination
            alpha = 0.05
            result["significant"] = result["p_value"] < alpha
            result["alpha"] = alpha

            return result

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


def _run_ttest_ind(groups, alternative, scipy_stats):
    if len(groups) != 2:
        raise ValueError("t-test requires exactly 2 groups")
    stat, p_value = scipy_stats.ttest_ind(groups[0], groups[1], alternative=alternative)
    df = len(groups[0]) + len(groups[1]) - 2
    return {
        "test": "Independent t-test",
        "statistic": float(stat),
        "statistic_name": "t",
        "p_value": float(p_value),
        "df": df,
    }


def _run_ttest_paired(groups, alternative, scipy_stats):
    if len(groups) != 2:
        raise ValueError("Paired t-test requires exactly 2 groups")
    stat, p_value = scipy_stats.ttest_rel(groups[0], groups[1], alternative=alternative)
    df = len(groups[0]) - 1
    return {
        "test": "Paired t-test",
        "statistic": float(stat),
        "statistic_name": "t",
        "p_value": float(p_value),
        "df": df,
    }


def _run_ttest_1samp(groups, alternative, scipy_stats):
    if len(groups) != 1:
        raise ValueError("One-sample t-test requires exactly 1 group")
    stat, p_value = scipy_stats.ttest_1samp(groups[0], 0, alternative=alternative)
    df = len(groups[0]) - 1
    return {
        "test": "One-sample t-test",
        "statistic": float(stat),
        "statistic_name": "t",
        "p_value": float(p_value),
        "df": df,
    }


def _run_brunner_munzel(groups, alternative, scipy_stats):
    if len(groups) != 2:
        raise ValueError("Brunner-Munzel requires exactly 2 groups")
    res = scipy_stats.brunnermunzel(groups[0], groups[1], alternative=alternative)
    return {
        "test": "Brunner-Munzel test",
        "statistic": float(res.statistic),
        "statistic_name": "BM",
        "p_value": float(res.pvalue),
    }


def _run_mannwhitneyu(groups, alternative, scipy_stats):
    if len(groups) != 2:
        raise ValueError("Mann-Whitney U requires exactly 2 groups")
    stat, p_value = scipy_stats.mannwhitneyu(
        groups[0], groups[1], alternative=alternative
    )
    return {
        "test": "Mann-Whitney U test",
        "statistic": float(stat),
        "statistic_name": "U",
        "p_value": float(p_value),
    }


def _run_wilcoxon(groups, alternative, scipy_stats):
    if len(groups) != 2:
        raise ValueError("Wilcoxon requires exactly 2 paired groups")
    stat, p_value = scipy_stats.wilcoxon(groups[0], groups[1], alternative=alternative)
    return {
        "test": "Wilcoxon signed-rank test",
        "statistic": float(stat),
        "statistic_name": "W",
        "p_value": float(p_value),
    }


def _run_anova(groups, scipy_stats):
    if len(groups) < 2:
        raise ValueError("ANOVA requires at least 2 groups")
    stat, p_value = scipy_stats.f_oneway(*groups)
    df_between = len(groups) - 1
    df_within = sum(len(g) for g in groups) - len(groups)
    return {
        "test": "One-way ANOVA",
        "statistic": float(stat),
        "statistic_name": "F",
        "p_value": float(p_value),
        "df_between": df_between,
        "df_within": df_within,
    }


def _run_kruskal(groups, scipy_stats):
    if len(groups) < 2:
        raise ValueError("Kruskal-Wallis requires at least 2 groups")
    stat, p_value = scipy_stats.kruskal(*groups)
    return {
        "test": "Kruskal-Wallis H test",
        "statistic": float(stat),
        "statistic_name": "H",
        "p_value": float(p_value),
        "df": len(groups) - 1,
    }


def _run_chi2(data, scipy_stats):
    table = np.array(data)
    chi2, p_value, dof, expected = scipy_stats.chi2_contingency(table)
    return {
        "test": "Chi-square test of independence",
        "statistic": float(chi2),
        "statistic_name": "chi2",
        "p_value": float(p_value),
        "df": int(dof),
        "expected_frequencies": expected.tolist(),
    }


def _run_fisher_exact(data, alternative, scipy_stats):
    table = np.array(data)
    if table.shape != (2, 2):
        raise ValueError("Fisher's exact test requires a 2x2 table")
    odds_ratio, p_value = scipy_stats.fisher_exact(table, alternative=alternative)
    return {
        "test": "Fisher's exact test",
        "statistic": float(odds_ratio),
        "statistic_name": "odds_ratio",
        "p_value": float(p_value),
    }


def _run_pearson(groups, scipy_stats):
    if len(groups) != 2:
        raise ValueError("Pearson correlation requires exactly 2 variables")
    r, p_value = scipy_stats.pearsonr(groups[0], groups[1])
    return {
        "test": "Pearson correlation",
        "statistic": float(r),
        "statistic_name": "r",
        "p_value": float(p_value),
    }


def _run_spearman(groups, scipy_stats):
    if len(groups) != 2:
        raise ValueError("Spearman correlation requires exactly 2 variables")
    r, p_value = scipy_stats.spearmanr(groups[0], groups[1])
    return {
        "test": "Spearman correlation",
        "statistic": float(r),
        "statistic_name": "rho",
        "p_value": float(p_value),
    }


def _run_kendall(groups, scipy_stats):
    if len(groups) != 2:
        raise ValueError("Kendall correlation requires exactly 2 variables")
    tau, p_value = scipy_stats.kendalltau(groups[0], groups[1])
    return {
        "test": "Kendall tau correlation",
        "statistic": float(tau),
        "statistic_name": "tau",
        "p_value": float(p_value),
    }


def _add_effect_size(result, groups):
    """Add effect size calculations to result."""
    from scitex.stats.effect_sizes import cliffs_delta, cohens_d

    if len(groups) == 2:
        d = cohens_d(groups[0], groups[1])
        delta = cliffs_delta(groups[0], groups[1])
        result["effect_size"] = {
            "cohens_d": float(d),
            "cliffs_delta": float(delta),
        }
    return result


# EOF
