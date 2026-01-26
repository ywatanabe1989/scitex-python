#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/stats/_mcp/_handlers/_recommend.py

"""Test recommendation handler."""

from __future__ import annotations

import asyncio
from datetime import datetime

__all__ = ["recommend_tests_handler"]


def _get_test_rationale(test_name: str) -> str:
    """Get rationale for recommending a specific test."""
    rationales = {
        "brunner_munzel": "Robust nonparametric test - no normality/equal variance assumptions",
        "ttest_ind": "Classic parametric test for comparing two independent groups",
        "ttest_paired": "Parametric test for paired/matched samples",
        "ttest_1samp": "One-sample t-test for comparing to a population mean",
        "mannwhitneyu": "Nonparametric alternative to independent t-test",
        "wilcoxon": "Nonparametric alternative to paired t-test",
        "anova": "Parametric test for comparing 3+ groups",
        "kruskal": "Nonparametric alternative to one-way ANOVA",
        "chi2": "Test for independence in contingency tables",
        "fisher_exact": "Exact test for small sample contingency tables",
        "pearson": "Parametric correlation coefficient",
        "spearman": "Nonparametric rank correlation",
        "kendall": "Robust nonparametric correlation for ordinal data",
    }
    return rationales.get(test_name, "Applicable to the given context")


async def recommend_tests_handler(
    n_groups: int = 2,
    sample_sizes: list[int] | None = None,
    outcome_type: str = "continuous",
    design: str = "between",
    paired: bool = False,
    has_control_group: bool = False,
    top_k: int = 3,
) -> dict:
    """Recommend appropriate statistical tests based on data characteristics."""
    try:
        from scitex.stats.auto import StatContext, recommend_tests

        loop = asyncio.get_event_loop()

        def do_recommend():
            ctx = StatContext(
                n_groups=n_groups,
                sample_sizes=sample_sizes or [30] * n_groups,
                outcome_type=outcome_type,
                design=design,
                paired=paired,
                has_control_group=has_control_group,
                n_factors=1,
            )
            tests = recommend_tests(ctx, top_k=top_k)

            # Get details about each recommended test
            from scitex.stats.auto._rules import TEST_RULES

            recommendations = []
            for test_name in tests:
                rule = TEST_RULES.get(test_name)
                if rule:
                    recommendations.append(
                        {
                            "name": test_name,
                            "family": rule.family,
                            "priority": rule.priority,
                            "needs_normality": rule.needs_normality,
                            "needs_equal_variance": rule.needs_equal_variance,
                            "rationale": _get_test_rationale(test_name),
                        }
                    )

            return recommendations

        recommendations = await loop.run_in_executor(None, do_recommend)

        return {
            "success": True,
            "context": {
                "n_groups": n_groups,
                "sample_sizes": sample_sizes,
                "outcome_type": outcome_type,
                "design": design,
                "paired": paired,
                "has_control_group": has_control_group,
            },
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
