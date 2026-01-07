#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/stats/_mcp_handlers.py
# ----------------------------------------

"""Handler implementations for the scitex-stats MCP server."""

from __future__ import annotations

import asyncio
from datetime import datetime

import numpy as np

__all__ = [
    "recommend_tests_handler",
    "run_test_handler",
    "format_results_handler",
    "power_analysis_handler",
    "correct_pvalues_handler",
    "describe_handler",
    "effect_size_handler",
    "normality_test_handler",
    "posthoc_test_handler",
    "p_to_stars_handler",
]


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
                if len(groups) != 2:
                    raise ValueError("t-test requires exactly 2 groups")
                stat, p_value = scipy_stats.ttest_ind(
                    groups[0], groups[1], alternative=alternative
                )
                df = len(groups[0]) + len(groups[1]) - 2
                result = {
                    "test": "Independent t-test",
                    "statistic": float(stat),
                    "statistic_name": "t",
                    "p_value": float(p_value),
                    "df": df,
                }

            elif test_name == "ttest_paired":
                if len(groups) != 2:
                    raise ValueError("Paired t-test requires exactly 2 groups")
                stat, p_value = scipy_stats.ttest_rel(
                    groups[0], groups[1], alternative=alternative
                )
                df = len(groups[0]) - 1
                result = {
                    "test": "Paired t-test",
                    "statistic": float(stat),
                    "statistic_name": "t",
                    "p_value": float(p_value),
                    "df": df,
                }

            elif test_name == "ttest_1samp":
                if len(groups) != 1:
                    raise ValueError("One-sample t-test requires exactly 1 group")
                stat, p_value = scipy_stats.ttest_1samp(
                    groups[0], 0, alternative=alternative
                )
                df = len(groups[0]) - 1
                result = {
                    "test": "One-sample t-test",
                    "statistic": float(stat),
                    "statistic_name": "t",
                    "p_value": float(p_value),
                    "df": df,
                }

            elif test_name == "brunner_munzel":
                if len(groups) != 2:
                    raise ValueError("Brunner-Munzel requires exactly 2 groups")
                res = scipy_stats.brunnermunzel(
                    groups[0], groups[1], alternative=alternative
                )
                result = {
                    "test": "Brunner-Munzel test",
                    "statistic": float(res.statistic),
                    "statistic_name": "BM",
                    "p_value": float(res.pvalue),
                }

            elif test_name == "mannwhitneyu":
                if len(groups) != 2:
                    raise ValueError("Mann-Whitney U requires exactly 2 groups")
                stat, p_value = scipy_stats.mannwhitneyu(
                    groups[0], groups[1], alternative=alternative
                )
                result = {
                    "test": "Mann-Whitney U test",
                    "statistic": float(stat),
                    "statistic_name": "U",
                    "p_value": float(p_value),
                }

            elif test_name == "wilcoxon":
                if len(groups) != 2:
                    raise ValueError("Wilcoxon requires exactly 2 paired groups")
                stat, p_value = scipy_stats.wilcoxon(
                    groups[0], groups[1], alternative=alternative
                )
                result = {
                    "test": "Wilcoxon signed-rank test",
                    "statistic": float(stat),
                    "statistic_name": "W",
                    "p_value": float(p_value),
                }

            elif test_name == "anova":
                if len(groups) < 2:
                    raise ValueError("ANOVA requires at least 2 groups")
                stat, p_value = scipy_stats.f_oneway(*groups)
                df_between = len(groups) - 1
                df_within = sum(len(g) for g in groups) - len(groups)
                result = {
                    "test": "One-way ANOVA",
                    "statistic": float(stat),
                    "statistic_name": "F",
                    "p_value": float(p_value),
                    "df_between": df_between,
                    "df_within": df_within,
                }

            elif test_name == "kruskal":
                if len(groups) < 2:
                    raise ValueError("Kruskal-Wallis requires at least 2 groups")
                stat, p_value = scipy_stats.kruskal(*groups)
                result = {
                    "test": "Kruskal-Wallis H test",
                    "statistic": float(stat),
                    "statistic_name": "H",
                    "p_value": float(p_value),
                    "df": len(groups) - 1,
                }

            elif test_name == "chi2":
                # Expects contingency table as data
                table = np.array(data)
                chi2, p_value, dof, expected = scipy_stats.chi2_contingency(table)
                result = {
                    "test": "Chi-square test of independence",
                    "statistic": float(chi2),
                    "statistic_name": "chi2",
                    "p_value": float(p_value),
                    "df": int(dof),
                    "expected_frequencies": expected.tolist(),
                }

            elif test_name == "fisher_exact":
                # Expects 2x2 contingency table
                table = np.array(data)
                if table.shape != (2, 2):
                    raise ValueError("Fisher's exact test requires a 2x2 table")
                odds_ratio, p_value = scipy_stats.fisher_exact(
                    table, alternative=alternative
                )
                result = {
                    "test": "Fisher's exact test",
                    "statistic": float(odds_ratio),
                    "statistic_name": "odds_ratio",
                    "p_value": float(p_value),
                }

            elif test_name == "pearson":
                if len(groups) != 2:
                    raise ValueError("Pearson correlation requires exactly 2 variables")
                r, p_value = scipy_stats.pearsonr(groups[0], groups[1])
                result = {
                    "test": "Pearson correlation",
                    "statistic": float(r),
                    "statistic_name": "r",
                    "p_value": float(p_value),
                }

            elif test_name == "spearman":
                if len(groups) != 2:
                    raise ValueError(
                        "Spearman correlation requires exactly 2 variables"
                    )
                r, p_value = scipy_stats.spearmanr(groups[0], groups[1])
                result = {
                    "test": "Spearman correlation",
                    "statistic": float(r),
                    "statistic_name": "rho",
                    "p_value": float(p_value),
                }

            elif test_name == "kendall":
                if len(groups) != 2:
                    raise ValueError("Kendall correlation requires exactly 2 variables")
                tau, p_value = scipy_stats.kendalltau(groups[0], groups[1])
                result = {
                    "test": "Kendall tau correlation",
                    "statistic": float(tau),
                    "statistic_name": "tau",
                    "p_value": float(p_value),
                }

            else:
                raise ValueError(f"Unknown test: {test_name}")

            # Calculate effect size if applicable
            if test_name in [
                "ttest_ind",
                "ttest_paired",
                "brunner_munzel",
                "mannwhitneyu",
            ]:
                from scitex.stats.effect_sizes import cliffs_delta, cohens_d

                if len(groups) == 2:
                    d = cohens_d(groups[0], groups[1])
                    delta = cliffs_delta(groups[0], groups[1])
                    result["effect_size"] = {
                        "cohens_d": float(d),
                        "cliffs_delta": float(delta),
                    }

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


async def format_results_handler(
    test_name: str,
    statistic: float,
    p_value: float,
    df: float | None = None,
    effect_size: float | None = None,
    effect_size_name: str | None = None,
    style: str = "apa",
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> dict:
    """Format statistical results in journal style."""
    try:
        loop = asyncio.get_event_loop()

        def do_format():
            from scitex.stats.auto import format_test_line, p_to_stars
            from scitex.stats.auto._formatting import EffectResultDict, TestResultDict

            # Build test result dict
            test_result: TestResultDict = {
                "test_name": test_name,
                "stat": statistic,
                "p_raw": p_value,
            }
            if df is not None:
                test_result["df"] = df

            # Build effect result if provided
            effects = None
            if effect_size is not None:
                effects = [
                    EffectResultDict(
                        name=effect_size_name or "d",
                        label=effect_size_name or "Cohen's d",
                        value=effect_size,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                    )
                ]

            # Map style names
            style_map = {
                "apa": "apa_latex",
                "nature": "nature",
                "science": "science",
                "brief": "brief",
            }
            style_id = style_map.get(style, "apa_latex")

            # Format the line
            formatted = format_test_line(
                test_result,
                effects=effects,
                style=style_id,
                include_n=False,
            )

            # Get stars representation
            stars = p_to_stars(p_value)

            return {
                "formatted": formatted,
                "stars": stars,
            }

        result = await loop.run_in_executor(None, do_format)

        return {
            "success": True,
            "style": style,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def power_analysis_handler(
    test_type: str = "ttest",
    effect_size: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    n: int | None = None,
    n_groups: int = 2,
    ratio: float = 1.0,
) -> dict:
    """Calculate statistical power or required sample size."""
    try:
        loop = asyncio.get_event_loop()

        def do_power():
            from scitex.stats.power._power import power_ttest, sample_size_ttest

            result = {}

            if test_type == "ttest":
                if n is not None and effect_size is not None:
                    # Calculate power given n and effect size
                    calculated_power = power_ttest(
                        effect_size=effect_size,
                        n1=n,
                        n2=int(n * ratio),
                        alpha=alpha,
                        test_type="two-sample",
                    )
                    result = {
                        "mode": "power_calculation",
                        "power": calculated_power,
                        "n1": n,
                        "n2": int(n * ratio),
                        "effect_size": effect_size,
                        "alpha": alpha,
                    }
                elif effect_size is not None:
                    # Calculate required sample size
                    n1, n2 = sample_size_ttest(
                        effect_size=effect_size,
                        power=power,
                        alpha=alpha,
                        ratio=ratio,
                    )
                    result = {
                        "mode": "sample_size_calculation",
                        "required_n1": n1,
                        "required_n2": n2,
                        "total_n": n1 + n2,
                        "effect_size": effect_size,
                        "target_power": power,
                        "alpha": alpha,
                    }
                else:
                    raise ValueError("Either n or effect_size must be provided")

            elif test_type == "anova":
                # Simplified ANOVA power (using f = d * sqrt(k-1) / sqrt(2k))
                if effect_size is None:
                    raise ValueError("effect_size required for ANOVA power")

                # Convert Cohen's f to d for approximation
                # This is a simplified calculation
                from scipy import stats as scipy_stats

                if n is not None:
                    df1 = n_groups - 1
                    df2 = n_groups * n - n_groups
                    nc = effect_size**2 * n * n_groups
                    f_crit = scipy_stats.f.ppf(1 - alpha, df1, df2)
                    power_val = 1 - scipy_stats.ncf.cdf(f_crit, df1, df2, nc)
                    result = {
                        "mode": "power_calculation",
                        "power": power_val,
                        "n_per_group": n,
                        "n_groups": n_groups,
                        "effect_size_f": effect_size,
                        "alpha": alpha,
                    }
                else:
                    # Binary search for n
                    n_min, n_max = 2, 1000
                    while n_max - n_min > 1:
                        n_mid = (n_min + n_max) // 2
                        df1 = n_groups - 1
                        df2 = n_groups * n_mid - n_groups
                        nc = effect_size**2 * n_mid * n_groups
                        f_crit = scipy_stats.f.ppf(1 - alpha, df1, df2)
                        power_val = 1 - scipy_stats.ncf.cdf(f_crit, df1, df2, nc)
                        if power_val < power:
                            n_min = n_mid
                        else:
                            n_max = n_mid

                    result = {
                        "mode": "sample_size_calculation",
                        "required_n_per_group": n_max,
                        "total_n": n_max * n_groups,
                        "n_groups": n_groups,
                        "effect_size_f": effect_size,
                        "target_power": power,
                        "alpha": alpha,
                    }

            elif test_type == "correlation":
                # Power for correlation coefficient
                from scipy import stats as scipy_stats

                if effect_size is None:
                    raise ValueError("effect_size (r) required for correlation power")

                if n is not None:
                    # Calculate power
                    z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
                    se = 1 / np.sqrt(n - 3)
                    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
                    power_val = (
                        1
                        - scipy_stats.norm.cdf(z_crit - z / se)
                        + scipy_stats.norm.cdf(-z_crit - z / se)
                    )
                    result = {
                        "mode": "power_calculation",
                        "power": power_val,
                        "n": n,
                        "effect_size_r": effect_size,
                        "alpha": alpha,
                    }
                else:
                    # Calculate required n (binary search)
                    z = 0.5 * np.log((1 + effect_size) / (1 - effect_size))
                    z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
                    z_power = scipy_stats.norm.ppf(power)
                    required_n = int(np.ceil(((z_crit + z_power) / z) ** 2 + 3))
                    result = {
                        "mode": "sample_size_calculation",
                        "required_n": required_n,
                        "effect_size_r": effect_size,
                        "target_power": power,
                        "alpha": alpha,
                    }

            elif test_type == "chi2":
                # Chi-square power (simplified)
                from scipy import stats as scipy_stats

                if effect_size is None:
                    raise ValueError("effect_size (w) required for chi2 power")

                df = n_groups - 1  # Simplified: using n_groups as number of cells

                if n is not None:
                    nc = effect_size**2 * n
                    chi2_crit = scipy_stats.chi2.ppf(1 - alpha, df)
                    power_val = 1 - scipy_stats.ncx2.cdf(chi2_crit, df, nc)
                    result = {
                        "mode": "power_calculation",
                        "power": power_val,
                        "n": n,
                        "df": df,
                        "effect_size_w": effect_size,
                        "alpha": alpha,
                    }
                else:
                    # Binary search for n
                    n_min, n_max = 10, 10000
                    while n_max - n_min > 1:
                        n_mid = (n_min + n_max) // 2
                        nc = effect_size**2 * n_mid
                        chi2_crit = scipy_stats.chi2.ppf(1 - alpha, df)
                        power_val = 1 - scipy_stats.ncx2.cdf(chi2_crit, df, nc)
                        if power_val < power:
                            n_min = n_mid
                        else:
                            n_max = n_mid

                    result = {
                        "mode": "sample_size_calculation",
                        "required_n": n_max,
                        "df": df,
                        "effect_size_w": effect_size,
                        "target_power": power,
                        "alpha": alpha,
                    }

            else:
                raise ValueError(f"Unknown test_type: {test_type}")

            return result

        result = await loop.run_in_executor(None, do_power)

        return {
            "success": True,
            "test_type": test_type,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def correct_pvalues_handler(
    pvalues: list[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> dict:
    """Apply multiple comparison correction to p-values."""
    try:
        loop = asyncio.get_event_loop()

        def do_correct():
            from statsmodels.stats.multitest import multipletests

            # Map method names
            method_map = {
                "bonferroni": "bonferroni",
                "fdr_bh": "fdr_bh",
                "fdr_by": "fdr_by",
                "holm": "holm",
                "sidak": "sidak",
            }
            sm_method = method_map.get(method, "fdr_bh")

            pvals = np.array(pvalues)
            reject, pvals_corrected, _, _ = multipletests(
                pvals, alpha=alpha, method=sm_method
            )

            return {
                "original_pvalues": pvalues,
                "corrected_pvalues": pvals_corrected.tolist(),
                "reject_null": reject.tolist(),
                "n_significant": int(reject.sum()),
                "n_tests": len(pvalues),
            }

        result = await loop.run_in_executor(None, do_correct)

        return {
            "success": True,
            "method": method,
            "alpha": alpha,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except ImportError:
        # Fallback implementation without statsmodels
        try:
            n = len(pvalues)
            pvals = np.array(pvalues)

            if method == "bonferroni":
                corrected = np.minimum(pvals * n, 1.0)
            elif method == "holm":
                sorted_idx = np.argsort(pvals)
                corrected = np.empty(n)
                cummax = 0.0
                for rank, idx in enumerate(sorted_idx, start=1):
                    adj = min((n - rank + 1) * pvals[idx], 1.0)
                    adj = max(adj, cummax)
                    corrected[idx] = adj
                    cummax = adj
            elif method == "fdr_bh":
                sorted_idx = np.argsort(pvals)
                corrected = np.empty(n)
                prev = 1.0
                for rank in range(n, 0, -1):
                    idx = sorted_idx[rank - 1]
                    bh = pvals[idx] * n / rank
                    val = min(bh, prev, 1.0)
                    corrected[idx] = val
                    prev = val
            elif method == "sidak":
                corrected = 1 - (1 - pvals) ** n
            else:
                corrected = pvals

            return {
                "success": True,
                "method": method,
                "alpha": alpha,
                "original_pvalues": pvalues,
                "corrected_pvalues": corrected.tolist(),
                "reject_null": (corrected < alpha).tolist(),
                "n_significant": int((corrected < alpha).sum()),
                "n_tests": n,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    except Exception as e:
        return {"success": False, "error": str(e)}


async def describe_handler(
    data: list[float],
    percentiles: list[float] | None = None,
) -> dict:
    """Calculate descriptive statistics for data."""
    try:
        loop = asyncio.get_event_loop()

        def do_describe():
            arr = np.array(data, dtype=float)
            arr = arr[~np.isnan(arr)]  # Remove NaN

            if len(arr) == 0:
                return {"error": "No valid data points"}

            percs = percentiles or [25, 50, 75]
            percentile_values = np.percentile(arr, percs)

            result = {
                "n": int(len(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "var": float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "sem": (
                    float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
                    if len(arr) > 1
                    else 0.0
                ),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "median": float(np.median(arr)),
                "percentiles": {
                    str(int(p)): float(v) for p, v in zip(percs, percentile_values)
                },
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            }

            # Add skewness and kurtosis if scipy available
            try:
                from scipy import stats as scipy_stats

                result["skewness"] = float(scipy_stats.skew(arr))
                result["kurtosis"] = float(scipy_stats.kurtosis(arr))
            except ImportError:
                pass

            return result

        result = await loop.run_in_executor(None, do_describe)

        return {
            "success": True,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def effect_size_handler(
    group1: list[float],
    group2: list[float],
    measure: str = "cohens_d",
    pooled: bool = True,
) -> dict:
    """Calculate effect size between groups."""
    try:
        from scitex.stats.effect_sizes import (
            cliffs_delta,
            cohens_d,
            interpret_cliffs_delta,
            interpret_cohens_d,
        )

        loop = asyncio.get_event_loop()

        def do_effect_size():
            g1 = np.array(group1, dtype=float)
            g2 = np.array(group2, dtype=float)

            result = {}

            if measure == "cohens_d":
                d = cohens_d(g1, g2)
                result = {
                    "measure": "Cohen's d",
                    "value": float(d),
                    "interpretation": interpret_cohens_d(d),
                }

            elif measure == "hedges_g":
                # Hedges' g is Cohen's d with bias correction
                d = cohens_d(g1, g2)
                n1, n2 = len(g1), len(g2)
                correction = 1 - (3 / (4 * (n1 + n2) - 9))
                g = d * correction
                result = {
                    "measure": "Hedges' g",
                    "value": float(g),
                    "interpretation": interpret_cohens_d(g),  # Same thresholds
                }

            elif measure == "glass_delta":
                # Glass's delta uses only control group std
                mean_diff = np.mean(g1) - np.mean(g2)
                delta = mean_diff / np.std(g2, ddof=1)
                result = {
                    "measure": "Glass's delta",
                    "value": float(delta),
                    "interpretation": interpret_cohens_d(delta),
                }

            elif measure == "cliffs_delta":
                delta = cliffs_delta(g1, g2)
                result = {
                    "measure": "Cliff's delta",
                    "value": float(delta),
                    "interpretation": interpret_cliffs_delta(delta),
                }

            else:
                raise ValueError(f"Unknown measure: {measure}")

            # Add confidence interval approximation for Cohen's d
            if measure in ["cohens_d", "hedges_g", "glass_delta"]:
                n1, n2 = len(g1), len(g2)
                se = np.sqrt(
                    (n1 + n2) / (n1 * n2) + result["value"] ** 2 / (2 * (n1 + n2))
                )
                result["ci_lower"] = float(result["value"] - 1.96 * se)
                result["ci_upper"] = float(result["value"] + 1.96 * se)

            return result

        result = await loop.run_in_executor(None, do_effect_size)

        return {
            "success": True,
            "group1_n": len(group1),
            "group2_n": len(group2),
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def normality_test_handler(
    data: list[float],
    method: str = "shapiro",
) -> dict:
    """Test whether data follows a normal distribution."""
    try:
        from scipy import stats as scipy_stats

        loop = asyncio.get_event_loop()

        def do_normality():
            arr = np.array(data, dtype=float)
            arr = arr[~np.isnan(arr)]

            if len(arr) < 3:
                return {"error": "Need at least 3 data points"}

            result = {}

            if method == "shapiro":
                stat, p_value = scipy_stats.shapiro(arr)
                result = {
                    "test": "Shapiro-Wilk",
                    "statistic": float(stat),
                    "statistic_name": "W",
                    "p_value": float(p_value),
                }

            elif method == "dagostino":
                if len(arr) < 8:
                    return {"error": "D'Agostino test requires at least 8 samples"}
                stat, p_value = scipy_stats.normaltest(arr)
                result = {
                    "test": "D'Agostino-Pearson",
                    "statistic": float(stat),
                    "statistic_name": "K2",
                    "p_value": float(p_value),
                }

            elif method == "anderson":
                res = scipy_stats.anderson(arr, dist="norm")
                # Use 5% significance level
                idx = 2  # Index for 5% level
                result = {
                    "test": "Anderson-Darling",
                    "statistic": float(res.statistic),
                    "statistic_name": "A2",
                    "critical_value_5pct": float(res.critical_values[idx]),
                    "normal": bool(res.statistic < res.critical_values[idx]),
                }

            elif method == "lilliefors":
                try:
                    from statsmodels.stats.diagnostic import lilliefors

                    stat, p_value = lilliefors(arr, dist="norm")
                    result = {
                        "test": "Lilliefors",
                        "statistic": float(stat),
                        "statistic_name": "D",
                        "p_value": float(p_value),
                    }
                except ImportError:
                    return {"error": "statsmodels required for Lilliefors test"}

            else:
                raise ValueError(f"Unknown method: {method}")

            # Add interpretation
            if "p_value" in result:
                result["is_normal"] = result["p_value"] >= 0.05
                result["interpretation"] = (
                    "Data appears normally distributed (p >= 0.05)"
                    if result["is_normal"]
                    else "Data deviates from normal distribution (p < 0.05)"
                )

            return result

        result = await loop.run_in_executor(None, do_normality)

        return {
            "success": True,
            "method": method,
            "n": len(data),
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def posthoc_test_handler(
    groups: list[list[float]],
    group_names: list[str] | None = None,
    method: str = "tukey",
    control_group: int = 0,
) -> dict:
    """Run post-hoc pairwise comparisons."""
    try:
        loop = asyncio.get_event_loop()

        def do_posthoc():
            group_arrays = [np.array(g, dtype=float) for g in groups]
            names = group_names or [f"Group_{i + 1}" for i in range(len(groups))]

            comparisons = []

            if method == "tukey":
                from scipy import stats as scipy_stats

                # All pairwise comparisons with Tukey HSD approximation
                all_data = np.concatenate(group_arrays)
                group_labels = np.concatenate(
                    [[names[i]] * len(g) for i, g in enumerate(group_arrays)]
                )

                # Use statsmodels if available, otherwise manual calculation
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd

                    tukey = pairwise_tukeyhsd(all_data, group_labels)

                    for i in range(len(tukey.summary().data) - 1):
                        row = tukey.summary().data[i + 1]
                        comparisons.append(
                            {
                                "group1": str(row[0]),
                                "group2": str(row[1]),
                                "mean_diff": float(row[2]),
                                "p_adj": float(row[3]),
                                "ci_lower": float(row[4]),
                                "ci_upper": float(row[5]),
                                "reject": bool(row[6]),
                            }
                        )
                except ImportError:
                    # Fallback: Bonferroni-corrected t-tests
                    n_comparisons = len(groups) * (len(groups) - 1) // 2
                    for i in range(len(groups)):
                        for j in range(i + 1, len(groups)):
                            stat, p = scipy_stats.ttest_ind(
                                group_arrays[i], group_arrays[j]
                            )
                            p_adj = min(p * n_comparisons, 1.0)
                            comparisons.append(
                                {
                                    "group1": names[i],
                                    "group2": names[j],
                                    "mean_diff": float(
                                        np.mean(group_arrays[i])
                                        - np.mean(group_arrays[j])
                                    ),
                                    "t_statistic": float(stat),
                                    "p_value": float(p),
                                    "p_adj": float(p_adj),
                                    "reject": p_adj < 0.05,
                                }
                            )

            elif method == "dunnett":
                from scipy import stats as scipy_stats

                # Compare all groups to control
                control = group_arrays[control_group]
                n_comparisons = len(groups) - 1

                for i, (name, group) in enumerate(zip(names, group_arrays)):
                    if i == control_group:
                        continue
                    stat, p = scipy_stats.ttest_ind(group, control)
                    p_adj = min(p * n_comparisons, 1.0)
                    comparisons.append(
                        {
                            "group": name,
                            "vs_control": names[control_group],
                            "mean_diff": float(np.mean(group) - np.mean(control)),
                            "t_statistic": float(stat),
                            "p_value": float(p),
                            "p_adj": float(p_adj),
                            "reject": p_adj < 0.05,
                        }
                    )

            elif method == "games_howell":
                from scipy import stats as scipy_stats

                # Games-Howell doesn't assume equal variances
                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        g1, g2 = group_arrays[i], group_arrays[j]
                        n1, n2 = len(g1), len(g2)
                        m1, m2 = np.mean(g1), np.mean(g2)
                        v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)

                        se = np.sqrt(v1 / n1 + v2 / n2)
                        t_stat = (m1 - m2) / se

                        # Welch-Satterthwaite df
                        df = (v1 / n1 + v2 / n2) ** 2 / (
                            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
                        )

                        p = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
                        n_comparisons = len(groups) * (len(groups) - 1) // 2
                        p_adj = min(p * n_comparisons, 1.0)

                        comparisons.append(
                            {
                                "group1": names[i],
                                "group2": names[j],
                                "mean_diff": float(m1 - m2),
                                "t_statistic": float(t_stat),
                                "df": float(df),
                                "p_value": float(p),
                                "p_adj": float(p_adj),
                                "reject": p_adj < 0.05,
                            }
                        )

            elif method == "dunn":
                from scipy import stats as scipy_stats

                # Dunn's test for Kruskal-Wallis post-hoc
                all_data = np.concatenate(group_arrays)
                ranks = scipy_stats.rankdata(all_data)

                # Assign ranks to groups
                idx = 0
                group_ranks = []
                for g in group_arrays:
                    group_ranks.append(ranks[idx : idx + len(g)])
                    idx += len(g)

                n_total = len(all_data)
                n_comparisons = len(groups) * (len(groups) - 1) // 2

                for i in range(len(groups)):
                    for j in range(i + 1, len(groups)):
                        n_i, n_j = len(group_arrays[i]), len(group_arrays[j])
                        r_i, r_j = np.mean(group_ranks[i]), np.mean(group_ranks[j])

                        se = np.sqrt(n_total * (n_total + 1) / 12 * (1 / n_i + 1 / n_j))
                        z = (r_i - r_j) / se
                        p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
                        p_adj = min(p * n_comparisons, 1.0)

                        comparisons.append(
                            {
                                "group1": names[i],
                                "group2": names[j],
                                "mean_rank_diff": float(r_i - r_j),
                                "z_statistic": float(z),
                                "p_value": float(p),
                                "p_adj": float(p_adj),
                                "reject": p_adj < 0.05,
                            }
                        )

            else:
                raise ValueError(f"Unknown method: {method}")

            return comparisons

        comparisons = await loop.run_in_executor(None, do_posthoc)

        return {
            "success": True,
            "method": method,
            "n_groups": len(groups),
            "n_comparisons": len(comparisons),
            "comparisons": comparisons,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def p_to_stars_handler(
    p_value: float,
    thresholds: list[float] | None = None,
) -> dict:
    """Convert p-value to significance stars."""
    try:
        thresh = thresholds or [0.001, 0.01, 0.05]

        if p_value < thresh[0]:
            stars = "***"
            significance = f"p < {thresh[0]}"
        elif p_value < thresh[1]:
            stars = "**"
            significance = f"p < {thresh[1]}"
        elif p_value < thresh[2]:
            stars = "*"
            significance = f"p < {thresh[2]}"
        else:
            stars = "ns"
            significance = f"p >= {thresh[2]} (not significant)"

        return {
            "success": True,
            "p_value": p_value,
            "stars": stars,
            "significance": significance,
            "thresholds": thresh,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
