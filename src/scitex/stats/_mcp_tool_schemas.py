#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/stats/_mcp_tool_schemas.py
# ----------------------------------------

"""Tool schemas for the scitex-stats MCP server."""

from __future__ import annotations

import mcp.types as types

__all__ = ["get_tool_schemas"]


def get_tool_schemas() -> list[types.Tool]:
    """Return all tool schemas for the Stats MCP server."""
    return [
        # Test Recommendation
        types.Tool(
            name="recommend_tests",
            description=(
                "Recommend appropriate statistical tests based on data characteristics. "
                "Returns ranked list of tests with rationale."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n_groups": {
                        "type": "integer",
                        "description": "Number of groups to compare",
                        "default": 2,
                    },
                    "sample_sizes": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Sample sizes for each group",
                    },
                    "outcome_type": {
                        "type": "string",
                        "description": "Type of outcome variable",
                        "enum": ["continuous", "ordinal", "categorical", "binary"],
                        "default": "continuous",
                    },
                    "design": {
                        "type": "string",
                        "description": "Study design",
                        "enum": ["between", "within", "mixed"],
                        "default": "between",
                    },
                    "paired": {
                        "type": "boolean",
                        "description": "Whether data is paired/matched",
                        "default": False,
                    },
                    "has_control_group": {
                        "type": "boolean",
                        "description": "Whether there is a control group",
                        "default": False,
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of recommendations to return",
                        "default": 3,
                    },
                },
            },
        ),
        # Run Statistical Test
        types.Tool(
            name="run_test",
            description=(
                "Execute a statistical test on provided data. "
                "Returns test statistic, p-value, effect size, and confidence intervals."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "Name of test to run",
                        "enum": [
                            "ttest_ind",
                            "ttest_paired",
                            "ttest_1samp",
                            "brunner_munzel",
                            "mannwhitneyu",
                            "wilcoxon",
                            "anova",
                            "kruskal",
                            "chi2",
                            "fisher_exact",
                            "pearson",
                            "spearman",
                            "kendall",
                        ],
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Data arrays for each group",
                    },
                    "alternative": {
                        "type": "string",
                        "description": "Alternative hypothesis",
                        "enum": ["two-sided", "less", "greater"],
                        "default": "two-sided",
                    },
                },
                "required": ["test_name", "data"],
            },
        ),
        # Format Results
        types.Tool(
            name="format_results",
            description=(
                "Format statistical results in journal style (APA, Nature, etc.)"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "test_name": {
                        "type": "string",
                        "description": "Name of the statistical test",
                    },
                    "statistic": {
                        "type": "number",
                        "description": "Test statistic value",
                    },
                    "p_value": {
                        "type": "number",
                        "description": "P-value",
                    },
                    "df": {
                        "type": "number",
                        "description": "Degrees of freedom (if applicable)",
                    },
                    "effect_size": {
                        "type": "number",
                        "description": "Effect size value",
                    },
                    "effect_size_name": {
                        "type": "string",
                        "description": "Name of effect size measure (d, r, eta2, etc.)",
                    },
                    "style": {
                        "type": "string",
                        "description": "Journal formatting style",
                        "enum": ["apa", "nature", "science", "brief"],
                        "default": "apa",
                    },
                    "ci_lower": {
                        "type": "number",
                        "description": "Lower bound of confidence interval",
                    },
                    "ci_upper": {
                        "type": "number",
                        "description": "Upper bound of confidence interval",
                    },
                },
                "required": ["test_name", "statistic", "p_value"],
            },
        ),
        # Power Analysis
        types.Tool(
            name="power_analysis",
            description=(
                "Calculate statistical power or required sample size. "
                "Supports various test types."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "test_type": {
                        "type": "string",
                        "description": "Type of statistical test",
                        "enum": ["ttest", "anova", "correlation", "chi2"],
                        "default": "ttest",
                    },
                    "effect_size": {
                        "type": "number",
                        "description": "Expected effect size (Cohen's d, f, r, w)",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Significance level",
                        "default": 0.05,
                    },
                    "power": {
                        "type": "number",
                        "description": "Desired statistical power (for sample size calculation)",
                        "default": 0.8,
                    },
                    "n": {
                        "type": "integer",
                        "description": "Sample size (for power calculation)",
                    },
                    "n_groups": {
                        "type": "integer",
                        "description": "Number of groups (for ANOVA)",
                        "default": 2,
                    },
                    "ratio": {
                        "type": "number",
                        "description": "Ratio of group sizes (n2/n1)",
                        "default": 1.0,
                    },
                },
            },
        ),
        # Correct P-values
        types.Tool(
            name="correct_pvalues",
            description=(
                "Apply multiple comparison correction to p-values. "
                "Supports Bonferroni, FDR (Benjamini-Hochberg), Holm, and Sidak methods."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pvalues": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of p-values to correct",
                    },
                    "method": {
                        "type": "string",
                        "description": "Correction method",
                        "enum": ["bonferroni", "fdr_bh", "fdr_by", "holm", "sidak"],
                        "default": "fdr_bh",
                    },
                    "alpha": {
                        "type": "number",
                        "description": "Family-wise error rate",
                        "default": 0.05,
                    },
                },
                "required": ["pvalues"],
            },
        ),
        # Descriptive Statistics
        types.Tool(
            name="describe",
            description=(
                "Calculate descriptive statistics for data. "
                "Returns mean, std, median, quartiles, skewness, kurtosis."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Data array to describe",
                    },
                    "percentiles": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Percentiles to calculate (0-100)",
                        "default": [25, 50, 75],
                    },
                },
                "required": ["data"],
            },
        ),
        # Effect Size Calculation
        types.Tool(
            name="effect_size",
            description=(
                "Calculate effect size between groups. "
                "Returns Cohen's d, Hedges' g, or other appropriate measure."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "group1": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "First group data",
                    },
                    "group2": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Second group data",
                    },
                    "measure": {
                        "type": "string",
                        "description": "Effect size measure",
                        "enum": ["cohens_d", "hedges_g", "glass_delta", "cliffs_delta"],
                        "default": "cohens_d",
                    },
                    "pooled": {
                        "type": "boolean",
                        "description": "Use pooled standard deviation",
                        "default": True,
                    },
                },
                "required": ["group1", "group2"],
            },
        ),
        # Normality Test
        types.Tool(
            name="normality_test",
            description=(
                "Test whether data follows a normal distribution. "
                "Returns test statistic and p-value."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Data to test for normality",
                    },
                    "method": {
                        "type": "string",
                        "description": "Normality test method",
                        "enum": ["shapiro", "dagostino", "anderson", "lilliefors"],
                        "default": "shapiro",
                    },
                },
                "required": ["data"],
            },
        ),
        # Post-hoc Tests
        types.Tool(
            name="posthoc_test",
            description=(
                "Run post-hoc pairwise comparisons after significant ANOVA/Kruskal. "
                "Supports Tukey HSD, Dunnett, Games-Howell, Dunn."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "groups": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                        "description": "Data arrays for each group",
                    },
                    "group_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names for each group",
                    },
                    "method": {
                        "type": "string",
                        "description": "Post-hoc method",
                        "enum": ["tukey", "dunnett", "games_howell", "dunn"],
                        "default": "tukey",
                    },
                    "control_group": {
                        "type": "integer",
                        "description": "Index of control group (for Dunnett)",
                        "default": 0,
                    },
                },
                "required": ["groups"],
            },
        ),
        # P-value to Stars
        types.Tool(
            name="p_to_stars",
            description=("Convert p-value to significance stars (*, **, ***, ns)"),
            inputSchema={
                "type": "object",
                "properties": {
                    "p_value": {
                        "type": "number",
                        "description": "P-value to convert",
                    },
                    "thresholds": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Significance thresholds",
                        "default": [0.001, 0.01, 0.05],
                    },
                },
                "required": ["p_value"],
            },
        ),
    ]


# EOF
