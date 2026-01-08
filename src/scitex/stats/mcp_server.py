#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/stats/mcp_server.py
# ----------------------------------------

"""
MCP Server for SciTeX Stats - Statistical Testing Framework

Provides tools for:
- Recommending appropriate statistical tests
- Running statistical tests
- Formatting results for publication
- Power analysis and sample size calculation
- Multiple comparison correction
- Descriptive statistics
- Effect size calculation
- Normality testing
- Post-hoc comparisons
"""

from __future__ import annotations

import asyncio

# Graceful MCP dependency handling
try:
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    types = None  # type: ignore
    Server = None  # type: ignore
    NotificationOptions = None  # type: ignore
    InitializationOptions = None  # type: ignore
    stdio_server = None  # type: ignore

__all__ = ["StatsServer", "main", "MCP_AVAILABLE"]


class StatsServer:
    """MCP Server for Statistical Testing Framework."""

    def __init__(self):
        self.server = Server("scitex-stats")
        self.setup_handlers()

    def setup_handlers(self):
        """Set up MCP server handlers."""
        from ._mcp_handlers import (
            correct_pvalues_handler,
            describe_handler,
            effect_size_handler,
            format_results_handler,
            normality_test_handler,
            p_to_stars_handler,
            posthoc_test_handler,
            power_analysis_handler,
            recommend_tests_handler,
            run_test_handler,
        )
        from ._mcp_tool_schemas import get_tool_schemas

        @self.server.list_tools()
        async def handle_list_tools():
            return get_tool_schemas()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            # Test Recommendation
            if name == "recommend_tests":
                return await self._wrap_result(recommend_tests_handler(**arguments))

            # Run Statistical Test
            elif name == "run_test":
                return await self._wrap_result(run_test_handler(**arguments))

            # Format Results
            elif name == "format_results":
                return await self._wrap_result(format_results_handler(**arguments))

            # Power Analysis
            elif name == "power_analysis":
                return await self._wrap_result(power_analysis_handler(**arguments))

            # Correct P-values
            elif name == "correct_pvalues":
                return await self._wrap_result(correct_pvalues_handler(**arguments))

            # Descriptive Statistics
            elif name == "describe":
                return await self._wrap_result(describe_handler(**arguments))

            # Effect Size
            elif name == "effect_size":
                return await self._wrap_result(effect_size_handler(**arguments))

            # Normality Test
            elif name == "normality_test":
                return await self._wrap_result(normality_test_handler(**arguments))

            # Post-hoc Test
            elif name == "posthoc_test":
                return await self._wrap_result(posthoc_test_handler(**arguments))

            # P-value to Stars
            elif name == "p_to_stars":
                return await self._wrap_result(p_to_stars_handler(**arguments))

            else:
                raise ValueError(f"Unknown tool: {name}")

        @self.server.list_resources()
        async def handle_list_resources():
            """List available statistical resources."""
            resources = [
                types.Resource(
                    uri="stats://tests/parametric",
                    name="Parametric Tests",
                    description="List of available parametric statistical tests",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="stats://tests/nonparametric",
                    name="Non-parametric Tests",
                    description="List of available non-parametric statistical tests",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="stats://effect-sizes",
                    name="Effect Size Measures",
                    description="Available effect size calculations",
                    mimeType="application/json",
                ),
                types.Resource(
                    uri="stats://correction-methods",
                    name="Multiple Comparison Methods",
                    description="Available p-value correction methods",
                    mimeType="application/json",
                ),
            ]
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str):
            """Read a statistical resource."""
            import json

            if uri == "stats://tests/parametric":
                content = json.dumps(
                    {
                        "tests": [
                            {
                                "name": "ttest_ind",
                                "description": "Independent samples t-test",
                                "groups": 2,
                                "assumptions": ["normality", "equal_variance"],
                            },
                            {
                                "name": "ttest_paired",
                                "description": "Paired samples t-test",
                                "groups": 2,
                                "assumptions": ["normality"],
                            },
                            {
                                "name": "ttest_1samp",
                                "description": "One-sample t-test",
                                "groups": 1,
                                "assumptions": ["normality"],
                            },
                            {
                                "name": "anova",
                                "description": "One-way ANOVA",
                                "groups": "2+",
                                "assumptions": ["normality", "equal_variance"],
                            },
                            {
                                "name": "pearson",
                                "description": "Pearson correlation",
                                "groups": 2,
                                "assumptions": ["normality", "linearity"],
                            },
                        ]
                    },
                    indent=2,
                )
                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=content,
                )

            elif uri == "stats://tests/nonparametric":
                content = json.dumps(
                    {
                        "tests": [
                            {
                                "name": "brunner_munzel",
                                "description": "Brunner-Munzel test (recommended)",
                                "groups": 2,
                                "assumptions": [],
                            },
                            {
                                "name": "mannwhitneyu",
                                "description": "Mann-Whitney U test",
                                "groups": 2,
                                "assumptions": [],
                            },
                            {
                                "name": "wilcoxon",
                                "description": "Wilcoxon signed-rank test",
                                "groups": 2,
                                "assumptions": ["paired"],
                            },
                            {
                                "name": "kruskal",
                                "description": "Kruskal-Wallis H test",
                                "groups": "2+",
                                "assumptions": [],
                            },
                            {
                                "name": "spearman",
                                "description": "Spearman rank correlation",
                                "groups": 2,
                                "assumptions": [],
                            },
                            {
                                "name": "kendall",
                                "description": "Kendall tau correlation",
                                "groups": 2,
                                "assumptions": [],
                            },
                        ]
                    },
                    indent=2,
                )
                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=content,
                )

            elif uri == "stats://effect-sizes":
                content = json.dumps(
                    {
                        "measures": [
                            {
                                "name": "cohens_d",
                                "description": "Cohen's d - standardized mean difference",
                                "type": "parametric",
                                "interpretation": {
                                    "small": 0.2,
                                    "medium": 0.5,
                                    "large": 0.8,
                                },
                            },
                            {
                                "name": "hedges_g",
                                "description": "Hedges' g - bias-corrected Cohen's d",
                                "type": "parametric",
                                "interpretation": {
                                    "small": 0.2,
                                    "medium": 0.5,
                                    "large": 0.8,
                                },
                            },
                            {
                                "name": "glass_delta",
                                "description": "Glass's delta - using control group SD",
                                "type": "parametric",
                                "interpretation": {
                                    "small": 0.2,
                                    "medium": 0.5,
                                    "large": 0.8,
                                },
                            },
                            {
                                "name": "cliffs_delta",
                                "description": "Cliff's delta - nonparametric effect size",
                                "type": "nonparametric",
                                "interpretation": {
                                    "negligible": 0.147,
                                    "small": 0.33,
                                    "medium": 0.474,
                                    "large": 0.474,
                                },
                            },
                        ]
                    },
                    indent=2,
                )
                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=content,
                )

            elif uri == "stats://correction-methods":
                content = json.dumps(
                    {
                        "methods": [
                            {
                                "name": "bonferroni",
                                "description": "Bonferroni correction (FWER control)",
                                "conservative": True,
                            },
                            {
                                "name": "holm",
                                "description": "Holm-Bonferroni step-down (FWER control)",
                                "conservative": True,
                            },
                            {
                                "name": "sidak",
                                "description": "Sidak correction (FWER control)",
                                "conservative": True,
                            },
                            {
                                "name": "fdr_bh",
                                "description": "Benjamini-Hochberg (FDR control)",
                                "conservative": False,
                            },
                            {
                                "name": "fdr_by",
                                "description": "Benjamini-Yekutieli (FDR control)",
                                "conservative": False,
                            },
                        ]
                    },
                    indent=2,
                )
                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=content,
                )

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

    async def _wrap_result(self, coro):
        """Wrap handler result as MCP TextContent."""
        import json

        try:
            result = await coro
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str),
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"success": False, "error": str(e)}, indent=2),
                )
            ]


async def _run_server():
    """Run the MCP server (internal)."""
    server = StatsServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-stats",
                server_version="0.1.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    """Main entry point for the MCP server."""
    if not MCP_AVAILABLE:
        import sys

        print("=" * 60)
        print("MCP Server 'scitex-stats' requires the 'mcp' package.")
        print()
        print("Install with:")
        print("  pip install mcp")
        print()
        print("Or install scitex with MCP support:")
        print("  pip install scitex[mcp]")
        print("=" * 60)
        sys.exit(1)

    asyncio.run(_run_server())


if __name__ == "__main__":
    main()


# EOF
