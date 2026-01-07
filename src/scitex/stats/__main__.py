#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/stats/__main__.py
# ----------------------------------------

"""Stats CLI entry point.

Subcommand-based interface:
- recommend: Recommend statistical tests based on data characteristics
- test: Run a statistical test
- power: Power analysis and sample size calculation
- mcp: Start MCP server for LLM integration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


def create_parser():
    """Create main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="python -m scitex.stats",
        description="""
SciTeX Stats - Statistical Testing Framework
═══════════════════════════════════════════

Subcommand interface:
  recommend - Recommend statistical tests
  test      - Run a statistical test
  power     - Power analysis
  mcp       - Start MCP server for LLM integration
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    # ========================================
    # Subcommand: recommend
    # ========================================
    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Recommend statistical tests",
        description="Recommend appropriate statistical tests based on data characteristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    recommend_parser.add_argument(
        "--n-groups",
        type=int,
        default=2,
        help="Number of groups to compare (default: 2)",
    )
    recommend_parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        help="Sample sizes for each group",
    )
    recommend_parser.add_argument(
        "--outcome",
        type=str,
        choices=["continuous", "ordinal", "categorical", "binary"],
        default="continuous",
        help="Outcome variable type (default: continuous)",
    )
    recommend_parser.add_argument(
        "--design",
        type=str,
        choices=["between", "within", "mixed"],
        default="between",
        help="Study design (default: between)",
    )
    recommend_parser.add_argument(
        "--paired",
        action="store_true",
        help="Data is paired/matched",
    )
    recommend_parser.add_argument(
        "--has-control",
        action="store_true",
        help="Has a control group",
    )
    recommend_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of recommendations (default: 3)",
    )

    # ========================================
    # Subcommand: test
    # ========================================
    test_parser = subparsers.add_parser(
        "test",
        help="Run a statistical test",
        description="Execute a statistical test on provided data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    test_parser.add_argument(
        "test_name",
        type=str,
        help="Name of the test to run",
    )
    test_parser.add_argument(
        "--data",
        type=str,
        help="JSON array of group data, e.g., '[[1,2,3],[4,5,6]]'",
    )
    test_parser.add_argument(
        "--alternative",
        type=str,
        choices=["two-sided", "less", "greater"],
        default="two-sided",
        help="Alternative hypothesis (default: two-sided)",
    )

    # ========================================
    # Subcommand: power
    # ========================================
    power_parser = subparsers.add_parser(
        "power",
        help="Power analysis",
        description="Calculate statistical power or required sample size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    power_parser.add_argument(
        "--test-type",
        type=str,
        choices=["ttest", "anova", "correlation", "chi2"],
        default="ttest",
        help="Type of statistical test (default: ttest)",
    )
    power_parser.add_argument(
        "--effect-size",
        type=float,
        help="Expected effect size (Cohen's d, f, r, or w)",
    )
    power_parser.add_argument(
        "--n",
        type=int,
        help="Sample size per group (for power calculation)",
    )
    power_parser.add_argument(
        "--power",
        type=float,
        default=0.8,
        help="Desired power (default: 0.8)",
    )
    power_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level (default: 0.05)",
    )

    # ========================================
    # Subcommand: mcp
    # ========================================
    subparsers.add_parser(
        "mcp",
        help="Start MCP server for LLM integration",
        description="Start the MCP (Model Context Protocol) server for Claude/LLM integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    return parser


def run_recommend(args):
    """Run test recommendation."""
    from scitex.stats.auto import StatContext, recommend_tests
    from scitex.stats.auto._rules import TEST_RULES

    ctx = StatContext(
        n_groups=args.n_groups,
        sample_sizes=args.sample_sizes or [30] * args.n_groups,
        outcome_type=args.outcome,
        design=args.design,
        paired=args.paired,
        has_control_group=args.has_control,
        n_factors=1,
    )

    tests = recommend_tests(ctx, top_k=args.top_k)

    print("\n=== Recommended Statistical Tests ===\n")
    for i, test_name in enumerate(tests, 1):
        rule = TEST_RULES.get(test_name)
        if rule:
            print(f"{i}. {test_name}")
            print(f"   Family: {rule.family}")
            print(f"   Priority: {rule.priority}")
            if rule.needs_normality:
                print("   Requires: normality assumption")
            if rule.needs_equal_variance:
                print("   Requires: equal variance assumption")
            print()

    return 0


def run_test(args):
    """Run a statistical test."""
    if not args.data:
        print("Error: --data is required")
        return 1

    data = json.loads(args.data)

    from scitex.stats._mcp_handlers import run_test_handler

    result = asyncio.run(
        run_test_handler(
            test_name=args.test_name,
            data=data,
            alternative=args.alternative,
        )
    )

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


def run_power(args):
    """Run power analysis."""
    from scitex.stats._mcp_handlers import power_analysis_handler

    result = asyncio.run(
        power_analysis_handler(
            test_type=args.test_type,
            effect_size=args.effect_size,
            n=args.n,
            power=args.power,
            alpha=args.alpha,
        )
    )

    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


async def run_mcp_server():
    """Run MCP server."""
    from .mcp_server import main as mcp_main

    print("Starting Stats MCP server...", file=sys.stderr)
    await mcp_main()
    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "recommend":
        return run_recommend(args)
    elif args.command == "test":
        return run_test(args)
    elif args.command == "power":
        return run_power(args)
    elif args.command == "mcp":
        return asyncio.run(run_mcp_server())
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


# EOF
