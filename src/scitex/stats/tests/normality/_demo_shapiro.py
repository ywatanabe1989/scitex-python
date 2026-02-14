#!/usr/bin/env python3
# Timestamp: "2025-10-01 15:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/normality/_demo_shapiro.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Demo script for Shapiro-Wilk normality test.

Demonstrates various use cases of test_shapiro().
"""

"""Imports"""
import argparse  # noqa: E402

import numpy as np  # noqa: E402

import scitex as stx  # noqa: E402
from scitex.logging import getLogger  # noqa: E402

logger = getLogger(__name__)

"""Main function"""


def main(args):
    """Demonstrate Shapiro-Wilk test functionality."""
    from ._test_shapiro import test_normality, test_shapiro

    logger.info("Demonstrating Shapiro-Wilk normality test")

    # Set random seed
    np.random.seed(42)

    # Example 1: Normal data
    logger.info("\n=== Example 1: Normal data ===")

    x_normal = np.random.normal(0, 1, 100)
    _ = test_shapiro(x_normal, var_x="Normal", verbose=True)

    # Example 2: Non-normal data (exponential)
    logger.info("\n=== Example 2: Non-normal data (exponential) ===")

    x_exp = np.random.exponential(2, 100)
    _ = test_shapiro(x_exp, var_x="Exponential", verbose=True)

    # Example 3: With Q-Q plot
    logger.info("\n=== Example 3: Visual assessment with Q-Q plot ===")

    x_mixed = np.concatenate(
        [
            np.random.normal(0, 1, 90),
            np.random.normal(5, 1, 10),  # Outliers
        ]
    )

    _ = test_shapiro(x_mixed, var_x="Mixed Distribution", plot=True, verbose=True)
    stx.io.save(stx.plt.gcf(), "./shapiro_example3.jpg")
    stx.plt.close()

    # Example 4: Multiple samples check
    logger.info("\n=== Example 4: Check multiple samples ===")

    x1 = np.random.normal(0, 1, 50)
    x2 = np.random.exponential(2, 50)
    x3 = np.random.normal(0, 1, 50)

    check_result = test_normality(
        x1, x2, x3, var_names=["Sample A", "Sample B", "Sample C"], warn=True
    )

    logger.info(f"All normal: {check_result['all_normal']}")
    logger.info(f"Recommendation: {check_result['recommendation']}")

    # Example 5: Different distributions comparison
    logger.info("\n=== Example 5: Distribution comparison ===")

    distributions = {
        "Normal": np.random.normal(0, 1, 100),
        "Exponential": np.random.exponential(2, 100),
        "Uniform": np.random.uniform(-3, 3, 100),
        "Gamma": np.random.gamma(2, 2, 100),
        "t-dist (df=3)": np.random.standard_t(3, 100),
    }

    results_comp = []
    for name, data in distributions.items():
        result = test_shapiro(data, var_x=name, verbose=True)
        results_comp.append(result)

    # Example 6: Export results
    logger.info("\n=== Example 6: Export results ===")

    from scitex.stats._utils._normalizers import export_summary

    export_summary(
        results_comp,
        "./shapiro_results.csv",
        columns=[
            "var_x",
            "statistic",
            "pvalue",
            "stars",
            "normal",
            "recommendation",
        ],
    )
    logger.info("Results exported to ./shapiro_results.csv")

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Demonstrate Shapiro-Wilk normality test"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    global CONFIG, sys, plt

    import sys

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(  # type: ignore[name-defined]
        sys,  # type: ignore[name-defined]
        plt,  # type: ignore[name-defined]
        args=args,
        file=__file__,
        verbose=args.verbose,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,  # type: ignore[name-defined]
        verbose=args.verbose,
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
