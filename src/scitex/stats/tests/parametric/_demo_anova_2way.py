#!/usr/bin/env python3
# Timestamp: "2025-10-01 17:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/parametric/_demo_anova_2way.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Demo script for two-way ANOVA examples.

Run with: python -m scitex.stats.tests.parametric._demo_anova_2way
"""

import argparse

import numpy as np
import pandas as pd

import scitex as stx
from scitex.logging import getLogger

from ._test_anova_2way import test_anova_2way

logger = getLogger(__name__)


def main(args):
    """Run demonstration."""
    logger.info("=" * 70)
    logger.info("Two-way ANOVA Examples")
    logger.info("=" * 70)

    # Example 1: Drug × Gender (with interaction)
    logger.info("\n[Example 1] Drug × Gender (interaction present)")
    logger.info("-" * 70)

    np.random.seed(42)

    data = pd.DataFrame(
        {
            "Drug": ["Placebo"] * 30 + ["Active"] * 30,
            "Gender": (["Male"] * 15 + ["Female"] * 15) * 2,
            "Score": np.concatenate(
                [
                    np.random.normal(50, 10, 15),  # Placebo, Male
                    np.random.normal(55, 10, 15),  # Placebo, Female
                    np.random.normal(65, 10, 15),  # Active, Male
                    np.random.normal(75, 10, 15),  # Active, Female (interaction)
                ]
            ),
        }
    )

    results = test_anova_2way(
        data,
        factor_a="Drug",
        factor_b="Gender",
        value="Score",
        plot=True,
        verbose=True,
    )
    stx.io.save(plt.gcf(), "./.dev/anova_2way_example1.jpg")  # type: ignore[name-defined]
    plt.close()  # type: ignore[name-defined]

    for effect in results:
        logger.info(
            f"{effect['effect']:20s}: F({effect['df_effect']},{effect['df_error']}) = "
            f"{effect['statistic']:.3f}, p = {effect['pvalue']:.4f} {effect['stars']}, "
            f"η²p = {effect['effect_size']:.3f}"
        )

    # Example 2: No interaction (additive effects)
    logger.info("\n[Example 2] Temperature × Time (no interaction)")
    logger.info("-" * 70)

    data2 = pd.DataFrame(
        {
            "Temperature": (["Low"] * 20 + ["Medium"] * 20 + ["High"] * 20),
            "Time": (["Short", "Long"] * 30),
            "Yield": np.concatenate(
                [
                    np.random.normal(40, 5, 10)
                    + np.random.normal(0, 2, 10),  # Low, Short
                    np.random.normal(40, 5, 10)
                    + np.random.normal(10, 2, 10),  # Low, Long
                    np.random.normal(50, 5, 10)
                    + np.random.normal(0, 2, 10),  # Medium, Short
                    np.random.normal(50, 5, 10)
                    + np.random.normal(10, 2, 10),  # Medium, Long
                    np.random.normal(60, 5, 10)
                    + np.random.normal(0, 2, 10),  # High, Short
                    np.random.normal(60, 5, 10)
                    + np.random.normal(10, 2, 10),  # High, Long
                ]
            ),
        }
    )

    results2 = test_anova_2way(
        data2,
        factor_a="Temperature",
        factor_b="Time",
        value="Yield",
        plot=True,
        verbose=True,
    )
    stx.io.save(plt.gcf(), "./.dev/anova_2way_example2.jpg")  # type: ignore[name-defined]
    plt.close()  # type: ignore[name-defined]

    logger.info("\nMain effects should be significant, interaction should not be:")
    for effect in results2:
        logger.info(
            f"{effect['effect']:25s}: p = {effect['pvalue']:.4f} ({effect['effect_size_interpretation']})"
        )

    # Example 3: DataFrame output
    logger.info("\n[Example 3] DataFrame output format")
    logger.info("-" * 70)

    results_df = test_anova_2way(
        data,
        factor_a="Drug",
        factor_b="Gender",
        value="Score",
        return_as="dataframe",
        verbose=True,
    )

    logger.info(
        f"\n{results_df[['effect', 'statistic', 'pvalue', 'effect_size', 'significant']].to_string()}"  # type: ignore[call-overload]
    )

    # Example 4: Export
    logger.info("\n[Example 4] Export results")
    logger.info("-" * 70)

    from scitex.stats._utils._normalizers import convert_results

    convert_results(
        results_df,
        return_as="excel",  # type: ignore[arg-type]
        path="./.dev/anova_2way_results.xlsx",
    )
    logger.info("Saved to: ./.dev/anova_2way_results.xlsx")

    # EOF

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    global CONFIG, CC, sys, plt

    import sys

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(  # type: ignore[name-defined]
        sys,  # type: ignore[name-defined]
        stx.plt,
        args=args,
        file=__FILE__,
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
