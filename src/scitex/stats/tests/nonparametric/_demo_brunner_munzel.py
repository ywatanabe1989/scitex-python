#!/usr/bin/env python3
# Timestamp: "2025-10-01 22:40:43 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/nonparametric/_demo_brunner_munzel.py

"""
Demo script for Brunner-Munzel test.

Demonstrates various use cases of test_brunner_munzel().
"""

from __future__ import annotations

import argparse
import os

import numpy as np

import scitex as stx
from scitex.logging import getLogger
from scitex.stats._utils._normalizers import export_results, export_summary

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

logger = getLogger(__name__)


def main(args):  # noqa: C901
    """Demonstrate Brunner-Munzel test functionality."""
    from ._test_brunner_munzel import test_brunner_munzel

    logger.info("Demonstrating Brunner-Munzel test")
    np.random.seed(42)

    def example_01_normal_distributions():
        """Run example 01 with normal distributions."""
        logger.info("\n=== Example 1: Normal distributions ===")
        x1 = np.random.normal(0, 1, 50)
        y1 = np.random.normal(0.6, 1, 50)
        test_brunner_munzel(
            x1, y1, var_x="Control", var_y="Treatment", plot=True, verbose=True
        )
        stx.io.save(stx.plt.gcf(), "./example_01_normal_distributions.jpg")
        return x1, y1

    def example_02_skewed_distributions():
        """Run example 02 with skewed distributions."""
        logger.info("\n=== Example 2: Non-normal (skewed) distributions ===")
        x2 = np.random.gamma(2, 2, 40)
        y2 = np.random.gamma(3, 2, 40)
        result_df = test_brunner_munzel(
            x2,
            y2,
            var_x="Group A",
            var_y="Group B",
            return_as="dataframe",
            plot=True,
            verbose=True,
        )
        logger.info(f"Cliff's δ = {result_df['effect_size_secondary'].iloc[0]:.3f}")
        stx.io.save(stx.plt.gcf(), "./example_02_skewed_distributions.jpg")
        stx.io.save(result_df, "./example_02_skewed_distributions.csv")
        stx.io.save(result_df, "./example_02_skewed_distributions.xlsx")

    def example_03_data_with_outliers():
        """Run example 03 with data containing outliers."""
        logger.info("\n=== Example 3: Data with outliers ===")
        x3 = np.concatenate([np.random.normal(0, 1, 35), [10, 12]])
        y3 = np.random.normal(0.5, 1, 40)
        result_df = test_brunner_munzel(
            x3,
            y3,
            var_x="With Outliers",
            var_y="Normal",
            return_as="dataframe",
            plot=True,
            verbose=True,
        )
        stx.io.save(stx.plt.gcf(), "./example_03_data_with_outliers.jpg")
        stx.io.save(result_df, "./example_03_data_with_outliers.csv")
        stx.io.save(result_df, "./example_03_data_with_outliers.xlsx")

    def example_04_unequal_variances():
        """Run example 04 with unequal variances."""
        logger.info("\n=== Example 4: Unequal variances ===")
        x4 = np.random.normal(0, 1, 50)
        y4 = np.random.normal(0.5, 3, 50)
        result_df = test_brunner_munzel(
            x4,
            y4,
            var_x="Low Variance",
            var_y="High Variance",
            return_as="dataframe",
            plot=True,
            verbose=True,
        )
        stx.io.save(stx.plt.gcf(), "./example_04_unequal_variances.jpg")
        stx.io.save(result_df, "./example_04_unequal_variances.csv")
        stx.io.save(result_df, "./example_04_unequal_variances.xlsx")
        logger.info(f"Variance ratio: {np.var(y4) / np.var(x4):.1f}")

    def example_05_one_sided_test():
        """Run example 05 with one-sided test."""
        logger.info("\n=== Example 5: One-sided test ===")
        x5 = np.random.normal(0, 1, 40)
        y5 = np.random.normal(0.8, 1, 40)
        test_brunner_munzel(x5, y5, alternative="two-sided", plot=True, verbose=True)
        test_brunner_munzel(x5, y5, alternative="less", plot=True, verbose=True)

    def example_06_with_visualization():
        """Run example 06 with visualization."""
        logger.info("\n=== Example 6: With visualization ===")
        x6 = np.random.exponential(2, 50)
        y6 = np.random.exponential(3, 50)
        test_brunner_munzel(
            x6,
            y6,
            var_x="Exponential (λ=0.5)",
            var_y="Exponential (λ=0.33)",
            return_as="dataframe",
            plot=True,
            verbose=True,
        )
        stx.io.save(stx.plt.gcf(), "./example_06_with_visualization.jpg")

    def example_07_dataframe_output():
        """Run example 07 with DataFrame output."""
        logger.info("\n=== Example 7: DataFrame output ===")
        x1, y1 = example_01_normal_distributions()
        df_result = test_brunner_munzel(x1, y1, return_as="dataframe")
        logger.info(f"\n{df_result.T}")  # type: ignore[union-attr]

    def example_08_multiple_comparisons():
        """Run example 08 with multiple comparisons."""
        logger.info("\n=== Example 8: Multiple comparisons ===")
        from scitex.stats._utils._normalizers import combine_results

        results_list = []
        for ii in range(5):
            x_temp = np.random.exponential(2, 30)
            y_temp = np.random.exponential(2.5, 30)
            result_temp = test_brunner_munzel(
                x_temp, y_temp, var_x=f"Control_{ii}", var_y=f"Treatment_{ii}"
            )
            results_list.append(result_temp)
        df_all = combine_results(results_list)
        logger.info(
            f"\n{df_all[['var_x', 'var_y', 'pvalue', 'stars', 'effect_size', 'effect_size_secondary']]}"
        )
        return df_all

    def example_09_export_to_various_formats():
        """Run example 09 exporting to various formats."""
        logger.info("\n=== Example 9: Export to various formats ===")
        df_all = example_08_multiple_comparisons()
        csv_path = export_results(
            df_all, "./example_09_export_to_various_formats_results.csv"
        )
        logger.info(f"Exported full results to: {csv_path}")
        summary_path = export_summary(
            df_all, "./example_09_export_to_various_formats_summary.csv"
        )
        logger.info(f"Exported summary to: {summary_path}")
        json_path = export_results(
            df_all, "./example_09_export_to_various_formats_results.json"
        )
        logger.info(f"Exported to JSON: {json_path}")
        try:
            xlsx_path = export_results(
                df_all, "./example_09_export_to_various_formats_results.xlsx"
            )
            logger.info(f"Exported to Excel: {xlsx_path}")
        except ImportError:
            logger.warning("openpyxl not available, skipping Excel export")
        latex_path = export_summary(
            df_all,
            "./example_09_export_to_various_formats_table.tex",
            columns=[
                "var_x",
                "var_y",
                "pvalue",
                "stars",
                "effect_size",
                "effect_size_secondary",
            ],
        )
        logger.info(f"Exported LaTeX table: {latex_path}")
        txt_path = export_results(
            df_all,
            "./example_09_export_to_various_formats_results.txt",
            format="txt",
        )
        logger.info(f"Exported to text: {txt_path}")

    example_01_normal_distributions()
    example_02_skewed_distributions()
    example_03_data_with_outliers()
    example_04_unequal_variances()
    example_05_one_sided_test()
    example_06_with_visualization()
    example_07_dataframe_output()
    example_08_multiple_comparisons()
    example_09_export_to_various_formats()

    return 0


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Demonstrate Brunner-Munzel test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_main():
    """Initialize SciTeX framework and run main."""
    import sys

    import matplotlib.pyplot as plt  # noqa: F401

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, _CC, _rng_manager = stx.session.start(
        sys,
        plt,
        args=args,
        file=__file__,
        verbose=args.verbose,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=args.verbose,
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
