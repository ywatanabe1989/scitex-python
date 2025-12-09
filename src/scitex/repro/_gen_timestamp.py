#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 17:44:32 (ywatanabe)"
# File: ./scitex_repo/src/scitex/repro/_gen_timestamp.py

from datetime import datetime as _datetime


def gen_timestamp():
    """Generate a timestamp string for file naming.

    Returns a timestamp in the format YYYY-MMDD-HHMM, suitable for
    creating unique filenames or version identifiers.

    Returns
    -------
    str
        Timestamp string in format "YYYY-MMDD-HHMM"

    Examples
    --------
    >>> timestamp = gen_timestamp()
    >>> print(timestamp)
    '2025-0531-1230'

    >>> filename = f"experiment_{gen_timestamp()}.csv"
    >>> print(filename)
    'experiment_2025-0531-1230.csv'
    """
    return _datetime.now().strftime("%Y-%m%d-%H%M")


timestamp = gen_timestamp


# ================================================================================
# Example Usage
# ================================================================================
def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate timestamp generation")
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of timestamps to generate (default: 5)",
    )
    return parser.parse_args()


def main(args):
    """Main execution function.

    Demonstrates timestamp generation for file naming.
    """
    import time

    print(f"\n{'=' * 60}")
    print("Timestamp Generation Demo")
    print(f"{'=' * 60}")

    # Generate single timestamp
    print(f"\n{'Single Timestamp':-^60}")
    ts = gen_timestamp()
    print(f"Generated: {ts}")
    print(f"Example usage: experiment_{ts}.csv")

    # Generate multiple timestamps with small delays
    print(f"\n{'Multiple Timestamps':-^60}")
    print(f"Generating {args.count} timestamps...")
    timestamps = []
    for i in range(args.count):
        ts = gen_timestamp()
        timestamps.append(ts)
        print(f"{i + 1}. {ts}")
        time.sleep(0.5)  # Small delay to show time progression

    print(f"\n{'=' * 60}")
    print("Demo completed successfully!")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
        sys,
        plt,
        args=args,
        file=__file__,
        sdir_suffix="gen_timestamp_demo",
        verbose=True,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=True,
        notify=False,
        message="Timestamp generation demo completed",
        exit_status=exit_status,
    )

# EOF
