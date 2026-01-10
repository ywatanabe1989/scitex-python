#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-07 17:53:38 (ywatanabe)"
# File: ./scitex_repo/src/scitex/repro/_gen_ID.py

import random as _random
import string as _string
from datetime import datetime as _datetime


def gen_id(time_format="%YY-%mM-%dD-%Hh%Mm%Ss", N=8):
    """Generate a unique identifier with timestamp and random characters.

    Creates a unique ID by combining a formatted timestamp with random
    alphanumeric characters. Useful for creating unique experiment IDs,
    run identifiers, or temporary file names.

    Parameters
    ----------
    time_format : str, optional
        Format string for timestamp portion. Default is "%YY-%mM-%dD-%Hh%Mm%Ss"
        which produces "2025Y-05M-31D-12h30m45s" format.
    N : int, optional
        Number of random characters to append. Default is 8.

    Returns
    -------
    str
        Unique identifier in format "{timestamp}_{random_chars}"

    Examples
    --------
    >>> id1 = gen_id()
    >>> print(id1)
    '2025Y-05M-31D-12h30m45s_a3Bc9xY2'

    >>> id2 = gen_id(time_format="%Y%m%d", N=4)
    >>> print(id2)
    '20250531_xY9a'

    >>> # For experiment tracking
    >>> exp_id = gen_id()
    >>> save_path = f"results/experiment_{exp_id}.pkl"
    """
    now_str = _datetime.now().strftime(time_format)
    rand_str = "".join(
        [_random.choice(_string.ascii_letters + _string.digits) for i in range(N)]
    )
    return now_str + "_" + rand_str


# Backward compatibility
gen_ID = gen_id  # Deprecated: use gen_id instead


# ================================================================================
# Example Usage
# ================================================================================
def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate ID generation")
    parser.add_argument(
        "--format",
        type=str,
        default="%YY-%mM-%dD-%Hh%Mm%Ss",
        help="Time format (default: %%YY-%%mM-%%dD-%%Hh%%Mm%%Ss)",
    )
    parser.add_argument(
        "--length", type=int, default=8, help="Random string length (default: 8)"
    )
    return parser.parse_args()


def main(args):
    """Main execution function.

    Demonstrates ID generation with different formats.
    """
    print(f"\n{'=' * 60}")
    print("ID Generation Demo")
    print(f"{'=' * 60}")

    # Generate with default format
    print(f"\n{'Default Format':-^60}")
    id1 = gen_id()
    print(f"Generated ID: {id1}")

    # Generate with custom format
    print(f"\n{'Custom Format':-^60}")
    id2 = gen_id(time_format=args.format, N=args.length)
    print(f"Format: {args.format}")
    print(f"Length: {args.length}")
    print(f"Generated ID: {id2}")

    # Generate multiple IDs
    print(f"\n{'Multiple IDs':-^60}")
    ids = [gen_id(N=4) for _ in range(5)]
    for i, id_str in enumerate(ids, 1):
        print(f"{i}. {id_str}")

    print(f"\n{'=' * 60}")
    print("Demo completed successfully!")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys,
        plt,
        args=args,
        file=__file__,
        sdir_suffix="gen_ID_demo",
        verbose=True,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=True,
        notify=False,
        message="ID generation demo completed",
        exit_status=exit_status,
    )

# EOF
