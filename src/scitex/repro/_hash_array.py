#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-09-14 02:18:30 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/scitex_repo/src/scitex/reproduce/_hash_array.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import hashlib

import numpy as np


def hash_array(array_data: np.ndarray) -> str:
    """Generate hash for array data.

    Creates a deterministic hash for numpy arrays, useful for
    verifying data integrity and reproducibility.

    Parameters
    ----------
    array_data : np.ndarray
        Array to hash

    Returns
    -------
    str
        16-character hash string

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> hash1 = hash_array(data)
    >>> hash2 = hash_array(data)
    >>> hash1 == hash2
    True
    """
    data_bytes = array_data.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()[:16]


# ================================================================================
# Example Usage
# ================================================================================
def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate array hashing")
    parser.add_argument(
        "--size", type=int, default=100, help="Array size (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main(args):
    """Main execution function.

    Demonstrates array hashing for reproducibility verification.
    """
    print(f"\n{'=' * 60}")
    print("Array Hashing Demo")
    print(f"{'=' * 60}")
    print(f"Array size: {args.size}")
    print(f"Seed: {args.seed}")

    # Generate arrays using rng
    gen = rng("demo")

    # Create array and hash it
    print(f"\n{'Hash Generation':-^60}")
    data1 = gen.random(args.size)
    hash1 = hash_array(data1)
    print(f"Array 1 hash: {hash1}")

    # Same data should produce same hash
    hash1_again = hash_array(data1)
    print(f"Array 1 hash (again): {hash1_again}")
    print(f"Hashes match: {hash1 == hash1_again}")

    # Different data should produce different hash
    print(f"\n{'Different Data':-^60}")
    data2 = gen.random(args.size)
    hash2 = hash_array(data2)
    print(f"Array 2 hash: {hash2}")
    print(f"Hashes differ: {hash1 != hash2}")

    # Reset generator and create same data
    print(f"\n{'Reproducibility Check':-^60}")
    gen_repro = rng("demo")  # Same name = same seed
    data3 = gen_repro.random(args.size)
    hash3 = hash_array(data3)
    print(f"Array 3 hash (reproduced): {hash3}")
    print(f"Reproduces original: {hash1 == hash3}")

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
        sdir_suffix="hash_array_demo",
        verbose=True,
        agg=True,
        seed=args.seed,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=True,
        notify=False,
        message="Array hashing demo completed",
        exit_status=exit_status,
    )

# EOF
