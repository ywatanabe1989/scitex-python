#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Using verbose option for metadata operations

This demo shows how to use the verbose=True option to see detailed
logging information about metadata embedding and loading operations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
import scitex as stx
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint


def demo_with_verbose(filename="demo_fig_with_metadata.jpg"):
    """Show metadata embedding and loading with verbose logging."""

    # Create a simple figure
    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)

    ax.plot(t, signal, "b-", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Damped Sine Wave")
    ax.grid(True, alpha=0.3)

    # Get output directory
    ldir = Path(__file__).parent / "output"

    # Saving with metadata (verbose logging shows the process)
    print("\n📝 Saving figure with metadata...")
    print("=" * 70)
    stx.io.save(
        fig,
        f"{ldir}/{filename}",
        metadata={"exp": "s01", "subj": "S001"},
        verbose=True,
    )
    plt.close()

    # Loading and verifying (verbose logging shows what was found)
    print("\n✅ Loading and verifying metadata...")
    print("=" * 70)
    img, meta = stx.io.load(f"{ldir}/{filename}", verbose=True)

    print("\nEmbedded metadata:")
    print("=" * 70)
    pprint(meta)

    return img, meta


if __name__ == "__main__":
    print("=" * 70)
    print("Demo: Verbose Metadata Operations")
    print("=" * 70)

    img, meta = demo_with_verbose()

    print("\n" + "=" * 70)
    print("✓ Demo completed!")
    print("=" * 70)
