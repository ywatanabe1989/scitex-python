#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-14 08:36:26 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/metadata_demo/demo_jpg.py


"""Demo: Minimal metadata embedding in JPG file"""


import numpy as np
import matplotlib.pyplot as plt
import scitex as stx


def demo_without_qr(filename, verbose=False):
    """Show metadata without QR code (just embedded)."""

    fig, ax = stx.plt.subplots()

    t = np.linspace(0, 2, 1000)
    signal = np.sin(2 * np.pi * 5 * t) * np.exp(-t / 2)

    ax.plot(t, signal, "b-", linewidth=2)
    ax.set_xyt(
        "Time (s)",
        "Amplitude",
        "Clean Figure (metadata embedded, no QR overlay)",
    )

    # Saving
    stx.io.save(
        fig,
        filename,
        metadata={"exp": "s01", "subj": "S001"},
        symlink_to="./data",
        verbose=verbose,
    )
    plt.close()

    # Loading
    ldir = __file__.replace(".py", "_out")
    img, meta = stx.io.load(
        f"{ldir}/{filename}",
        verbose=verbose,
    )


@stx.session.session
def main(filename="demo_fig_with_metadata.jpg", verbose=True):
    """Run all demos."""

    demo_without_qr(filename, verbose=verbose)

    return 0


if __name__ == "__main__":
    main()

# EOF
