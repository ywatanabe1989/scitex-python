#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_06_wide_format.py

"""
Demo 06: Wide format figure (1x4) - Horizontal layout

Port: 5056

Usage:
    ./demo_06_wide_format.py              # Flask backend (default)
    ./demo_06_wide_format.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5056


def create_figure(output_dir: Path) -> Path:
    """Create figure with different aspect ratio (wide)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    STYLE["fig_size_mm"] = (180, 60)  # Wide aspect ratio
    fig, axes = stx.plt.subplots(1, 4, **STYLE)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    axes[0].plot(x, y, "-", id="line-1")
    axes[0].set_title("Panel A")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    axes[1].scatter(np.random.randn(30), np.random.randn(30), id="scatter-1")
    axes[1].set_title("Panel B")
    axes[1].set_xlabel("X")

    axes[2].bar([0, 1, 2], [1, 2, 1.5], id="bar-1")
    axes[2].set_title("Panel C")
    axes[2].set_xlabel("Group")

    axes[3].hist(np.random.randn(100), bins=15, id="hist-1")
    axes[3].set_title("Panel D")
    axes[3].set_xlabel("Value")

    png_path = output_dir / "06_wide_format.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


@stx.session
def main(
    backend: Backend = "flask",
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """
    Demo 06: Wide format figure (1x4)

    Features:
        - Wide aspect ratio (180x60mm)
        - Four horizontal panels
        - Line, Scatter, Bar, Histogram

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5056
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 06: Wide format (1x4)")
    logger.info(f"Port: {PORT}")
    logger.info("=" * 60)

    json_path = create_figure(out)
    logger.info(f"Created: {json_path}")

    logger.info(f"\nLaunching editor (backend={backend}, port={PORT})...")
    stx.vis.edit(str(json_path), backend=backend, port=PORT)

    return 0


if __name__ == "__main__":
    main()

# EOF
