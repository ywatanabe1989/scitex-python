#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_07_tall_format.py

"""
Demo 07: Tall format figure (4x1) - Vertical layout

Port: 5057

Usage:
    ./demo_07_tall_format.py              # Flask backend (default)
    ./demo_07_tall_format.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5057


def create_figure(output_dir: Path) -> Path:
    """Create figure with tall aspect ratio."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    STYLE["fig_size_mm"] = (60, 180)  # Tall aspect ratio
    fig, axes = stx.plt.subplots(4, 1, **STYLE)

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    axes[0].plot(x, y, "-", id="line-tall-1")
    axes[0].set_title("A")
    axes[0].set_ylabel("Y")

    axes[1].scatter(
        np.random.randn(30), np.random.randn(30), id="scatter-tall-1"
    )
    axes[1].set_title("B")
    axes[1].set_ylabel("Y")

    groups = ["A", "B", "C"]
    group_data = [np.random.randn(20) + i for i in range(3)]
    axes[2].stx_box(group_data, labels=groups, id="box-tall")
    axes[2].set_title("C")
    axes[2].set_ylabel("Value")

    axes[3].hist(np.random.randn(100), bins=15, id="hist-tall")
    axes[3].set_title("D")
    axes[3].set_xlabel("Value")
    axes[3].set_ylabel("Count")

    png_path = output_dir / "07_tall_format.png"
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
    Demo 07: Tall format figure (4x1)

    Features:
        - Tall aspect ratio (60x180mm)
        - Four vertical panels
        - Line, Scatter, Box, Histogram

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5057
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 07: Tall format (4x1)")
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
