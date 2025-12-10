#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_08_single_panel.py

"""
Demo 08: Single panel figure (1x1) - Multiple elements

Port: 5058

Usage:
    ./demo_08_single_panel.py              # Flask backend (default)
    ./demo_08_single_panel.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5058


def create_figure(output_dir: Path) -> Path:
    """Create single panel figure with multiple elements."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    STYLE = SCITEX_STYLE.copy()
    STYLE["fig_size_mm"] = (100, 80)
    fig, ax = stx.plt.subplots(1, 1, **STYLE)

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.exp(-x / 5)

    ax.plot(x, y1, "-", label="sin(x)", id="line-sin")
    ax.plot(x, y2, "--", label="cos(x)", id="line-cos")
    ax.plot(x, y3, "-.", label="damped", id="line-damped")
    ax.fill_between(x, y1 - 0.2, y1 + 0.2, alpha=0.2, id="fill-sin")
    ax.stx_fillv([2], [4], alpha=0.1, color="yellow", id="region")
    ax.scatter(x[::10], y1[::10], s=30, zorder=5, id="markers")

    ax.set_title("Single Panel - Multiple Elements")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.legend()

    png_path = output_dir / "08_single_panel.png"
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
    Demo 08: Single panel figure (1x1)

    Features:
        - Single panel (100x80mm)
        - Multiple overlaid elements
        - sin(x), cos(x), damped sine
        - Fill between, fill vertical, scatter markers

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5058
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 08: Single panel (1x1)")
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
