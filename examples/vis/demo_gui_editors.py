#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: /home/ywatanabe/proj/scitex-code/examples/vis/demo_gui_editors.py
"""
Demo: stx.vis.edit() - Interactive Figure Editors

Usage:
    ./demo_gui_editors.py                    # Auto-detect best backend
    ./demo_gui_editors.py --backend flask    # Browser-based
    ./demo_gui_editors.py --backend dearpygui # GPU-accelerated
    ./demo_gui_editors.py --backend qt       # Qt desktop
    ./demo_gui_editors.py --backend tkinter  # Built-in Python
    ./demo_gui_editors.py --backend mpl      # Minimal matplotlib

Install GUI backends: pip install scitex[gui]
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx

# Type alias for backend choices
Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]


def create_sample_figure(output_dir: Path) -> Path:
    """Create a sample figure for editing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = stx.plt.subplots()
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, np.cos(x), label="cos(x)")
    ax.set_xyt(x="Phase (rad)", y="Amplitude", t="Trigonometric Functions")
    ax.legend(frameon=False)

    png_path = output_dir / "figure.png"
    stx.io.save(fig, png_path)
    fig.close()

    return png_path.with_suffix(".json")


@stx.session
def main(
    backend: Backend = "auto",
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Launch GUI editor with specified backend."""
    out = Path(CONFIG.SDIR_OUT)
    json_path = create_sample_figure(out)
    stx.vis.edit(str(json_path), backend=backend)


if __name__ == "__main__":
    main()

# EOF
