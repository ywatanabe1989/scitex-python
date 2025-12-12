#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: /home/ywatanabe/proj/scitex-code/examples/fig/demo_gui_editors.py
"""
Demo: stx.fig.edit() - Interactive Figure Editors

Usage:
    ./demo_gui_editors.py                    # Auto-detect best backend
    ./demo_gui_editors.py --backend flask    # Browser-based
    ./demo_gui_editors.py --backend dearpygui # GPU-accelerated
    ./demo_gui_editors.py --backend qt       # Qt desktop
    ./demo_gui_editors.py --backend tkinter  # Built-in Python
    ./demo_gui_editors.py --backend mpl      # Minimal matplotlib

Install GUI backends: pip install scitex[gui]
"""

from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.dev.plt import plot_multi_line

# Type alias for backend choices
Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]


def create_sample_figure(output_dir: Path, plt, rng) -> Path:
    """Create a sample figure for editing using scitex.dev.plt plotter."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plot_multi_line(plt, rng)

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
    rng = rng_manager("gui_editors_demo")
    json_path = create_sample_figure(out, plt, rng)
    stx.fig.edit(str(json_path), backend=backend)


if __name__ == "__main__":
    main()

# EOF
