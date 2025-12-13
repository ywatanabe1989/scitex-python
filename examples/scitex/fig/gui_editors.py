#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-13
# File: /home/ywatanabe/proj/scitex-code/examples/scitex/fig/gui_editors.py
"""
Demo: Interactive GUI Figure Editor

Launch interactive editor for figure editing:
    ./gui_editors.py                    # Auto-detect best backend
    ./gui_editors.py --backend flask    # Browser-based (recommended)
    ./gui_editors.py --backend qt       # Qt desktop
    ./gui_editors.py --backend tkinter  # Built-in Python
    ./gui_editors.py --backend mpl      # Minimal matplotlib

For programmatic (non-GUI) editing, see cui_editor.py

Install GUI backends: pip install scitex[gui]
"""

from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.dev.plt import PLOTTERS_STX

# Type alias for backend choices
Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]


def create_sample_figure(output_dir: Path, plt, rng) -> Path:
    """Create a sample figure as .figz.d bundle for editing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a single panel figure
    plotter = PLOTTERS_STX["stx_line"]
    fig, ax = plotter(plt, rng)

    # Save as pltz first (temporary)
    pltz_path = output_dir / "panel_A.pltz.d"
    stx.io.save(fig, pltz_path, dpi=150)
    plt.close(fig)

    # Create figz bundle with single panel
    figz_path = output_dir / "Figure1.figz.d"
    panels = {"A": str(pltz_path)}
    stx.fig.save_figz(panels, figz_path)

    return figz_path


@stx.session
def main(
    backend: Backend = "auto",
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Launch interactive GUI editor."""
    out = Path(CONFIG.SDIR_OUT)
    rng = rng_manager("gui_editors_demo")

    logger.info("=" * 60)
    logger.info("GUI Figure Editor Demo")
    logger.info("=" * 60)

    # Create sample figure as figz bundle
    logger.info("\nCreating sample figz bundle...")
    bundle_path = create_sample_figure(out, plt, rng)
    logger.success(f"Figz bundle created: {bundle_path}")

    # List bundle contents
    logger.info("\nFigz bundle contents:")
    for f in sorted(bundle_path.iterdir()):
        if f.is_dir():
            logger.info(f"  {f.name}/")
        else:
            logger.info(f"  {f.name}")

    # Launch interactive editor
    logger.info(f"\nLaunching editor with backend: {backend}")
    logger.info("(Close the editor window/browser to exit)")
    logger.info("=" * 60)

    # Launch the GUI editor
    stx.fig.edit(str(bundle_path), backend=backend)


if __name__ == "__main__":
    main()

# EOF
