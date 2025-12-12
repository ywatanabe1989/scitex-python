#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: /home/ywatanabe/proj/scitex-code/examples/fig/demo_vis_editor.py
"""
Demo: stx.fig.edit() - Interactive Figure Editor

Launch editor for any saved figure:
    stx.fig.edit("path/to/figure.json")  # Auto-detect best backend
    stx.fig.edit("path/to/figure.png")   # Auto-finds JSON sibling

Backends (auto-detected): web > dearpygui > qt > tkinter > mpl

See demo_gui_editors.py for detailed examples of each backend.
"""

from pathlib import Path
import scitex as stx
from scitex.dev.plt import plot_multi_line


def create_sample_figure(output_dir: Path, plt, rng) -> Path:
    """Create a sample figure for editing using scitex.dev.plt plotter."""
    fig, ax = plot_multi_line(plt, rng)

    output_path = output_dir / "editable_figure.png"
    stx.io.save(fig, output_path)
    fig.close()

    return output_dir / "editable_figure.json"


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Create sample figure and show editor usage."""
    out = Path(CONFIG.SDIR_OUT)
    rng = rng_manager("fig_editor_demo")
    json_path = create_sample_figure(out, plt, rng)

    print("\n" + "=" * 60)
    print("Interactive Figure Editor Demo")
    print("=" * 60)
    print(f"\nFigure saved to: {json_path}")
    print("\nTo launch the editor, run:")
    print(f"  stx.fig.edit('{json_path}')")
    print("\nOr with a specific backend:")
    print(f"  stx.fig.edit('{json_path}', backend='web')")
    print(f"  stx.fig.edit('{json_path}', backend='dearpygui')")
    print(f"  stx.fig.edit('{json_path}', backend='qt')")
    print(f"  stx.fig.edit('{json_path}', backend='tkinter')")
    print(f"  stx.fig.edit('{json_path}', backend='mpl')")
    print("=" * 60)

    # Uncomment to launch the editor:
    # stx.fig.edit(str(json_path))


if __name__ == "__main__":
    main()

# EOF
