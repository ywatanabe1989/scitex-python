#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: /home/ywatanabe/proj/scitex-code/examples/vis/demo_vis_editor.py
"""
Demo: stx.vis.edit() - Interactive Figure Editor

Launch editor for any saved figure:
    stx.vis.edit("path/to/figure.json")  # Auto-detect best backend
    stx.vis.edit("path/to/figure.png")   # Auto-finds JSON sibling

Backends (auto-detected): web > dearpygui > qt > tkinter > mpl

See demo_gui_editors.py for detailed examples of each backend.
"""

import numpy as np
from pathlib import Path
import scitex as stx


def create_sample_figure(output_dir: Path) -> Path:
    """Create a sample figure for editing."""
    fig, ax = stx.plt.subplots()
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, np.cos(x), label="cos(x)")
    ax.set_xyt(x="Time [s]", y="Amplitude", t="Trigonometric Functions")
    ax.legend(frameon=False)

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
    json_path = create_sample_figure(out)

    print("\n" + "=" * 60)
    print("Interactive Figure Editor Demo")
    print("=" * 60)
    print(f"\nFigure saved to: {json_path}")
    print("\nTo launch the editor, run:")
    print(f"  stx.vis.edit('{json_path}')")
    print("\nOr with a specific backend:")
    print(f"  stx.vis.edit('{json_path}', backend='web')")
    print(f"  stx.vis.edit('{json_path}', backend='dearpygui')")
    print(f"  stx.vis.edit('{json_path}', backend='qt')")
    print(f"  stx.vis.edit('{json_path}', backend='tkinter')")
    print(f"  stx.vis.edit('{json_path}', backend='mpl')")
    print("=" * 60)

    # Uncomment to launch the editor:
    # stx.vis.edit(str(json_path))


if __name__ == "__main__":
    main()

# EOF
