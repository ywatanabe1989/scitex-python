#!/usr/bin/env python3
"""Plotting session script template."""

TEMPLATE = {
    "name": "Plotting Session Script",
    "description": "@stx.session script optimized for figure generation with auto CSV export",
    "filename": "plot_script.py",
    "usage": """
Usage:
  python plot_script.py
  python plot_script.py --help

Note: stx.io.save(fig, "plot.png") automatically exports plot.csv with plotted data
""",
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "{timestamp}"
# File: {filepath}

"""
{docstring}

Note: stx.io.save(fig, "plot.png") automatically exports plotted data to plot.csv
"""

import numpy as np
import scitex as stx


@stx.session
def main(
    n_points=100,
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Generate publication-ready figures with automatic data export."""
    # Generate data
    x = np.linspace(0, 2 * np.pi, n_points)
    y = np.sin(x)

    # Create figure using stx.plt (records all plotting calls)
    fig, ax = plt.subplots()

    # Use plot_line for automatic CSV export
    ax.plot_line(x, y, color=COLORS.blue, label="sin(x)")
    ax.set_xyt("X (radians)", "Y", "Sine Wave")
    ax.legend_()

    # Save figure (auto-exports CSV of plotted data)
    stx.io.save(fig, "figure.png", symlink_to="./data")

    logger.info(f"Figure saved to: {{CONFIG.SDIR_OUT}}/figure.png")
    logger.info(f"Data exported to: {{CONFIG.SDIR_OUT}}/figure.csv")

    return 0


if __name__ == "__main__":
    main()

# EOF
''',
}

# EOF
