#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_01_multi_type.py

"""
Demo 01: Multi-type per axis figure (2x2) - Line + Scatter + Fill

Port: 5051

Usage:
    ./demo_01_multi_type.py              # Flask backend (default)
    ./demo_01_multi_type.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5051


def create_figure(output_dir: Path) -> Path:
    """Create multi-type per axis figure: Line + Scatter + Fill."""
    from scipy import stats

    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    y1 = np.sin(x) + np.random.normal(0, 0.1, n_samples)
    groups = ["Control", "Treatment A", "Treatment B"]
    group_data = [np.random.normal(loc, 0.5, 50) for loc in [0, 1, 2]]

    STYLE = SCITEX_STYLE.copy()
    fig, axes = stx.plt.subplots(2, 2, **STYLE)

    # Panel A: Line + Scatter + Vertical fill regions
    axes[0, 0].plot(x, y1, "-", linewidth=1, label="Trend", id="trend-line")
    axes[0, 0].plot(
        x,
        y1,
        "o",
        markersize=2,
        alpha=0.5,
        label="Data points",
        id="data-points",
    )
    axes[0, 0].stx_fillv([2, 6], [4, 8], alpha=0.2, color="gray", id="regions")
    axes[0, 0].set_title("A) Line + Scatter + Fill Regions")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Signal [a.u.]")
    axes[0, 0].legend()

    # Panel B: Multiple lines + error bars
    y_mean = np.sin(x)
    y_err = 0.2 + 0.1 * np.random.rand(n_samples)
    axes[0, 1].plot(x, y_mean, "-", color="blue", label="Mean", id="mean-line")
    axes[0, 1].fill_between(
        x, y_mean - y_err, y_mean + y_err, alpha=0.3, id="uncertainty"
    )
    axes[0, 1].errorbar(
        x[::10],
        y_mean[::10],
        yerr=y_err[::10],
        fmt="o",
        markersize=4,
        capsize=2,
        label="Error bars",
        id="errorbars",
    )
    axes[0, 1].set_title("B) Line + Fill + Error Bars")
    axes[0, 1].set_xlabel("Time [s]")
    axes[0, 1].set_ylabel("Value [a.u.]")
    axes[0, 1].legend()

    # Panel C: Box + Strip + Scatter overlay
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = np.random.normal(0, 0.05, len(data))
        axes[1, 0].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.3,
            s=10,
            id=f"scatter-{label}",
        )
    axes[1, 0].stx_box(group_data, labels=groups, id="boxplot")
    axes[1, 0].set_title("C) Box + Scatter Overlay")
    axes[1, 0].set_xlabel("Group")
    axes[1, 0].set_ylabel("Value [a.u.]")

    # Panel D: Histogram + KDE overlay
    hist_data = np.concatenate(group_data)
    axes[1, 1].hist(
        hist_data,
        bins=25,
        density=True,
        alpha=0.6,
        label="Histogram",
        id="hist",
    )
    kde_x = np.linspace(hist_data.min() - 1, hist_data.max() + 1, 200)
    kde = stats.gaussian_kde(hist_data)
    axes[1, 1].plot(
        kde_x, kde(kde_x), "-", linewidth=2, label="KDE", id="kde-overlay"
    )
    for i, (data, label) in enumerate(zip(group_data, groups)):
        axes[1, 1].axvline(
            np.mean(data),
            linestyle="--",
            alpha=0.7,
            label=f"{label} mean",
            id=f"mean-{label}",
        )
    axes[1, 1].set_title("D) Histogram + KDE + Mean Lines")
    axes[1, 1].set_xlabel("Value [a.u.]")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend(fontsize=5)

    png_path = output_dir / "01_multi_type_per_axis.png"
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
    Demo 01: Multi-type per axis figure (2x2)

    Features:
        - Line + Scatter + Fill Regions
        - Line + Fill + Error Bars
        - Box + Scatter Overlay
        - Histogram + KDE + Mean Lines

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5051
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 01: Multi-type per axis (2x2)")
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
