#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: examples/vis/gui_editors/demo_02_scientific.py

"""
Demo 02: Scientific figure (2x3) - Time series + Statistics + Correlation

Port: 5052

Usage:
    ./demo_02_scientific.py              # Flask backend (default)
    ./demo_02_scientific.py --backend qt # Qt backend
"""

import numpy as np
from pathlib import Path
from typing import Literal
import scitex as stx
from scitex.plt.styles.presets import SCITEX_STYLE

Backend = Literal["auto", "flask", "dearpygui", "qt", "tkinter", "mpl"]
PORT = 5052


def create_figure(output_dir: Path) -> Path:
    """Create scientific figure: Time series + Statistics + Correlation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    groups = ["Control", "Treatment A", "Treatment B"]
    group_data = [np.random.normal(loc, 0.5, 50) for loc in [0, 1, 2]]

    STYLE = SCITEX_STYLE.copy()
    fig, axes = stx.plt.subplots(2, 3, **STYLE)

    # Panel A: Multi-channel time series with annotations
    time = np.linspace(0, 5, 500)
    ch1 = np.sin(2 * np.pi * time) + 0.3 * np.random.randn(500)
    ch2 = np.sin(2 * np.pi * time + np.pi / 4) + 0.3 * np.random.randn(500) + 3
    ch3 = np.sin(2 * np.pi * time + np.pi / 2) + 0.3 * np.random.randn(500) + 6

    axes[0, 0].plot(time, ch1, label="Ch1", id="ch1")
    axes[0, 0].plot(time, ch2, label="Ch2", id="ch2")
    axes[0, 0].plot(time, ch3, label="Ch3", id="ch3")
    axes[0, 0].stx_fillv(
        [1, 3], [2, 4], alpha=0.2, color="red", id="stim-period"
    )
    axes[0, 0].stx_rectangle(
        1,
        ch1.min() - 0.5,
        1,
        ch3.max() - ch1.min() + 1,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        id="stim-box",
    )
    axes[0, 0].set_title("A) Multi-channel Recording")
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Amplitude [mV]")
    axes[0, 0].legend(loc="upper right")

    # Panel B: Shaded line with individual traces
    data_2d = np.random.randn(20, 100)
    mean_trace = data_2d.mean(axis=0)
    std_trace = data_2d.std(axis=0)
    trace_x = np.arange(100)

    for i in range(min(5, len(data_2d))):
        axes[0, 1].plot(
            trace_x, data_2d[i], alpha=0.2, color="gray", id=f"trace-{i}"
        )
    axes[0, 1].stx_shaded_line(
        trace_x,
        mean_trace - std_trace,
        mean_trace,
        mean_trace + std_trace,
        id="mean-std",
    )
    axes[0, 1].set_title("B) Individual Traces + Mean +/- Std")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Value [a.u.]")

    # Panel C: Heatmap with contour overlay
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    zz = np.exp(-(xx**2 + yy**2) / 2) + 0.5 * np.exp(
        -((xx - 1) ** 2 + (yy - 1) ** 2) / 0.5
    )
    axes[0, 2].imshow(
        zz, extent=[-3, 3, -3, 3], origin="lower", cmap="viridis", id="heatmap"
    )
    axes[0, 2].contour(
        xx, yy, zz, levels=5, colors="white", linewidths=0.5, id="contours"
    )
    axes[0, 2].set_title("C) Heatmap + Contour Overlay")
    axes[0, 2].set_xlabel("X [a.u.]")
    axes[0, 2].set_ylabel("Y [a.u.]")

    # Panel D: Violin + Strip + Mean markers
    for i, (data, label) in enumerate(zip(group_data, groups)):
        jitter = np.random.normal(0, 0.08, len(data))
        axes[1, 0].scatter(
            np.full_like(data, i) + jitter,
            data,
            alpha=0.4,
            s=8,
            id=f"strip-{label}",
        )
    axes[1, 0].stx_violin(group_data, labels=groups, id="violin")
    for i, data in enumerate(group_data):
        axes[1, 0].plot(
            i,
            np.mean(data),
            "D",
            color="red",
            markersize=6,
            id=f"mean-marker-{i}",
        )
    axes[1, 0].set_title("D) Violin + Strip + Mean")
    axes[1, 0].set_xlabel("Group")
    axes[1, 0].set_ylabel("Value [a.u.]")

    # Panel E: Scatter + Regression + CI
    scatter_x = np.random.randn(50)
    scatter_y = 0.8 * scatter_x + 0.5 * np.random.randn(50)
    axes[1, 1].scatter(scatter_x, scatter_y, alpha=0.6, s=20, id="scatter")
    slope, intercept = np.polyfit(scatter_x, scatter_y, 1)
    reg_x = np.linspace(scatter_x.min(), scatter_x.max(), 100)
    reg_y = slope * reg_x + intercept
    axes[1, 1].plot(
        reg_x,
        reg_y,
        "-",
        color="red",
        linewidth=2,
        label=f"y={slope:.2f}x+{intercept:.2f}",
        id="regression",
    )
    ci = 0.3
    axes[1, 1].fill_between(
        reg_x, reg_y - ci, reg_y + ci, alpha=0.2, color="red", id="ci-band"
    )
    axes[1, 1].set_title("E) Scatter + Regression + CI")
    axes[1, 1].set_xlabel("X [a.u.]")
    axes[1, 1].set_ylabel("Y [a.u.]")
    axes[1, 1].legend()

    # Panel F: Bar + Error bars + Significance markers
    means = [np.mean(d) for d in group_data]
    stds = [np.std(d) for d in group_data]
    bar_x = np.arange(len(groups))
    axes[1, 2].bar(bar_x, means, yerr=stds, capsize=3, alpha=0.7, id="bars")
    y_max = max(means) + max(stds) + 0.3
    axes[1, 2].plot(
        [0, 0, 2, 2],
        [y_max, y_max + 0.1, y_max + 0.1, y_max],
        "k-",
        linewidth=1,
        id="sig-bracket",
    )
    axes[1, 2].text(
        1, y_max + 0.15, "***", ha="center", va="bottom", fontsize=10
    )
    axes[1, 2].set_xticks(bar_x)
    axes[1, 2].set_xticklabels(groups)
    axes[1, 2].set_title("F) Bar + Error + Significance")
    axes[1, 2].set_xlabel("Group")
    axes[1, 2].set_ylabel("Mean Value [a.u.]")

    png_path = output_dir / "02_scientific_figure.png"
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
    Demo 02: Scientific figure (2x3)

    Features:
        - Multi-channel Recording
        - Individual Traces + Mean +/- Std
        - Heatmap + Contour Overlay
        - Violin + Strip + Mean
        - Scatter + Regression + CI
        - Bar + Error + Significance

    Parameters
    ----------
    backend : str
        GUI backend: flask, dearpygui, qt, tkinter, mpl

    Port: 5052
    """
    out = Path(CONFIG.SDIR_OUT)

    logger.info("=" * 60)
    logger.info("Demo 02: Scientific figure (2x3)")
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
