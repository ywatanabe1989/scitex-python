#!/usr/bin/env python3
import pytest
pytest.importorskip("zarr")
# -*- coding: utf-8 -*-
"""Generate various test figures for Flask GUI editor testing.

Uses scitex.dev.plt plotters with consistent API.

Test cases:
1. Single axis with multiple traces
2. Multiple axes (2x2 grid)
3. Different plot types (bar, scatter, heatmap, etc.)
"""

import os
import sys
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')

import scitex
import scitex.plt as plt
from scitex.dev.plt import (
    plot_stx_line,
    plot_stx_scatter,
    plot_stx_bar,
    plot_stx_heatmap,
    plot_stx_mean_ci,
    plot_stx_violin,
    plot_mpl_hist,
    plot_mpl_errorbar,
    PLOTTERS_STX,
)


class SimpleRNG:
    """Simple RNG wrapper for dev.plt plotters."""
    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_single_axis_multiple_traces():
    """Test: Single axis with multiple overlaid traces."""
    rng = SimpleRNG(42)
    fig, ax = plt.subplots()

    x = np.linspace(0, 2*np.pi, 100)

    # Multiple traces
    ax.plot(x, np.sin(x), id="sine", label="sin(x)")
    ax.plot(x, np.cos(x), id="cosine", label="cos(x)")
    ax.plot(x, np.sin(2*x), id="sine2x", label="sin(2x)", linestyle="--")
    ax.scatter(x[::10], np.sin(x[::10]) + 0.5, id="scatter_pts", label="points", s=20)

    ax.set_xlabel("X [rad]")
    ax.set_ylabel("Amplitude")
    ax.set_title("Multiple Traces Test")
    ax.legend(loc="upper right")

    output_dir = tempfile.mkdtemp(prefix="test_multi_trace_")
    save_path = os.path.join(output_dir, "multi_trace_figure.png")
    scitex.io.save(fig, save_path)
    print(f"  Saved to: {output_dir}")
    return output_dir


def test_grid_2x2():
    """Test: 2x2 grid with different plot types."""
    rng = SimpleRNG(42)

    # Create 2x2 grid
    fig, axes = plt.subplots(nrows=2, ncols=2)

    x = np.linspace(0, 10, 100)

    # Access axes by iteration (returns AxisWrapper objects)
    ax_iter = iter(axes)
    ax00 = next(ax_iter)
    ax01 = next(ax_iter)
    ax10 = next(ax_iter)
    ax11 = next(ax_iter)

    # Top-left: Line plot
    ax00.plot(x, np.sin(x), id="sine", label="sin(x)")
    ax00.plot(x, np.cos(x), id="cosine", label="cos(x)")
    ax00.set_title("Sine & Cosine")
    ax00.legend()

    # Top-right: Scatter with trend
    x_scatter = rng.uniform(0, 10, 50)
    y_scatter = x_scatter + rng.normal(0, 2, 50)
    ax01.scatter(x_scatter, y_scatter, id="scatter", alpha=0.6)
    ax01.plot([0, 10], [0, 10], '--', id="trend", color='red')
    ax01.set_title("Scatter with Trend")

    # Bottom-left: Bar chart
    categories = ['A', 'B', 'C', 'D', 'E']
    values = rng.integers(1, 10, size=5)
    ax10.bar(categories, values, id="bars")
    ax10.set_title("Bar Chart")

    # Bottom-right: Histogram
    data = rng.normal(0, 1, 500)
    ax11.hist(data, bins=25, id="hist", alpha=0.7)
    ax11.set_title("Histogram")

    fig.tight_layout()

    output_dir = tempfile.mkdtemp(prefix="test_grid_2x2_")
    save_path = os.path.join(output_dir, "grid_2x2_figure.png")
    scitex.io.save(fig, save_path)
    print(f"  Saved to: {output_dir}")
    return output_dir


def test_grid_2x3():
    """Test: 2x3 grid with various plot types."""
    rng = SimpleRNG(42)

    fig, axes = plt.subplots(nrows=2, ncols=3)

    x = np.linspace(0, 10, 100)
    ax_list = list(axes)

    # Row 1
    ax_list[0].plot(x, np.sin(x), id="line")
    ax_list[0].set_title("Line Plot")

    x_s = rng.uniform(0, 10, 30)
    ax_list[1].scatter(x_s, rng.normal(0, 2, 30), id="scatter")
    ax_list[1].set_title("Scatter")

    ax_list[2].bar(['A', 'B', 'C'], rng.integers(1, 10, 3), id="bar")
    ax_list[2].set_title("Bar")

    # Row 2
    ax_list[3].hist(rng.normal(0, 1, 200), bins=20, id="hist")
    ax_list[3].set_title("Histogram")

    ax_list[4].errorbar([1, 2, 3], [2, 4, 3], yerr=[0.5, 0.8, 0.6], id="errorbar", fmt='o-')
    ax_list[4].set_title("Error Bar")

    ax_list[5].fill_between(x, np.sin(x)-0.2, np.sin(x)+0.2, id="fill", alpha=0.3)
    ax_list[5].plot(x, np.sin(x), id="fill_line")
    ax_list[5].set_title("Fill Between")

    fig.tight_layout()

    output_dir = tempfile.mkdtemp(prefix="test_grid_2x3_")
    save_path = os.path.join(output_dir, "grid_2x3_figure.png")
    scitex.io.save(fig, save_path)
    print(f"  Saved to: {output_dir}")
    return output_dir


def test_vertical_layout():
    """Test: 3x1 vertical layout (common for neuroscience data)."""
    rng = SimpleRNG(42)

    fig, axes = plt.subplots(nrows=3, ncols=1)
    axes_flat = list(axes.flat)

    t = np.linspace(0, 2, 1000)

    # EEG-like signal
    eeg = np.sin(10 * 2 * np.pi * t) + 0.5 * np.sin(30 * 2 * np.pi * t) + 0.2 * rng.normal(size=len(t))
    axes_flat[0].plot(t, eeg, id="eeg_signal", linewidth=0.5)
    axes_flat[0].set_title("EEG Channel 1")
    axes_flat[0].set_ylabel("μV")

    # EMG-like signal
    emg = np.sin(100 * 2 * np.pi * t) * (1 + 0.5 * np.sin(2 * np.pi * t)) + 0.3 * rng.normal(size=len(t))
    axes_flat[1].plot(t, emg, id="emg_signal", linewidth=0.5)
    axes_flat[1].set_title("EMG")
    axes_flat[1].set_ylabel("mV")

    # Event markers
    events = np.zeros(len(t))
    event_times = [0.5, 1.0, 1.5]
    for et in event_times:
        idx = int(et * 500)
        events[idx:idx+10] = 1
    axes_flat[2].plot(t, events, id="events", drawstyle='steps-post')
    axes_flat[2].set_title("Events")
    axes_flat[2].set_xlabel("Time [s]")
    axes_flat[2].set_ylabel("Event")
    axes_flat[2].set_ylim(-0.2, 1.5)

    fig.tight_layout()

    output_dir = tempfile.mkdtemp(prefix="test_vertical_")
    save_path = os.path.join(output_dir, "vertical_figure.png")
    scitex.io.save(fig, save_path)
    print(f"  Saved to: {output_dir}")
    return output_dir


def test_all_stx_plotters():
    """Test: Generate figures for ALL scitex.dev.plt plotters."""
    rng = SimpleRNG(42)
    output_dir = tempfile.mkdtemp(prefix="test_all_plotters_")

    for name, plotter in PLOTTERS_STX.items():
        try:
            fig, ax = plotter(plt, rng)
            save_path = os.path.join(output_dir, f"{name}.png")
            scitex.io.save(fig, save_path)
            print(f"    [OK] {name}")
        except Exception as e:
            print(f"    [FAIL] {name}: {e}")

    print(f"  Saved all to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    print("Generating test figures for Flask GUI editor...\n")

    dirs = []

    print("1. Single axis with multiple traces")
    dirs.append(test_single_axis_multiple_traces())

    print("\n2. 2x2 grid")
    dirs.append(test_grid_2x2())

    print("\n3. 2x3 grid with various plot types")
    dirs.append(test_grid_2x3())

    print("\n4. Vertical layout (3x1)")
    dirs.append(test_vertical_layout())

    print("\n5. All STX plotters (one figure each)")
    dirs.append(test_all_stx_plotters())

    print("\n" + "="*50)
    print("Test figures generated. Use these paths to test the Flask editor:")
    for d in dirs:
        print(f"  {d}")
    print("\nRun Flask editor with:")
    print("  python -m scitex.canvas.editor --port 5555 <path>")
