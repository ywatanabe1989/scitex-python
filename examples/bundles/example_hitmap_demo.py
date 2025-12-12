#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-12 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/bundles/example_hitmap_demo.py

"""
Demonstrates hitmap generation with complex, multi-element figures.

This script creates figures with multiple overlapping elements to test
hitmap-based element selection:
- Multiple lines with different styles
- Scatter points with varying sizes
- Bar charts with multiple series
- Fill between areas
- Annotations and markers
"""

import numpy as np
import scitex as stx
import scitex.io as sio


def plot_multi_line(plt, rng):
    """Create plot with multiple overlapping lines."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 4 * np.pi, 200)

    # Multiple lines with different styles
    ax.plot(x, np.sin(x), '-', linewidth=2, label='sin(x)', color='#1f77b4')
    ax.plot(x, np.cos(x), '--', linewidth=2, label='cos(x)', color='#ff7f0e')
    ax.plot(x, np.sin(x) * np.cos(x), ':', linewidth=3, label='sin·cos', color='#2ca02c')
    ax.plot(x, 0.5 * np.sin(2*x), '-.', linewidth=2, label='0.5·sin(2x)', color='#d62728')

    # Add scatter points on top
    sample_idx = np.arange(0, len(x), 20)
    ax.scatter(x[sample_idx], np.sin(x[sample_idx]), s=50, c='#1f77b4', marker='o', zorder=5)
    ax.scatter(x[sample_idx], np.cos(x[sample_idx]), s=50, c='#ff7f0e', marker='s', zorder=5)

    ax.set_xyt('x (radians)', 'y', 'Multi-Line Plot with Scatter Overlay')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    return fig, ax


def plot_bar_grouped(plt, rng):
    """Create grouped bar chart with multiple series."""
    fig, ax = plt.subplots(figsize=(8, 6))

    categories = ['A', 'B', 'C', 'D', 'E']
    x = np.arange(len(categories))
    width = 0.25

    # Three series of bars
    values1 = rng.uniform(10, 30, len(categories))
    values2 = rng.uniform(15, 35, len(categories))
    values3 = rng.uniform(5, 25, len(categories))

    ax.bar(x - width, values1, width, label='Series 1', color='#1f77b4')
    ax.bar(x, values2, width, label='Series 2', color='#ff7f0e')
    ax.bar(x + width, values3, width, label='Series 3', color='#2ca02c')

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xyt('Category', 'Value', 'Grouped Bar Chart')
    ax.legend()
    return fig, ax


def plot_scatter_sizes(plt, rng):
    """Create scatter plot with varying sizes and colors."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n_points = 50
    x = rng.uniform(0, 10, n_points)
    y = rng.uniform(0, 10, n_points)
    sizes = rng.uniform(20, 200, n_points)
    colors = rng.uniform(0, 1, n_points)

    scatter = ax.scatter(x, y, s=sizes, c=colors, cmap='viridis', alpha=0.7, edgecolors='white', linewidths=1)
    fig.colorbar(scatter, ax=ax, label='Value')

    ax.set_xyt('X', 'Y', 'Scatter with Size and Color Encoding')
    return fig, ax


def plot_fill_between(plt, rng):
    """Create plot with fill between areas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)

    # Multiple fill between areas
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = 0.5 * np.sin(2*x)

    ax.fill_between(x, y1, y2, alpha=0.3, label='sin-cos region', color='#1f77b4')
    ax.fill_between(x, y2, y3, alpha=0.3, label='cos-sin2x region', color='#ff7f0e')
    ax.fill_between(x, y3, -1, alpha=0.3, label='sin2x-bottom region', color='#2ca02c')

    # Overlay lines
    ax.plot(x, y1, '-', linewidth=2, color='#1f77b4')
    ax.plot(x, y2, '-', linewidth=2, color='#ff7f0e')
    ax.plot(x, y3, '-', linewidth=2, color='#2ca02c')

    ax.set_xyt('x', 'y', 'Fill Between Plot')
    ax.legend()
    return fig, ax


def plot_multi_panel(plt, rng):
    """Create multi-panel figure with subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    # Panel 1: Line plot
    x = np.linspace(0, 2*np.pi, 100)
    axes[0].plot(x, np.sin(x), label='sin')
    axes[0].plot(x, np.cos(x), label='cos')
    axes[0].set_xyt('x', 'y', 'Trigonometric Functions')
    axes[0].legend()

    # Panel 2: Bar chart
    axes[1].bar(['A', 'B', 'C', 'D'], rng.uniform(5, 20, 4))
    axes[1].set_xyt('Category', 'Value', 'Bar Chart')

    # Panel 3: Scatter
    axes[2].scatter(rng.uniform(0, 10, 30), rng.uniform(0, 10, 30), s=50)
    axes[2].set_xyt('X', 'Y', 'Scatter')

    # Panel 4: Histogram
    data = rng.standard_normal(1000)
    axes[3].hist(data, bins=30, edgecolor='white')
    axes[3].set_xyt('Value', 'Count', 'Histogram')

    fig.tight_layout()
    return fig, axes[0]


def plot_complex_annotations(plt, rng):
    """Create plot with annotations and markers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x) * np.exp(-x/10)

    ax.plot(x, y, '-', linewidth=2, label='Damped sine', color='#1f77b4')

    # Add vertical/horizontal lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=np.pi, color='red', linestyle=':', linewidth=2, label='x=π')

    # Add markers at peaks
    peaks_x = [np.pi/2, 5*np.pi/2]
    peaks_y = [np.sin(px) * np.exp(-px/10) for px in peaks_x]
    ax.scatter(peaks_x, peaks_y, s=100, c='red', marker='*', zorder=10, label='Peaks')

    ax.set_xyt('x', 'y', 'Damped Sine with Annotations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


@stx.session
def main(
    CONFIG=stx.INJECTED,
    plt=stx.INJECTED,
    COLORS=stx.INJECTED,
    rng_manager=stx.INJECTED,
):
    """Create complex demo figures for hitmap testing."""
    sdir = CONFIG["SDIR_RUN"]
    rng = rng_manager("hitmap_demo")

    print("Creating complex demo figures for hitmap testing...")

    # 1. Multi-line plot
    fig, ax = plot_multi_line(plt, rng)
    sio.save(fig, sdir / "multi_line.pltz.d", dpi=150)
    plt.close(fig)
    print("  Created: multi_line.pltz.d")

    # 2. Grouped bar chart
    fig, ax = plot_bar_grouped(plt, rng)
    sio.save(fig, sdir / "bar_grouped.pltz.d", dpi=150)
    plt.close(fig)
    print("  Created: bar_grouped.pltz.d")

    # 3. Scatter with sizes
    fig, ax = plot_scatter_sizes(plt, rng)
    sio.save(fig, sdir / "scatter_sizes.pltz.d", dpi=150)
    plt.close(fig)
    print("  Created: scatter_sizes.pltz.d")

    # 4. Fill between
    fig, ax = plot_fill_between(plt, rng)
    sio.save(fig, sdir / "fill_between.pltz.d", dpi=150)
    plt.close(fig)
    print("  Created: fill_between.pltz.d")

    # 5. Multi-panel
    fig, ax = plot_multi_panel(plt, rng)
    sio.save(fig, sdir / "multi_panel.pltz.d", dpi=150)
    plt.close(fig)
    print("  Created: multi_panel.pltz.d")

    # 6. Complex annotations
    fig, ax = plot_complex_annotations(plt, rng)
    sio.save(fig, sdir / "annotations.pltz.d", dpi=150)
    plt.close(fig)
    print("  Created: annotations.pltz.d")

    print(f"\nAll bundles saved to: {sdir}")
    print("Each bundle contains: plot.png, plot_hitmap.png, plot_hitmap.svg, overview.png")

    return 0


if __name__ == "__main__":
    main()

# EOF
