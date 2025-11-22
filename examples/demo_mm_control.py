#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_mm_control.py

"""
Demonstration of millimeter-based figure control in scitex.plt.

This script demonstrates how to create publication-quality figures with
precise millimeter control over dimensions, margins, and styling.
Perfect for journal requirements (Nature, Science, etc.).
"""

import numpy as np
from scitex.plt.utils import create_figure_ax_mm, mm_to_pt


def demo_basic_mm_control():
    """Basic example: Create a simple plot with mm control."""
    print("\n" + "=" * 60)
    print("Demo 1: Basic mm-controlled figure")
    print("=" * 60)

    # Define Nature-like style specifications
    nature_style = {
        "axis_thickness_mm": 0.2,  # Axis spine thickness
        "trace_thickness_mm": 0.12,  # Plot line thickness
        "tick_length_mm": 0.8,  # Tick mark length
        "tick_thickness_mm": 0.2,  # Tick mark thickness
        "axis_font_size_pt": 8,  # Axis label font size
        "tick_font_size_pt": 7,  # Tick label font size
    }

    # Create figure with exact dimensions: 35mm wide, 24.5mm tall
    fig, ax = create_figure_ax_mm(
        fig_width_mm=35,
        fig_height_mm=24.5,
        dpi=300,
        left_margin_mm=4.0,
        right_margin_mm=2.0,
        bottom_margin_mm=4.0,
        top_margin_mm=2.0,
        style=nature_style,
    )

    # Generate sample data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Plot with mm-controlled line width
    trace_lw = mm_to_pt(nature_style["trace_thickness_mm"])
    ax.plot(x, y, color="tab:blue", lw=trace_lw, label="sin(x)")

    # Add labels and title
    ax.set_xlabel("X axis (rad)")
    ax.set_ylabel("Y axis")
    ax.set_title("Basic mm Control Demo")
    ax.legend(fontsize=7)

    # Save figure
    output_path = "/tmp/demo_basic_mm_control.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✅ Figure saved to: {output_path}")
    print(f"   Physical size: 35.0 mm × 24.5 mm")
    print(f"   Pixel size: {int(35/25.4*300)} × {int(24.5/25.4*300)} px at 300 DPI")


def demo_multiple_traces():
    """Advanced example: Multiple traces with different styles."""
    print("\n" + "=" * 60)
    print("Demo 2: Multiple traces with mm control")
    print("=" * 60)

    style = {
        "axis_thickness_mm": 0.2,
        "trace_thickness_mm": 0.15,  # Slightly thicker for visibility
        "tick_length_mm": 1.0,  # Longer ticks
        "tick_thickness_mm": 0.2,
        "axis_font_size_pt": 9,
        "tick_font_size_pt": 8,
    }

    # Create a wider figure
    fig, ax = create_figure_ax_mm(
        fig_width_mm=50,
        fig_height_mm=35,
        dpi=300,
        left_margin_mm=5.0,
        right_margin_mm=2.0,
        bottom_margin_mm=5.0,
        top_margin_mm=2.0,
        style=style,
    )

    # Generate multiple traces
    x = np.linspace(0, 4 * np.pi, 200)
    trace_lw = mm_to_pt(style["trace_thickness_mm"])

    functions = [
        ("sin(x)", np.sin(x), "tab:blue"),
        ("cos(x)", np.cos(x), "tab:orange"),
        ("sin(x)/2", np.sin(x) / 2, "tab:green"),
    ]

    for label, y, color in functions:
        ax.plot(x, y, lw=trace_lw, label=label, color=color)

    # Customize
    ax.set_xlabel("X axis (rad)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Multiple Traces Demo")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # Save
    output_path = "/tmp/demo_multiple_traces_mm.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✅ Figure saved to: {output_path}")
    print(f"   Physical size: 50.0 mm × 35.0 mm")


def demo_square_panel():
    """Example: Square panel (common in Nature figures)."""
    print("\n" + "=" * 60)
    print("Demo 3: Square panel (Nature-style)")
    print("=" * 60)

    style = {
        "axis_thickness_mm": 0.2,
        "trace_thickness_mm": 0.12,
        "tick_length_mm": 0.8,
        "tick_thickness_mm": 0.2,
        "axis_font_size_pt": 8,
        "tick_font_size_pt": 7,
    }

    # Create square figure: 30mm × 30mm
    fig, ax = create_figure_ax_mm(
        fig_width_mm=30,
        fig_height_mm=30,
        dpi=300,
        left_margin_mm=4.0,
        right_margin_mm=2.0,
        bottom_margin_mm=4.0,
        top_margin_mm=2.0,
        style=style,
    )

    # Create a 2D heatmap-style plot
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-(X**2 + Y**2) / 2)

    # Plot
    im = ax.contourf(X, Y, Z, levels=20, cmap="viridis")
    trace_lw = mm_to_pt(style["trace_thickness_mm"])
    ax.contour(X, Y, Z, levels=5, colors="white", linewidths=trace_lw, alpha=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Square Panel Demo")
    ax.set_aspect("equal")

    # Save
    output_path = "/tmp/demo_square_panel_mm.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✅ Figure saved to: {output_path}")
    print(f"   Physical size: 30.0 mm × 30.0 mm (square)")


def demo_custom_aspect_ratio():
    """Example: Custom aspect ratio (70% height)."""
    print("\n" + "=" * 60)
    print("Demo 4: Custom aspect ratio (70% height)")
    print("=" * 60)

    style = {
        "axis_thickness_mm": 0.2,
        "trace_thickness_mm": 0.12,
        "tick_length_mm": 0.8,
        "tick_thickness_mm": 0.2,
        "axis_font_size_pt": 8,
        "tick_font_size_pt": 7,
    }

    # Create figure with 70% aspect ratio
    width_mm = 40
    height_mm = width_mm * 0.7

    fig, ax = create_figure_ax_mm(
        fig_width_mm=width_mm,
        fig_height_mm=height_mm,
        dpi=300,
        left_margin_mm=5.0,
        right_margin_mm=2.0,
        bottom_margin_mm=4.0,
        top_margin_mm=2.0,
        style=style,
    )

    # Create bar plot
    categories = ["A", "B", "C", "D", "E"]
    values = [23, 45, 56, 78, 32]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    bars = ax.bar(categories, values, color=colors, alpha=0.7)

    # Customize bars
    for bar in bars:
        bar.set_edgecolor("black")
        bar.set_linewidth(mm_to_pt(0.15))

    ax.set_xlabel("Category")
    ax.set_ylabel("Value")
    ax.set_title("Custom Aspect Ratio (70%)")
    ax.set_ylim(0, 100)

    # Save
    output_path = "/tmp/demo_custom_aspect_mm.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"✅ Figure saved to: {output_path}")
    print(f"   Physical size: {width_mm:.1f} mm × {height_mm:.1f} mm")
    print(f"   Aspect ratio: {height_mm/width_mm:.2%}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("SciTeX mm-Control Demonstration")
    print("=" * 60)
    print("\nThis script demonstrates precise millimeter control for")
    print("publication-quality figures using scitex.plt.")

    # Run all demos
    demo_basic_mm_control()
    demo_multiple_traces()
    demo_square_panel()
    demo_custom_aspect_ratio()

    # Summary
    print("\n" + "=" * 60)
    print("All demonstrations completed! ✨")
    print("=" * 60)
    print("\nKey features demonstrated:")
    print("  ✅ Precise mm control for figure dimensions")
    print("  ✅ mm-based margins and axis positioning")
    print("  ✅ mm-based tick lengths and line widths")
    print("  ✅ Nature/Science journal-compliant styling")
    print("  ✅ Custom aspect ratios")
    print("\nAll figures saved to /tmp/")
    print("\nNext steps:")
    print("  - Use these functions in your VIS rendering pipeline")
    print("  - Pass JSON configs with mm specifications")
    print("  - Adjust title/labels by re-rendering with updated JSON")
    print()


if __name__ == "__main__":
    main()

# EOF
