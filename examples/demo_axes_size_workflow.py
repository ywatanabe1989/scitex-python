#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 12:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_axes_size_workflow.py

"""
Demo: New workflow for specifying AXES size (not figure size).

This demonstrates the solution to the user's requirements:
1. Specify axes box size directly (not figure size)
2. Simple usage with scitex.plt wrappers
3. Display vs Publication modes
4. Dimension viewer for debugging
"""

import numpy as np
from scitex.plt.utils import (
    compare_modes,
    create_axes_with_size_mm,
    print_dimension_info,
    view_dimensions,
)


def demo_basic_workflow():
    """
    Basic workflow: Specify axes size, get figure automatically.

    This is what you want for publication - specify the axes box size
    (30mm × 21mm), and the figure size is calculated automatically
    based on margins.
    """
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Workflow - Specify Axes Size")
    print("=" * 70)
    print("\n💡 Key idea: You specify AXES size, not FIGURE size")
    print("   The figure size is automatically calculated as:")
    print("   figure_size = axes_size + margins\n")

    # Create figure by specifying AXES size (30mm × 21mm)
    fig, ax = create_axes_with_size_mm(
        axes_width_mm=30,  # This is what you care about!
        axes_height_mm=21,  # This is what you care about!
        mode="publication",  # For exact mm control
    )

    # Plot data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    ax.plot(x, y, "b-", lw=1)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title("30mm × 21mm Axes Box")

    # Show dimension info (helps understand mm/inch/pixel/DPI)
    print_dimension_info(fig, ax)

    # Save for publication
    output = "/tmp/demo_publication_30x21mm.tiff"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"✅ Saved publication figure to: {output}")
    print(
        "   This file is exactly 30×21mm when printed or embedded in Word/LaTeX\n"
    )

    return fig, ax


def demo_display_vs_publication():
    """
    Display vs Publication modes.

    During development, you want LARGE figures for screen viewing.
    For publication, you want EXACT mm sizes.

    Solution: Use mode='display' or mode='publication'
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Display vs Publication Modes")
    print("=" * 70)
    print("\n💡 Problem: 30mm axes is tiny on screen, perfect for publication")
    print("   Solution: mode='display' scales 3x for screen viewing\n")

    # Same axes size (30×21mm), different modes
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    # Publication mode (exact 30×21mm)
    print("\n📄 PUBLICATION MODE (exact 30×21mm axes):")
    print("-" * 70)
    fig_pub, ax_pub = create_axes_with_size_mm(
        axes_width_mm=30, axes_height_mm=21, mode="publication"
    )
    ax_pub.plot(x, y, "b-", lw=1)
    ax_pub.set_xlabel("X")
    ax_pub.set_ylabel("Y")
    ax_pub.set_title("Publication: 30×21mm")
    print_dimension_info(fig_pub, ax_pub)

    output_pub = "/tmp/demo_publication_mode.png"
    fig_pub.savefig(output_pub, dpi=300, bbox_inches="tight")
    print(f"✅ Saved to: {output_pub}\n")

    # Display mode (scaled 3x = 90×63mm for screen)
    print("\n🖥️  DISPLAY MODE (same 30×21mm, scaled 3x for screen):")
    print("-" * 70)
    fig_disp, ax_disp = create_axes_with_size_mm(
        axes_width_mm=30, axes_height_mm=21, mode="display"  # 3x larger!
    )
    ax_disp.plot(x, y, "b-", lw=1)
    ax_disp.set_xlabel("X")
    ax_disp.set_ylabel("Y")
    ax_disp.set_title("Display: 30×21mm (3x scaled)")
    print_dimension_info(fig_disp, ax_disp)

    output_disp = "/tmp/demo_display_mode.png"
    fig_disp.savefig(output_disp, dpi=100)
    print(f"✅ Saved to: {output_disp}\n")

    print("📊 COMPARISON:")
    print(
        "   • Publication: Small on screen, perfect for paper (300 DPI, exact mm)"
    )
    print(
        "   • Display: Large on screen, for development (100 DPI, 3x scaled)\n"
    )


def demo_dimension_viewer():
    """
    Dimension viewer for debugging.

    When you're confused about mm/inch/pixel/DPI, use the dimension viewer
    to see exactly what's happening!
    """
    print("\n" + "=" * 70)
    print("DEMO 3: Dimension Viewer (Debug Tool)")
    print("=" * 70)
    print(
        "\n💡 Confused about mm/inch/pixel/DPI? Use the dimension viewer!\n"
    )

    # Create a figure
    fig, ax = create_axes_with_size_mm(
        axes_width_mm=30,
        axes_height_mm=21,
        mode="publication",
        margin_mm={"left": 6, "right": 3, "top": 3, "bottom": 6},
    )

    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), "b-", lw=1)
    ax.set_xlabel("X axis (with label)")
    ax.set_ylabel("Y axis")
    ax.set_title("Sample Plot")

    # View dimensions visually
    output_viewer = "/tmp/demo_dimension_viewer.png"
    fig_viewer = view_dimensions(
        fig, ax, show_rulers=True, show_grid=True, output_path=output_viewer
    )

    print(f"✅ Dimension viewer saved to: {output_viewer}")
    print("   Open this file to see:")
    print("   • Figure canvas (gray rectangle)")
    print("   • Axes box (blue rectangle)")
    print("   • Margins (labeled)")
    print("   • Rulers with mm markings")
    print("   • All dimensions annotated\n")


def demo_mode_comparison():
    """
    Compare publication vs display modes side-by-side.
    """
    print("\n" + "=" * 70)
    print("DEMO 4: Mode Comparison (Publication vs Display)")
    print("=" * 70)
    print("\n💡 Visual comparison of the two modes\n")

    output_comp = "/tmp/demo_mode_comparison.png"
    fig_comp = compare_modes(
        axes_width_mm=30, axes_height_mm=21, output_path=output_comp
    )

    print(f"\n✅ Mode comparison saved to: {output_comp}")
    print("   This shows both modes side-by-side with dimensions\n")


def demo_custom_margins():
    """
    Custom margins for axes with long labels.
    """
    print("\n" + "=" * 70)
    print("DEMO 5: Custom Margins")
    print("=" * 70)
    print("\n💡 Need space for long labels? Adjust margins!\n")

    # Create axes with custom margins
    fig, ax = create_axes_with_size_mm(
        axes_width_mm=30,
        axes_height_mm=21,
        mode="publication",
        margin_mm={
            "left": 8,  # Extra space for long y-axis label
            "right": 3,
            "bottom": 8,  # Extra space for long x-axis label
            "top": 4,  # Extra space for title
        },
    )

    x = np.linspace(0, 10, 50)
    ax.plot(x, x**2, "b-", lw=1)
    ax.set_xlabel("Very Long X-Axis Label (with units)")
    ax.set_ylabel("Y-Axis with\nMultiple Lines")
    ax.set_title("Custom Margins Example")

    print_dimension_info(fig, ax)

    output = "/tmp/demo_custom_margins.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"✅ Saved to: {output}\n")


def demo_with_style():
    """
    Combine axes size control with mm-based styling.
    """
    print("\n" + "=" * 70)
    print("DEMO 6: Axes Size + mm Styling")
    print("=" * 70)
    print("\n💡 Combine axes size control with Nature-style formatting\n")

    # Nature-style specification
    nature_style = {
        "axis_thickness_mm": 0.2,  # Thin axes
        "trace_thickness_mm": 0.12,  # Thin lines
        "tick_length_mm": 0.8,  # Short ticks
        "tick_thickness_mm": 0.2,
        "axis_font_size_pt": 8,
        "tick_font_size_pt": 7,
    }

    fig, ax = create_axes_with_size_mm(
        axes_width_mm=35,  # Nature single-column width
        axes_height_mm=35,  # Square panel
        mode="publication",
        style_mm=nature_style,
    )

    # Plot data
    x = np.linspace(0, 2 * np.pi, 100)
    ax.plot(x, np.sin(x), "b-", lw=0.5)
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_title("Nature-style 35×35mm Panel")

    print_dimension_info(fig, ax)

    output = "/tmp/demo_nature_style.tiff"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"✅ Saved Nature-style figure to: {output}\n")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("AXES SIZE WORKFLOW DEMO")
    print("=" * 70)
    print(
        "\nThis demonstrates the new workflow for publication-quality figures:"
    )
    print("  1. Specify AXES size (not figure size)")
    print("  2. Simple API usage")
    print("  3. Display vs Publication modes")
    print("  4. Dimension debugging tools")
    print()

    # Run demos
    demo_basic_workflow()
    demo_display_vs_publication()
    demo_dimension_viewer()
    demo_mode_comparison()
    demo_custom_margins()
    demo_with_style()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Key Takeaways")
    print("=" * 70)
    print("\n✅ What you wanted:")
    print(
        "   • Specify AXES box size (30×21mm) - the actual plot area you care about"
    )
    print("   • Simple usage - one function call")
    print(
        "   • Display mode for development (3x larger) vs Publication mode (exact mm)"
    )
    print("   • Tools to understand mm/inch/pixel/DPI relationships")
    print("\n✅ How to use:")
    print("   ```python")
    print("   from scitex.plt.utils import create_axes_with_size_mm")
    print("")
    print("   # For development (large display)")
    print("   fig, ax = create_axes_with_size_mm(")
    print("       axes_width_mm=30,")
    print("       axes_height_mm=21,")
    print("       mode='display'")
    print("   )")
    print("")
    print("   # For publication (exact mm)")
    print("   fig, ax = create_axes_with_size_mm(")
    print("       axes_width_mm=30,")
    print("       axes_height_mm=21,")
    print("       mode='publication'")
    print("   )")
    print("")
    print("   ax.plot(x, y)")
    print("   fig.savefig('figure.tiff', dpi=300, bbox_inches='tight')")
    print("   ```")
    print("\n✅ Debugging tools:")
    print("   • print_dimension_info(fig, ax) - Print all dimensions")
    print("   • view_dimensions(fig, ax) - Visual diagram")
    print("   • compare_modes() - Side-by-side comparison")
    print("\n📁 All demo outputs saved to /tmp/demo_*.png")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

# EOF
