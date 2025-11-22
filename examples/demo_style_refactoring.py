#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: Style Refactoring - Separated Style Properties

This demonstrates the new style-separated architecture where:
- PlotModel has a 'style' field with PlotStyle
- AxesModel has a 'style' field with AxesStyle
- GuideModel has a 'style' field with GuideStyle
- AnnotationModel has a 'style' field with TextStyle

Benefits:
- Easy UI property panel generation
- Clean style copy/paste
- Batch style application
- Backward compatible with old flat JSON format
"""

import numpy as np
import sys
sys.path.insert(0, '/home/ywatanabe/proj/scitex-code/src')

from scitex.vis.model import (
    FigureModel,
    AxesModel,
    PlotModel,
    GuideModel,
    AnnotationModel,
    PlotStyle,
    AxesStyle,
    GuideStyle,
    TextStyle,
    copy_plot_style,
    apply_style_to_plots,
)
from scitex.vis.backend import build_figure_from_json


def demo_new_style_structure():
    """Demonstrate using the new separated style structure."""
    print("=" * 60)
    print("Demo 1: New Style Structure")
    print("=" * 60)

    # Create a plot with separated style
    plot_style = PlotStyle(
        color="blue",
        linewidth=2.0,
        linestyle="-",
        marker="o",
        markersize=6,
        alpha=0.8,
    )

    x = np.linspace(0, 10, 50)
    y = np.sin(x)

    plot = PlotModel(
        plot_type="line",
        data={"x": x.tolist(), "y": y.tolist()},
        label="sin(x)",
        plot_id="plot-1",
        tags=["primary", "sine"],
        style=plot_style,
    )

    # Create axes with separated style
    axes_style = AxesStyle(
        grid=True,
        grid_alpha=0.3,
        grid_linestyle="--",
        legend=True,
        legend_loc="upper right",
        facecolor="#f9f9f9",
    )

    axes = AxesModel(
        row=0,
        col=0,
        xlabel="X axis",
        ylabel="Y axis",
        title="Separated Style Demo",
        xlim=[0, 10],
        ylim=[-1.5, 1.5],
        style=axes_style,
    )

    axes.add_plot(plot.to_dict())

    # Create figure
    fig_model = FigureModel(
        width_mm=89,
        height_mm=80,
        nrows=1,
        ncols=1,
    )
    fig_model.add_axes(axes.to_dict())

    # Render
    fig_json = fig_model.to_dict()
    fig, axes_list = build_figure_from_json(fig_json)

    print("✓ Created figure with new style structure")
    print(f"  - PlotModel has style: {plot.style}")
    print(f"  - AxesModel has style: {axes.style}")

    return fig, fig_json


def demo_backward_compatibility():
    """Demonstrate backward compatibility with old flat JSON format."""
    print("\n" + "=" * 60)
    print("Demo 2: Backward Compatibility")
    print("=" * 60)

    # Old format: flat structure without 'style' field
    old_format_plot = {
        "plot_type": "line",
        "data": {"x": [0, 1, 2, 3], "y": [0, 1, 4, 9]},
        "color": "red",
        "linewidth": 2,
        "alpha": 0.7,
        "label": "Old format",
    }

    # Should be automatically converted to new format
    plot = PlotModel.from_dict(old_format_plot)

    print("✓ Old format JSON loaded successfully")
    print(f"  - Style extracted: color={plot.style.color}, linewidth={plot.style.linewidth}")
    print(f"  - New format: {plot.to_dict()}")


def demo_style_copy():
    """Demonstrate style copy/paste functionality."""
    print("\n" + "=" * 60)
    print("Demo 3: Style Copy/Paste")
    print("=" * 60)

    # Create source plot with nice style
    blue_style = PlotStyle(
        color="darkblue",
        linewidth=2.5,
        alpha=0.9,
        linestyle="-",
    )

    x = np.linspace(0, 10, 50)

    plot1 = PlotModel(
        plot_type="line",
        data={"x": x.tolist(), "y": np.sin(x).tolist()},
        label="Plot 1",
        style=blue_style,
    )

    # Create destination plot with default style
    plot2 = PlotModel(
        plot_type="line",
        data={"x": x.tolist(), "y": np.cos(x).tolist()},
        label="Plot 2",
    )

    print(f"Before copy: plot2.style.color = {plot2.style.color}")

    # Copy style from plot1 to plot2
    copy_plot_style(plot1, plot2)

    print(f"After copy:  plot2.style.color = {plot2.style.color}")
    print("✓ Style copied successfully")


def demo_batch_style_application():
    """Demonstrate applying style to multiple plots."""
    print("\n" + "=" * 60)
    print("Demo 4: Batch Style Application")
    print("=" * 60)

    # Create consistent style for all plots
    consistent_style = PlotStyle(
        color="#2E86AB",
        linewidth=1.8,
        alpha=0.85,
        marker="o",
        markersize=4,
    )

    # Create multiple plots
    x = np.linspace(0, 10, 30)
    plots = [
        PlotModel(
            plot_type="line",
            data={"x": x.tolist(), "y": (np.sin(x * (i + 1))).tolist()},
            label=f"Harmonic {i+1}",
        )
        for i in range(3)
    ]

    print(f"Before batch apply: {[p.style.color for p in plots]}")

    # Apply style to all plots at once
    apply_style_to_plots(consistent_style, plots)

    print(f"After batch apply:  {[p.style.color for p in plots]}")
    print("✓ Style applied to all plots")


def demo_ui_property_panel_simulation():
    """Simulate how UI property panels would work."""
    print("\n" + "=" * 60)
    print("Demo 5: UI Property Panel Simulation")
    print("=" * 60)

    plot = PlotModel(
        plot_type="scatter",
        data={"x": [1, 2, 3], "y": [2, 4, 6]},
        label="Test",
    )

    print("UI Property Panel for PlotStyle:")
    print("-" * 40)

    # Simulate property panel by listing all style fields
    style_dict = plot.style.to_dict()
    for prop, value in PlotStyle.__annotations__.items():
        current_value = getattr(plot.style, prop)
        print(f"  [{prop:20s}] ({value.__name__:15s}) = {current_value}")

    print("\n✓ Property panel would auto-generate from PlotStyle dataclass")
    print("  - Each field becomes a form input")
    print("  - Type hints determine input widget type")
    print("  - Easy to serialize/deserialize for backend")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Style Refactoring Demo".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    demo_new_style_structure()
    demo_backward_compatibility()
    demo_style_copy()
    demo_batch_style_application()
    demo_ui_property_panel_simulation()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
    print("\nKey improvements:")
    print("  ✓ Clean separation of structure and style")
    print("  ✓ Easy property panel generation")
    print("  ✓ Simple copy/paste operations")
    print("  ✓ Batch style application")
    print("  ✓ Backward compatible with old JSON format")
    print("  ✓ Type-safe with dataclass validation")
    print()


if __name__ == "__main__":
    main()

# EOF
