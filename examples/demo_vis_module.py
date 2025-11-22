#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo: scitex.vis - Structured Visualization Module

This example demonstrates the new scitex.vis module that provides
a JSON-based approach to creating publication-quality figures.

The vis module completes the SciTeX ecosystem:
- scholar: Literature management
- writer: Document generation
- vis: Visualization (NEW!)
"""

import numpy as np
import scitex as stx


def demo_basic_usage():
    """Demonstrate basic vis module usage."""
    print("=" * 60)
    print("Demo 1: Basic Usage - Create Figure from JSON")
    print("=" * 60)

    # Create figure using Nature single-column template
    fig_json = stx.vis.get_template("nature_single", height_mm=100)

    # Add a simple line plot
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)

    fig_json["axes"] = [
        {
            "row": 0,
            "col": 0,
            "xlabel": "Time (s)",
            "ylabel": "Amplitude",
            "title": "Sine Wave",
            "plots": [
                {
                    "plot_type": "line",
                    "data": {"x": x.tolist(), "y": y.tolist()},
                    "color": "blue",
                    "linewidth": 2,
                    "label": "sin(x)",
                }
            ],
            "grid": True,
            "legend": True,
        }
    ]

    # Save the JSON specification
    stx.vis.save_figure_json(fig_json, "/tmp/demo_figure.json")
    print(f"✓ Saved figure JSON to /tmp/demo_figure.json")

    # Render to matplotlib figure
    fig, axes = stx.vis.build_figure_from_json(fig_json)
    print(f"✓ Built matplotlib figure: {type(fig)}")

    # Export to PNG
    stx.vis.export_figure(fig_json, "/tmp/demo_figure.png", dpi=300)
    print(f"✓ Exported to /tmp/demo_figure.png")

    print()


def demo_templates():
    """Demonstrate available templates."""
    print("=" * 60)
    print("Demo 2: Available Templates")
    print("=" * 60)

    templates = stx.vis.list_templates()
    print(f"Available templates: {templates}")

    for template_name in templates:
        template = stx.vis.get_template(template_name)
        print(
            f"  - {template_name:20s}: {template['width_mm']:6.1f} x {template['height_mm']:6.1f} mm"
        )

    print()


def demo_multiple_subplots():
    """Demonstrate multi-subplot figure."""
    print("=" * 60)
    print("Demo 3: Multi-Subplot Figure")
    print("=" * 60)

    # Create 2x2 subplot figure
    fig_json = stx.vis.get_template("nature_double", height_mm=160)
    fig_json["nrows"] = 2
    fig_json["ncols"] = 2

    # Generate data
    x = np.linspace(0, 10, 100)

    # Define plots for each subplot
    fig_json["axes"] = [
        # Top-left: Line plot
        {
            "row": 0,
            "col": 0,
            "xlabel": "X",
            "ylabel": "Y",
            "title": "Line Plot",
            "plots": [
                {
                    "plot_type": "line",
                    "data": {"x": x.tolist(), "y": np.sin(x).tolist()},
                    "color": "blue",
                }
            ],
        },
        # Top-right: Scatter plot
        {
            "row": 0,
            "col": 1,
            "xlabel": "X",
            "ylabel": "Y",
            "title": "Scatter Plot",
            "plots": [
                {
                    "plot_type": "scatter",
                    "data": {
                        "x": (x[::5] + np.random.randn(20) * 0.2).tolist(),
                        "y": (np.sin(x[::5]) + np.random.randn(20) * 0.2).tolist(),
                    },
                    "color": "red",
                    "alpha": 0.6,
                }
            ],
        },
        # Bottom-left: Bar plot
        {
            "row": 1,
            "col": 0,
            "xlabel": "Category",
            "ylabel": "Value",
            "title": "Bar Plot",
            "plots": [
                {
                    "plot_type": "bar",
                    "data": {
                        "x": [0, 1, 2, 3, 4],
                        "height": [3, 7, 2, 5, 8],
                    },
                    "color": "green",
                    "alpha": 0.7,
                }
            ],
        },
        # Bottom-right: Histogram
        {
            "row": 1,
            "col": 1,
            "xlabel": "Value",
            "ylabel": "Frequency",
            "title": "Histogram",
            "plots": [
                {
                    "plot_type": "hist",
                    "data": {"x": np.random.randn(1000).tolist()},
                    "bins": 30,
                    "color": "purple",
                    "alpha": 0.7,
                }
            ],
        },
    ]

    # Export
    output_path = "/tmp/demo_subplots.png"
    stx.vis.export_figure(fig_json, output_path, dpi=300)
    print(f"✓ Created 2x2 subplot figure: {output_path}")

    print()


def demo_project_workflow():
    """Demonstrate project-based workflow."""
    print("=" * 60)
    print("Demo 4: Project Workflow")
    print("=" * 60)

    project_dir = "/tmp/scitex_demo_project"

    # Create figure JSON
    fig_json = stx.vis.get_template("square", size_mm=120)
    x = np.linspace(0, 4 * np.pi, 200)

    fig_json["axes"] = [
        {
            "row": 0,
            "col": 0,
            "xlabel": "Phase (rad)",
            "ylabel": "Amplitude",
            "title": "Damped Oscillation",
            "plots": [
                {
                    "plot_type": "line",
                    "data": {
                        "x": x.tolist(),
                        "y": (np.exp(-x / 5) * np.sin(x)).tolist(),
                    },
                    "color": "darkblue",
                    "linewidth": 1.5,
                }
            ],
            "grid": True,
            "guides": [
                {
                    "guide_type": "axhline",
                    "y": 0,
                    "color": "gray",
                    "linestyle": "--",
                    "alpha": 0.5,
                }
            ],
        }
    ]

    # Save to project structure
    stx.vis.save_figure_json_to_project(
        project_dir=project_dir, figure_id="fig-damped-osc", fig_json=fig_json
    )
    print(f"✓ Saved to project: {project_dir}/scitex/vis/figs/fig-damped-osc.json")

    # List figures in project
    figures = stx.vis.io.list_figures_in_project(project_dir)
    print(f"✓ Figures in project: {figures}")

    # Load and render
    loaded_json = stx.vis.load_figure_json_from_project(
        project_dir, "fig-damped-osc"
    )
    stx.vis.export_figure(
        loaded_json, f"{project_dir}/output/fig-damped-osc.png", dpi=300
    )
    print(f"✓ Exported: {project_dir}/output/fig-damped-osc.png")

    print()


def demo_model_validation():
    """Demonstrate model validation."""
    print("=" * 60)
    print("Demo 5: Model Validation")
    print("=" * 60)

    # Create a FigureModel directly
    from scitex.vis.model import FigureModel, AxesModel, PlotModel

    fig_model = FigureModel(
        width_mm=stx.vis.NATURE_SINGLE_COLUMN_MM,
        height_mm=80,
        nrows=1,
        ncols=1,
    )

    # Validate
    try:
        fig_model.validate()
        print("✓ Figure model validation passed")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")

    # Add axes with plot
    axes_model = AxesModel(
        row=0,
        col=0,
        xlabel="Time",
        ylabel="Signal",
    )

    plot_model = PlotModel(
        plot_type="line",
        data={"x": [0, 1, 2, 3], "y": [0, 1, 4, 9]},
        color="navy",
    )

    axes_model.add_plot(plot_model.to_dict())
    fig_model.add_axes(axes_model.to_dict())

    print(f"✓ Created model with {len(fig_model.axes)} axes")

    # Convert to JSON and render
    fig_json = fig_model.to_dict()
    fig, axes = stx.vis.build_figure_from_json(fig_json)
    print(f"✓ Rendered model to matplotlib figure")

    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  SciTeX Visualization Module (scitex.vis) - Demo".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")

    demo_basic_usage()
    demo_templates()
    demo_multiple_subplots()
    demo_project_workflow()
    demo_model_validation()

    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
    print("\nThe scitex.vis module provides:")
    print("  • JSON-based figure specifications")
    print("  • Publication format templates (Nature, Science, etc.)")
    print("  • Project structure integration")
    print("  • Model validation")
    print("  • Seamless scitex.plt integration")
    print("\nThis completes the SciTeX ecosystem:")
    print("  • scholar/  → Literature management")
    print("  • writer/   → Document generation")
    print("  • vis/      → Visualization (NEW!)")
    print()


if __name__ == "__main__":
    main()

# EOF
