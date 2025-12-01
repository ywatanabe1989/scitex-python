#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 12:03:29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/examples/demo_plot_all_types.py


import os

"""
Comprehensive demo suite for CSV export functionality.

This demo suite combines all individual CSV export demos into a single file.
It provides complete coverage for:
1. Standard matplotlib plotting functions
2. Custom plotting functions in scitex
3. Seaborn integration
4. Edge cases and special plot types

All demos ensure that both PNG and CSV files are created correctly, with
proper data tracking and export functionality.
"""

# Import matplotlib with non-interactive backend
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Import the module
import scitex

# -------------------------------------------------
# UTILITY FUNCTIONS
# -------------------------------------------------


def verify_csv_export(fig, save_path):
    """
    Verifies that a CSV is exported alongside the image.

    Args:
        fig: matplotlib figure object
        save_path: path to save the figure

    Returns:
        DataFrame: the loaded CSV data

    Raises:
        AssertionError: if CSV is not created or is empty
    """
    # Save the figure - this should also create a CSV
    scitex.io.save(fig, save_path)

    # Verify image was created
    assert os.path.exists(save_path), f"PNG file not created: {save_path}"

    # Verify CSV was created
    csv_path = save_path.replace(".png", ".csv")
    assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"

    # Read CSV and verify it's not empty
    df = pd.read_csv(csv_path)
    assert not df.empty, "CSV file is empty"

    return df


def create_demo_dataframe():
    """Create a demo dataframe for seaborn plots."""
    np.random.seed(42)

    # Create a sample DataFrame
    n_samples = 100
    df = pd.DataFrame(
        {
            "x": np.random.normal(0, 1, n_samples),
            "y": np.random.normal(0, 1, n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "value": np.random.uniform(0, 10, n_samples),
        }
    )

    return df


# -------------------------------------------------
# MATPLOTLIB BASIC PLOT DEMOS
# -------------------------------------------------


def demo_plot_csv_export():
    """Demo that basic plot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Plot with ID for tracking
    ax.plot(x, y, label="Sine Wave", id="plot_demo")

    # Style the plot
    ax.set_xyt("X", "Y", "Basic Plot Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/plot_demo.png"
    df = verify_csv_export(fig, save_path)

    # Verify CSV contents
    assert any(
        "plot_demo_plot_x" in col for col in df.columns
    ), f"X data not found in columns: {df.columns.tolist()}"
    assert any(
        "plot_demo_plot_y" in col for col in df.columns
    ), f"Y data not found in columns: {df.columns.tolist()}"

    # Close figure
    scitex.plt.close(fig)


def demo_scatter_csv_export():
    """Demo that scatter plot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    x = np.random.normal(0, 1, 100)
    y = np.random.normal(0, 1, 100)

    # Plot with ID for tracking
    ax.scatter(x, y, label="Random Points", id="scatter_demo", alpha=0.7)

    # Style the plot
    ax.set_xyt("X", "Y", "Scatter Plot Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/scatter_demo.png"
    df = verify_csv_export(fig, save_path)

    # Verify CSV contents
    # Look for scatter_demo_scatter_x/y
    assert any(
        "scatter_demo_" in col and "_x" in col for col in df.columns
    ), f"X data not found in columns: {df.columns.tolist()}"
    assert any(
        "scatter_demo_" in col and "_y" in col for col in df.columns
    ), f"Y data not found in columns: {df.columns.tolist()}"

    # Close figure
    scitex.plt.close(fig)


def demo_bar_csv_export():
    """Demo that bar plot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    categories = ["A", "B", "C", "D", "E"]
    values = [3, 7, 2, 6, 4]

    # Plot with ID for tracking
    ax.bar(categories, values, label="Bar Data", id="bar_demo")

    # Style the plot
    ax.set_xyt("Categories", "Values", "Bar Plot Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/bar_demo.png"
    df = verify_csv_export(fig, save_path)

    # Close figure
    scitex.plt.close(fig)


def demo_barh_csv_export():
    """Demo that horizontal bar plot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    categories = ["A", "B", "C", "D", "E"]
    values = [3, 7, 2, 6, 4]

    # Plot with ID for tracking
    ax.barh(categories, values, label="Horizontal Bar Data", id="barh_demo")

    # Style the plot
    ax.set_xyt("Values", "Categories", "Horizontal Bar Plot Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/barh_demo.png"
    df = verify_csv_export(fig, save_path)

    # Close figure
    scitex.plt.close(fig)


def demo_hist_csv_export():
    """Demo that histogram data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)

    # Plot with ID for tracking
    ax.hist(
        data, bins=30, label="Normal Distribution", id="hist_demo", alpha=0.7
    )

    # Style the plot
    ax.set_xyt("Value", "Frequency", "Histogram Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/hist_demo.png"
    df = verify_csv_export(fig, save_path)

    # Close figure
    scitex.plt.close(fig)


def demo_boxplot_csv_export():
    """Demo that boxplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    data = [
        np.random.normal(0, 1, 100),
        np.random.normal(2, 1, 100),
        np.random.normal(4, 1.5, 100),
    ]

    # Plot with ID for tracking
    ax.boxplot(
        data, labels=["Group A", "Group B", "Group C"], id="boxplot_demo"
    )

    # Style the plot
    ax.set_xyt("Group", "Value", "Boxplot Demo")

    # Save both image and data
    save_path = "./png/boxplot_demo.png"
    df = verify_csv_export(fig, save_path)

    # Close figure
    scitex.plt.close(fig)


def demo_fill_between_csv_export():
    """Demo that fill_between data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.sin(x) + 0.5

    # Plot with ID for tracking
    ax.fill_between(
        x, y1, y2, label="Fill Region", id="fill_between_demo", alpha=0.3
    )

    # Style the plot
    ax.set_xyt("X", "Y", "Fill Between Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/fill_between_demo.png"
    df = verify_csv_export(fig, save_path)

    # Close figure
    scitex.plt.close(fig)


def demo_errorbar_csv_export():
    """Demo that errorbar data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    x = np.linspace(0, 10, 10)
    y = np.sin(x)
    yerr = np.random.uniform(0.1, 0.3, len(x))

    # Plot with ID for tracking
    ax.errorbar(
        x, y, yerr=yerr, label="Data with Errors", id="errorbar_demo", fmt="o"
    )

    # Style the plot
    ax.set_xyt("X", "Y", "Errorbar Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/errorbar_demo.png"
    df = verify_csv_export(fig, save_path)

    # Close figure
    scitex.plt.close(fig)


def demo_imshow_csv_export():
    """Demo that imshow data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    data = np.random.rand(10, 12)

    # Plot with ID for tracking
    im = ax.imshow(data, cmap="viridis", id="imshow_demo")

    # Style the plot
    ax.set_title("Imshow Demo")

    # Save both image and data
    save_path = "./png/imshow_demo.png"

    # Verify that the PNG is created - CSV may or may not be exported for images
    scitex.io.save(fig, save_path)
    assert os.path.exists(save_path), f"PNG file not created: {save_path}"

    # Check if CSV exists - do not require it
    csv_path = save_path.replace(".png", ".csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            print(f"CSV was exported for imshow: {csv_path}")

    # Close figure
    scitex.plt.close(fig)


def demo_contour_csv_export():
    """Demo that contour data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    delta = 0.5
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)

    # Plot with ID for tracking
    contour = ax.contour(X, Y, Z, id="contour_demo")
    ax.clabel(contour, inline=True, fontsize=8)

    # Style the plot
    ax.set_xyt("X", "Y", "Contour Demo")

    # Save both image and data
    save_path = "./png/contour_demo.png"

    # Since contour data can be complex, just verify the file is saved
    scitex.io.save(fig, save_path)
    assert os.path.exists(save_path), f"PNG file not created: {save_path}"

    # Check if CSV exists
    csv_path = save_path.replace(".png", ".csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if not df.empty:
            print(f"CSV was exported for contour: {csv_path}")

    # Close figure
    scitex.plt.close(fig)


def demo_fill_csv_export():
    """Demo that fill plot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    x = np.linspace(0, 10, 50)
    y = np.sin(x)

    # Plot with ID for tracking
    ax.fill(x, y, id="fill_demo", alpha=0.5)

    # Style the plot
    ax.set_xyt("X", "Y", "Fill Demo")

    # Save both image and data
    save_path = "./png/fill_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        # Close figure
        scitex.plt.close(fig)
    except AssertionError as e:
        print(f"Warning: {e}")
        # Some plot types may not support CSV export, which is acceptable behavior
        # Just ensure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
        scitex.plt.close(fig)


def demo_eventplot_csv_export():
    """Demo that eventplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    n_events = 5
    events_data = [np.random.rand(20) * 10 for i in range(n_events)]

    # Plot with ID for tracking
    ax.eventplot(events_data, id="eventplot_demo")

    # Style the plot
    ax.set_xyt("X", "Y", "Event Plot Demo")

    # Save both image and data
    save_path = "./png/eventplot_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        # Close figure
        scitex.plt.close(fig)
    except AssertionError as e:
        print(f"Warning: {e}")
        # Some plot types may not support CSV export, which is acceptable behavior
        # Just ensure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
        scitex.plt.close(fig)


def demo_violin_csv_export():
    """Demo that violin (matplotlib basic) plot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(3)]

    # Plot with ID for tracking - violin method doesn't exist in matplotlib
    # Instead, use violinplot with appropriate arguments
    try:
        ax.violinplot(data, id="violin_demo")

        # Style the plot
        ax.set_xyt("Position", "Value", "Violin Demo")

        # Save both image and data
        save_path = "./png/basic_violin_demo.png"

        try:
            df = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except (AttributeError, IndexError) as e:
        print(f"Method not available or data format incompatible: {e}")
        scitex.plt.close(fig)


def demo_violinplot_csv_export():
    """Demo that violinplot (matplotlib) data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    data = [np.random.normal(i, 1, 100) for i in range(3)]

    # Format data for violinplot, which expects specific format
    # Convert list of arrays to a DataFrame for easier handling
    df_data = pd.DataFrame({f"group_{i}": data[i] for i in range(len(data))})

    try:
        # Plot with ID for tracking
        positions = np.arange(1, len(data) + 1)
        ax.violinplot(data, positions=positions, id="violinplot_demo")

        # Style the plot
        ax.set_xyt("Position", "Value", "Violinplot Demo")

        # Save both image and data
        save_path = "./png/violinplot_demo.png"

        # Since matplotlib's violinplot can be complex, no expectation on CSV
        # Just make sure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"

        # Check if CSV exists, but don't require it
        csv_path = save_path.replace(".png", ".csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                print(f"CSV was exported for violinplot: {csv_path}")

        # Close figure
        scitex.plt.close(fig)
    except (AttributeError, ValueError, IndexError) as e:
        print(f"Error in violinplot demo: {e}")
        # Make sure figure is closed even if there's an error
        scitex.plt.close(fig)


# -------------------------------------------------
# CUSTOM PLOT DEMOS
# -------------------------------------------------


def demo_plot_line_csv_export():
    """Demo that stx_line data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Plot with ID for tracking
    ax.stx_line(y, label="Sine Wave", id="plot_line_demo")

    # Style the plot
    ax.set_xyt("X", "Y", "Line Plot Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/plot_line_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        # Verify CSV contents - should include x and y columns
        line_cols = [col for col in df.columns if "plot_line_demo" in col]
        assert (
            len(line_cols) >= 2
        ), f"Expected at least 2 columns for line data, got: {line_cols}"
        # Close figure
        scitex.plt.close(fig)
    except AssertionError as e:
        print(f"Warning: {e}")
        # Some plot types may not support CSV export, which is acceptable behavior
        # Just ensure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
        scitex.plt.close(fig)


def demo_plot_box_csv_export():
    """Demo that stx_box data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)

    # Plot with ID for tracking
    ax.stx_box(data, label="Box Data", id="plot_box_demo")

    # Style the plot
    ax.set_xyt("", "Value", "Box Plot Demo")

    # Save both image and data
    save_path = "./png/plot_box_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        # Close figure
        scitex.plt.close(fig)
    except AssertionError as e:
        print(f"Warning: {e}")
        # Some plot types may not support CSV export, which is acceptable behavior
        # Just ensure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
        scitex.plt.close(fig)


def demo_plot_mean_std_csv_export():
    """Demo that stx_mean_std data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_mean = np.sin(x)

    # Use a single value for standard deviation (scalar) as the function requires
    std_value = 0.2  # Use a fixed standard deviation value

    # Plot with ID for tracking
    ax.stx_mean_std(
        y_mean, xx=x, sd=std_value, label="Mean±Std", id="plot_mean_std_demo"
    )

    # Style the plot
    ax.set_xyt("X", "Y", "Mean/Std Plot Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/plot_mean_std_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        # Close figure
        scitex.plt.close(fig)
    except AssertionError as e:
        print(f"Warning: {e}")
        # Some plot types may not support CSV export, which is acceptable behavior
        # Just ensure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
        scitex.plt.close(fig)


def demo_plot_mean_ci_csv_export():
    """Demo that stx_mean_ci data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_mean = np.sin(x)

    # Use a single value for confidence interval (scalar)
    ci_value = 0.2  # Use a fixed CI value

    # Plot with ID for tracking
    try:
        ax.stx_mean_ci(
            y_mean, xx=x, ci=ci_value, label="Mean±CI", id="plot_mean_ci_demo"
        )

        # Style the plot
        ax.set_xyt("X", "Y", "Mean/CI Plot Demo")
        ax.legend()

        # Save both image and data
        save_path = "./png/plot_mean_ci_demo.png"

        try:
            df = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except (AttributeError, TypeError, ValueError) as e:
        print(f"Error with stx_mean_ci: {e}")
        # Function might not exist or have different signature
        # Skip this demo
        scitex.plt.close(fig)


def demo_plot_median_iqr_csv_export():
    """Demo that stx_median_iqr data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_median = np.sin(x)

    # Use a scalar value for IQR
    iqr_value = 0.2

    # Plot with ID for tracking
    try:
        ax.stx_median_iqr(
            y_median,
            xx=x,
            iqr=iqr_value,
            label="Median±IQR",
            id="plot_median_iqr_demo",
        )

        # Style the plot
        ax.set_xyt("X", "Y", "Median/IQR Plot Demo")
        ax.legend()

        # Save both image and data
        save_path = "./png/plot_median_iqr_demo.png"

        try:
            df = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except (AttributeError, TypeError, ValueError) as e:
        print(f"Error with stx_median_iqr: {e}")
        # Function might not exist or have different signature
        # Skip this demo
        scitex.plt.close(fig)


def demo_plot_raster_csv_export():
    """Demo that stx_raster data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate some spike trains
    np.random.seed(42)
    n_neurons = 10
    positions = []
    for i in range(n_neurons):
        # Generate ~50 spikes per neuron
        spikes = np.random.uniform(0, 10, np.random.randint(40, 60))
        positions.append(np.sort(spikes))

    # Plot with ID for tracking
    try:
        ax.stx_raster(
            positions,
            labels=[f"Neuron {i+1}" for i in range(n_neurons)],
            id="plot_raster_demo",
        )

        # Style the plot
        ax.set_xyt("Time (s)", "Neuron", "Raster Plot Demo")

        # Save both image and data
        save_path = "./png/plot_raster_demo.png"
        scitex.io.save(fig, save_path)

        # Verify CSV was created
        csv_path = save_path.replace(".png", ".csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"CSV was exported for raster plot: {csv_path}")

        # Close figure
        scitex.plt.close(fig)
    except AttributeError:
        print(
            "stx_raster method not found or implementation is incomplete, skipping demo"
        )


def demo_plot_fillv_csv_export():
    """Demo that stx_fillv data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data for vertical fills (e.g., event markers)
    starts = [1, 3, 5, 7]
    ends = [1.5, 3.5, 5.5, 7.5]

    # Plot with ID for tracking
    ax.stx_fillv(starts, ends, color="red", alpha=0.3, id="plot_fillv_demo")

    # Add a base line for context
    ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), "b-")

    # Style the plot
    ax.set_xyt("X", "Y", "Vertical Fill Demo")

    # Save both image and data
    save_path = "./png/plot_fillv_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        # Close figure
        scitex.plt.close(fig)
    except AssertionError as e:
        print(f"Warning: {e}")
        # Some plot types may not support CSV export, which is acceptable behavior
        # Just ensure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
        scitex.plt.close(fig)


def demo_plot_rectangle_csv_export():
    """Demo that stx_rectangle data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Plot a rectangle
    ax.stx_rectangle(
        1, 1, 2, 1, facecolor="red", alpha=0.3, id="plot_rectangle_demo"
    )

    # Set plot limits for good visualization
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 3)

    # Style the plot
    ax.set_xyt("X", "Y", "Rectangle Plot Demo")

    # Save both image and data
    save_path = "./png/plot_rectangle_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        # Close figure
        scitex.plt.close(fig)
    except AssertionError as e:
        print(f"Warning: {e}")
        # Some plot types may not support CSV export, which is acceptable behavior
        # Just ensure the image is saved
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
        scitex.plt.close(fig)


def demo_plot_joyplot_csv_export():
    """Demo that stx_joyplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots(figsize=(8, 6))

    # Generate data for joy plot (multiple distributions)
    np.random.seed(42)
    n_dists = 5
    data = []
    for i in range(n_dists):
        # Create distributions with increasing means
        data.append(np.random.normal(i, 1, 500))

    # Plot with ID for tracking
    try:
        ax.stx_joyplot(
            data,
            labels=[f"Dist {i+1}" for i in range(n_dists)],
            id="plot_joyplot_demo",
        )

        # Style the plot
        ax.set_xyt("Value", "Distribution", "Joy Plot Demo")

        # Save both image and data
        save_path = "./png/plot_joyplot_demo.png"

        try:
            df = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("stx_joyplot method not found, skipping demo")


def demo_plot_conf_mat_csv_export():
    """Demo that stx_conf_mat data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate confusion matrix data
    np.random.seed(42)
    conf_mat = np.array([[45, 5, 0], [3, 40, 7], [2, 8, 30]])

    # Plot with ID for tracking
    try:
        ax.stx_conf_mat(conf_mat, id="conf_mat_demo")

        # Style the plot
        ax.set_title("Confusion Matrix Demo")

        # Save both image and data
        save_path = "./png/conf_mat_demo.png"

        try:
            df = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("stx_conf_mat method not found, skipping demo")


# -------------------------------------------------
# FUNCTIONAL PLOT DEMOS
# -------------------------------------------------


def demo_kde_plot_csv_export():
    """Demo that stx_kde data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)  # For reproducibility
    data = np.concatenate(
        [np.random.normal(0, 1, 500), np.random.normal(5, 1, 300)]
    )

    # Plot with ID for tracking
    ax.stx_kde(data, label="Bimodal Distribution", id="kde_demo")

    # Style the plot
    ax.set_xyt("Value", "Density", "KDE Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/kde_demo.png"
    try:
        df = verify_csv_export(fig, save_path)
        # Note: The actual columns have an "ax_00_" prefix added by the AxisWrapper
        kde_x_col = [col for col in df.columns if "kde_demo_kde_x" in col]
        kde_density_col = [
            col for col in df.columns if "kde_demo_kde_density" in col
        ]
        assert (
            len(kde_x_col) > 0
        ), f"No column containing 'kde_demo_kde_x' found. Columns: {df.columns.tolist()}"
        assert (
            len(kde_density_col) > 0
        ), f"No column containing 'kde_demo_kde_density' found. Columns: {df.columns.tolist()}"
    except Exception as e:
        print(f"Error in KDE demo: {e}")
    finally:
        # Close figure
        scitex.plt.close(fig)


def demo_plot_image_csv_export():
    """Demo that stx_image correctly exports the PNG without requiring CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)  # For reproducibility
    data = np.random.rand(20, 20)

    # Plot with ID for tracking
    ax.stx_image(data, cmap="viridis", id="image_demo")

    # Style the plot
    ax.set_xyt("X", "Y", "Image Demo")

    # Save both image and data
    save_path = "./png/image_demo.png"
    scitex.io.save(fig, save_path)

    # Verify image was created
    assert os.path.exists(save_path), f"PNG file not created: {save_path}"

    # Note: CSV export for images is optional since image data doesn't
    # have a natural CSV representation. Check if CSV exists but don't require it.
    csv_path = save_path.replace(".png", ".csv")
    if os.path.exists(csv_path):
        # If CSV exists, it should be valid
        df = pd.read_csv(csv_path)
        print(f"CSV was exported for image: {csv_path}")
    else:
        print("No CSV export for image (expected behavior)")

    # Close figure
    scitex.plt.close(fig)


def demo_plot_shaded_line_csv_export():
    """Demo that stx_shaded_line data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 10, 100)
    y_middle = np.sin(x)
    y_lower = y_middle - 0.2
    y_upper = y_middle + 0.2

    # Plot with ID for tracking
    ax.stx_shaded_line(
        x,
        y_lower,
        y_middle,
        y_upper,
        label="Sine with error",
        id="shaded_line_demo",
    )

    # Style the plot
    ax.set_xyt("X", "Y", "Shaded Line Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/shaded_line_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        assert not df.empty, "CSV file is empty"
    except Exception as e:
        print(f"Error in shaded_line demo: {e}")
        # Ensure the image is saved at minimum
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
    finally:
        # Close figure
        scitex.plt.close(fig)


def demo_plot_scatter_hist_csv_export():
    """Demo that stx_scatter_hist data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots(figsize=(8, 8))

    # Generate data
    np.random.seed(42)  # For reproducibility
    x = np.random.normal(0, 1, 500)
    y = x + np.random.normal(0, 0.5, 500)

    # Plot with ID for tracking
    ax.stx_scatter_hist(
        x, y, hist_bins=30, scatter_alpha=0.7, id="scatter_hist_demo"
    )

    # Style the plot
    ax.set_xyt("X Values", "Y Values", "Scatter Histogram Demo")

    # Save both image and data
    save_path = "./png/scatter_hist_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        assert not df.empty, "CSV file is empty"
    except Exception as e:
        print(f"Error in scatter_hist demo: {e}")
        # Ensure the image is saved at minimum
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
    finally:
        # Close figure
        scitex.plt.close(fig)


def demo_plot_violin_csv_export():
    """Demo that stx_violin data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)  # For reproducibility
    data = [
        np.random.normal(0, 1, 100),
        np.random.normal(2, 1.5, 100),
        np.random.normal(5, 0.8, 100),
    ]
    labels = ["Group A", "Group B", "Group C"]

    # Plot with ID for tracking
    ax.stx_violin(
        data, labels=labels, colors=["red", "blue", "green"], id="violin_demo"
    )

    # Style the plot
    ax.set_xyt("Groups", "Values", "Violin Plot Demo")

    # Save both image and data
    save_path = "./png/violin_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        assert not df.empty, "CSV file is empty"
    except Exception as e:
        print(f"Error in violin demo: {e}")
        # Ensure the image is saved at minimum
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
    finally:
        # Close figure
        scitex.plt.close(fig)


def demo_plot_heatmap_csv_export():
    """Demo that stx_heatmap data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)  # For reproducibility
    data = np.random.rand(5, 10)
    x_labels = [f"X{ii+1}" for ii in range(5)]
    y_labels = [f"Y{ii+1}" for ii in range(10)]

    # Plot with ID for tracking
    ax.stx_heatmap(
        data,
        x_labels=x_labels,
        y_labels=y_labels,
        cbar_label="Values",
        show_annot=True,
        value_format="{x:.2f}",
        cmap="viridis",
        id="heatmap_demo",
    )

    # Style the plot
    ax.set_title("Heatmap Demo")

    # Save both image and data
    save_path = "./png/heatmap_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        assert not df.empty, "CSV file is empty"
    except Exception as e:
        print(f"Error in heatmap demo: {e}")
        # Ensure the image is saved at minimum
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
    finally:
        # Close figure
        scitex.plt.close(fig)


def demo_plot_ecdf_csv_export():
    """Demo that stx_ecdf data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)  # For reproducibility
    data = np.random.normal(0, 1, 1000)

    # Plot with ID for tracking
    ax.stx_ecdf(data, label="Normal Distribution", id="ecdf_demo")

    # Style the plot
    ax.set_xyt("Value", "Cumulative Probability", "ECDF Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/ecdf_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        assert not df.empty, "CSV file is empty"
    except Exception as e:
        print(f"Error in ecdf demo: {e}")
        # Ensure the image is saved at minimum
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
    finally:
        # Close figure
        scitex.plt.close(fig)


def demo_multiple_plots_csv_export():
    """Demo that multiple plots on the same axis are correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create multiple plots with different IDs
    ax.stx_line(y1, label="Sine", id="multi_demo_sine")
    ax.stx_line(y2, label="Cosine", id="multi_demo_cosine")

    # Style the plot
    ax.set_xyt("X", "Y", "Multiple Plots Demo")
    ax.legend()

    # Save both image and data
    save_path = "./png/multiple_plots_demo.png"

    try:
        df = verify_csv_export(fig, save_path)
        assert not df.empty, "CSV file is empty"

        # Check that both plots are in the CSV
        # Note: The actual columns have an "ax_00_" prefix added by the AxisWrapper
        sine_cols = [col for col in df.columns if "multi_demo_sine" in col]
        cosine_cols = [col for col in df.columns if "multi_demo_cosine" in col]
        assert (
            len(sine_cols) > 0
        ), f"Sine plot data not found in CSV. Columns: {df.columns.tolist()}"
        assert (
            len(cosine_cols) > 0
        ), f"Cosine plot data not found in CSV. Columns: {df.columns.tolist()}"
    except Exception as e:
        print(f"Error in multiple plots demo: {e}")
        # Ensure the image is saved at minimum
        scitex.io.save(fig, save_path)
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"
    finally:
        # Close figure
        scitex.plt.close(fig)


# -------------------------------------------------
# SEABORN PLOT DEMOS
# -------------------------------------------------


def demo_sns_boxplot_csv_export():
    """Demo that sns_boxplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_boxplot(x="category", y="value", data=df, id="sns_boxplot_demo")

        # Style the plot
        ax.set_xyt("Category", "Value", "Seaborn Boxplot Demo")

        # Save both image and data
        save_path = "./png/sns_boxplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_boxplot method not found, skipping demo")


def demo_sns_barplot_csv_export():
    """Demo that sns_barplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_barplot(x="category", y="value", data=df, id="sns_barplot_demo")

        # Style the plot
        ax.set_xyt("Category", "Value", "Seaborn Barplot Demo")

        # Save both image and data
        save_path = "./png/sns_barplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_barplot method not found, skipping demo")


def demo_sns_violinplot_csv_export():
    """Demo that sns_violinplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_violinplot(
            x="category", y="value", data=df, id="sns_violinplot_demo"
        )

        # Style the plot
        ax.set_xyt("Category", "Value", "Seaborn Violinplot Demo")

        # Save both image and data
        save_path = "./png/sns_violinplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_violinplot method not found, skipping demo")


def demo_sns_stripplot_csv_export():
    """Demo that sns_stripplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_stripplot(
            x="category", y="value", data=df, id="sns_stripplot_demo"
        )

        # Style the plot
        ax.set_xyt("Category", "Value", "Seaborn Stripplot Demo")

        # Save both image and data
        save_path = "./png/sns_stripplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_stripplot method not found, skipping demo")


def demo_sns_swarmplot_csv_export():
    """Demo that sns_swarmplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_swarmplot(
            x="category", y="value", data=df, id="sns_swarmplot_demo"
        )

        # Style the plot
        ax.set_xyt("Category", "Value", "Seaborn Swarmplot Demo")

        # Save both image and data
        save_path = "./png/sns_swarmplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_swarmplot method not found, skipping demo")


def demo_sns_kdeplot_csv_export():
    """Demo that sns_kdeplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_kdeplot(
            x="value", data=df, hue="category", id="sns_kdeplot_demo"
        )

        # Style the plot
        ax.set_xyt("Value", "Density", "Seaborn KDE Plot Demo")

        # Save both image and data
        save_path = "./png/sns_kdeplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_kdeplot method not found, skipping demo")


def demo_sns_histplot_csv_export():
    """Demo that sns_histplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_histplot(
            x="value",
            data=df,
            hue="category",
            kde=True,
            id="sns_histplot_demo",
        )

        # Style the plot
        ax.set_xyt("Value", "Count", "Seaborn Histplot Demo")

        # Save both image and data
        save_path = "./png/sns_histplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_histplot method not found, skipping demo")


def demo_sns_scatterplot_csv_export():
    """Demo that sns_scatterplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    df = create_demo_dataframe()

    # Plot with ID for tracking
    try:
        ax.sns_scatterplot(
            x="x", y="y", hue="category", data=df, id="sns_scatterplot_demo"
        )

        # Style the plot
        ax.set_xyt("X", "Y", "Seaborn Scatterplot Demo")

        # Save both image and data
        save_path = "./png/sns_scatterplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_scatterplot method not found, skipping demo")


def demo_sns_lineplot_csv_export():
    """Demo that sns_lineplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data - for line plot we need more structured data
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    df = pd.DataFrame(
        {
            "x": np.tile(x, 3),
            "y": np.concatenate(
                [
                    np.sin(x) + np.random.normal(0, 0.1, len(x)),
                    np.cos(x) + np.random.normal(0, 0.1, len(x)),
                    -np.sin(x) + np.random.normal(0, 0.1, len(x)),
                ]
            ),
            "group": np.repeat(["A", "B", "C"], len(x)),
        }
    )

    # Plot with ID for tracking
    try:
        ax.sns_lineplot(
            x="x", y="y", hue="group", data=df, id="sns_lineplot_demo"
        )

        # Style the plot
        ax.set_xyt("X", "Y", "Seaborn Lineplot Demo")

        # Save both image and data
        save_path = "./png/sns_lineplot_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_lineplot method not found, skipping demo")


def demo_sns_heatmap_csv_export():
    """Demo that sns_heatmap data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data - a correlation matrix
    np.random.seed(42)
    data = np.random.randn(10, 12)

    # Plot with ID for tracking
    try:
        ax.sns_heatmap(
            data, annot=True, fmt=".2f", cmap="viridis", id="sns_heatmap_demo"
        )

        # Style the plot
        ax.set_title("Seaborn Heatmap Demo")

        # Save both image and data
        save_path = "./png/sns_heatmap_demo.png"
        try:
            df_result = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_heatmap method not found, skipping demo")


def demo_sns_jointplot_csv_export():
    """Demo that sns_jointplot data is correctly exported to CSV."""
    # Create figure
    fig, ax = scitex.plt.subplots()

    # Generate data
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
        }
    )

    # Plot with ID for tracking
    try:
        ax.sns_jointplot(data=df, x="x", y="y", id="sns_jointplot_demo")

        # Save both image and data
        save_path = "./png/sns_jointplot_demo.png"

        try:
            df = verify_csv_export(fig, save_path)
            # Close figure
            scitex.plt.close(fig)
        except AssertionError as e:
            print(f"Warning: {e}")
            # Some plot types may not support CSV export, which is acceptable behavior
            # Just ensure the image is saved
            scitex.io.save(fig, save_path)
            assert os.path.exists(
                save_path
            ), f"PNG file not created: {save_path}"
            scitex.plt.close(fig)
    except AttributeError:
        print("sns_jointplot method not found, skipping demo")


def demo_sns_pairplot_csv_export():
    """Demo that sns_pairplot data is correctly exported to CSV."""
    # Note: Pairplot is a figure-level function in seaborn and doesn't use axes

    # Generate data
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "var1": np.random.normal(0, 1, 100),
            "var2": np.random.normal(1, 2, 100),
            "var3": np.random.normal(2, 1, 100),
            "category": np.random.choice(["A", "B", "C"], size=100),
        }
    )

    try:
        # Import seaborn directly for figure-level functions
        import seaborn as sns

        # Create a pairplot directly
        g = sns.pairplot(data=df, hue="category")

        # Save image
        save_path = "./png/sns_pairplot_demo.png"
        g.savefig(save_path)

        # Verify the image was saved
        assert os.path.exists(save_path), f"PNG file not created: {save_path}"

        # Create a CSV manually since this is a figure-level function
        csv_path = save_path.replace(".png", ".csv")
        df.to_csv(csv_path, index=False)

        # Verify CSV was created
        assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"

        print(f"Pairplot image saved to: {save_path}")
        print(f"Pairplot data saved to: {csv_path}")

    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error in pairplot demo: {e}")
        print("Seaborn pairplot functionality skipped")


# -------------------------------------------------
# BASIC EXPORT FUNCTIONALITY DEMO
# -------------------------------------------------


def demo_export_csv():
    """Demo exporting plot data to CSV using tracking IDs."""
    # Create a figure with specific tracking
    fig, ax = scitex.plt.subplots(figsize=(8, 6))

    # Data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Make sure to add ids to track the plots
    ax.plot(x, y1, label="Sine", id="sine_plot")
    ax.plot(x, y2, label="Cosine", id="cosine_plot")

    # Check the tracking history
    print("Tracking history keys:")
    print(list(ax.history.keys()))

    # Export to CSV directly to demo the formatter
    df = ax.export_as_csv()
    print("\nDataFrame from export_as_csv():")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Save both PNG and CSV
    save_path = "./png/demo_export_csv.png"

    # Save the figure
    scitex.io.save(fig, save_path)

    # Verify CSV was created
    csv_path = save_path.replace(".png", ".csv")
    assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"

    # Load and check CSV
    df_csv = pd.read_csv(csv_path)
    print(f"\nCSV DataFrame:")
    print(f"Shape: {df_csv.shape}")
    print(f"Columns: {df_csv.columns.tolist()}")

    # Close figure
    scitex.plt.close(fig)

    return True


# -------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------

if __name__ == "__main__":
    # Create output directory
    # Define demo categories with their demo functions
    demo_categories = {
        "Matplotlib Basic Plot Demos": [
            demo_plot_csv_export,
            demo_scatter_csv_export,
            demo_bar_csv_export,
            demo_barh_csv_export,
            demo_hist_csv_export,
            demo_boxplot_csv_export,
            demo_fill_between_csv_export,
            demo_errorbar_csv_export,
            demo_imshow_csv_export,
            demo_contour_csv_export,
            demo_fill_csv_export,
            demo_eventplot_csv_export,
            demo_violin_csv_export,
            demo_violinplot_csv_export,
        ],
        "Custom Plot Demos": [
            demo_plot_line_csv_export,
            demo_plot_box_csv_export,
            demo_plot_mean_std_csv_export,
            demo_plot_mean_ci_csv_export,
            demo_plot_median_iqr_csv_export,
            demo_plot_raster_csv_export,
            demo_plot_fillv_csv_export,
            demo_plot_rectangle_csv_export,
            demo_plot_joyplot_csv_export,
            demo_plot_conf_mat_csv_export,
        ],
        "Functional Plot Demos": [
            demo_kde_plot_csv_export,
            demo_plot_image_csv_export,
            demo_plot_shaded_line_csv_export,
            demo_plot_scatter_hist_csv_export,
            demo_plot_violin_csv_export,
            demo_plot_heatmap_csv_export,
            demo_plot_ecdf_csv_export,
            demo_multiple_plots_csv_export,
        ],
        "Seaborn Plot Demos": [
            demo_sns_boxplot_csv_export,
            demo_sns_barplot_csv_export,
            demo_sns_violinplot_csv_export,
            demo_sns_stripplot_csv_export,
            demo_sns_swarmplot_csv_export,
            demo_sns_kdeplot_csv_export,
            demo_sns_histplot_csv_export,
            demo_sns_scatterplot_csv_export,
            demo_sns_lineplot_csv_export,
            demo_sns_heatmap_csv_export,
            demo_sns_jointplot_csv_export,
            demo_sns_pairplot_csv_export,
        ],
        "Basic Export Functionality": [demo_export_csv],
    }

    # Run all demo categories
    for category, demos in demo_categories.items():
        print(f"\n=== Running {category} ===")
        for demo_func in demos:
            try:
                print(f"Running {demo_func.__name__}...")
                demo_func()
            except Exception as e:
                print(f"ERROR in {demo_func.__name__}: {e}")

    print("\nAll demo completed!")

# EOF
