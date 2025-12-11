#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-08 23:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/_generate.py

"""Gallery generation functionality."""

import os
from pathlib import Path

from ._plots import PLOT_FUNCTIONS
from ._registry import CATEGORIES


def generate(
    output_dir="./gallery",
    category=None,
    plot_type=None,
    figsize=(4, 3),
    dpi=150,
    save_csv=True,
    save_png=True,
    save_svg=True,
    verbose=True,
):
    """Generate gallery plots with CSVs.

    Parameters
    ----------
    output_dir : str or Path
        Output directory for the gallery.
    category : str, optional
        Generate only plots in this category.
        Available: line, statistical, distribution, categorical, scatter,
                   area, grid, contour, vector, special
    plot_type : str, optional
        Generate only this specific plot type.
    figsize : tuple
        Figure size (width, height) in inches.
    dpi : int
        Resolution for PNG output.
    save_csv : bool
        Whether to save CSV data files.
    save_png : bool
        Whether to save PNG image files.
    save_svg : bool
        Whether to save SVG image files for element selection.
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict
        Dictionary with generated file paths.

    Examples
    --------
    >>> import scitex as stx
    >>> stx.plt.gallery.generate("./my_gallery")
    >>> stx.plt.gallery.generate("./my_gallery", category="line")
    >>> stx.plt.gallery.generate("./my_gallery", plot_type="scatter")
    """
    import scitex as stx
    from scitex.plt.styles.presets import SCITEX_STYLE

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which plots to generate
    plots_to_generate = _get_plots_to_generate(category, plot_type)

    if verbose:
        print(f"Generating {len(plots_to_generate)} plots to {output_dir}")

    results = {"png": [], "svg": [], "csv": [], "errors": []}

    for plot_name in plots_to_generate:
        if plot_name not in PLOT_FUNCTIONS:
            if verbose:
                print(f"  [SKIP] {plot_name}: not implemented")
            continue

        plot_func = PLOT_FUNCTIONS[plot_name]
        cat_name = _get_category_for_plot(plot_name)

        # Create category subdirectory
        cat_dir = output_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Create figure
            style = SCITEX_STYLE.copy()
            style["figsize"] = figsize
            fig, ax = stx.plt.subplots(**style)

            # Generate plot
            fig, ax = plot_func(fig, ax, stx)

            # Save PNG (this also generates JSON and CSV automatically)
            if save_png:
                png_path = cat_dir / f"{plot_name}.png"
                stx.io.save(fig, png_path, dpi=dpi)
                results["png"].append(str(png_path))
                if verbose:
                    print(f"  [PNG] {png_path}")

                # CSV is auto-generated alongside PNG by stx.io.save
                csv_path = cat_dir / f"{plot_name}.csv"
                if csv_path.exists():
                    results["csv"].append(str(csv_path))
                    if verbose:
                        print(f"  [CSV] {csv_path}")

            # Save SVG for element selection (using gid attributes)
            if save_svg:
                svg_path = cat_dir / f"{plot_name}.svg"
                stx.io.save(fig, svg_path)
                results["svg"].append(str(svg_path))
                if verbose:
                    print(f"  [SVG] {svg_path}")

            stx.plt.close(fig._fig_mpl if hasattr(fig, "_fig_mpl") else fig)

        except Exception as e:
            results["errors"].append({"plot": plot_name, "error": str(e)})
            if verbose:
                print(f"  [ERROR] {plot_name}: {e}")

    if verbose:
        print(f"\nGenerated: {len(results['png'])} PNG, {len(results['svg'])} SVG, {len(results['csv'])} CSV")
        if results["errors"]:
            print(f"Errors: {len(results['errors'])}")

    return results


def _get_plots_to_generate(category=None, plot_type=None):
    """Get list of plots to generate based on filters."""
    if plot_type is not None:
        return [plot_type]

    if category is not None:
        if category not in CATEGORIES:
            raise ValueError(
                f"Unknown category: {category}. Available: {list(CATEGORIES.keys())}"
            )
        return CATEGORIES[category]["plots"]

    # All plots
    all_plots = []
    for cat_info in CATEGORIES.values():
        all_plots.extend(cat_info["plots"])
    return all_plots


def _get_category_for_plot(plot_name):
    """Find which category a plot belongs to."""
    for cat_name, cat_info in CATEGORIES.items():
        if plot_name in cat_info["plots"]:
            return cat_name
    return "uncategorized"


# EOF
