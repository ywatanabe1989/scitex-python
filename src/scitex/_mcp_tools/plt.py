#!/usr/bin/env python3
# Timestamp: 2026-01-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/plt.py
"""Plt module tools - delegates to figrecipe public API.

This module registers figrecipe's functionality under scitex's unified MCP server
with [plt] prefix for consistency with other scitex modules.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def register_plt_tools(mcp) -> None:
    """Register plt tools by delegating to figrecipe's public API."""
    # Ensure branding is set before any figrecipe imports
    os.environ.setdefault("FIGRECIPE_BRAND", "scitex.plt")
    os.environ.setdefault("FIGRECIPE_ALIAS", "plt")

    # Check if figrecipe is available
    try:
        import figrecipe as fr

        _FIGRECIPE_AVAILABLE = True
    except ImportError:
        _FIGRECIPE_AVAILABLE = False
        fr = None

    if not _FIGRECIPE_AVAILABLE:

        @mcp.tool()
        def plt_not_available() -> str:
            """[plt] figrecipe not installed."""
            return "figrecipe is required. Install with: pip install figrecipe"

        return

    # Re-register figrecipe tools with [plt] prefix
    @mcp.tool()
    def plt_plot(
        spec: Dict[str, Any],
        output_path: str,
        dpi: int = 300,
        save_recipe: bool = True,
    ) -> Dict[str, Any]:
        """[plt] Create a matplotlib figure from a declarative specification.

        See figrecipe._api._plot module docstring for full spec format.

        Parameters
        ----------
        spec : dict
            Declarative specification. Key sections: figure, plots, stat_annotations,
            xlabel, ylabel, title, legend, xlim, ylim.
        output_path : str
            Path to save the output figure.
        dpi : int
            DPI for raster output (default: 300).
        save_recipe : bool
            If True, also save as figrecipe YAML recipe.

        Returns
        -------
        dict
            Result with 'image_path' and 'recipe_path'.

        Raises
        ------
        ValueError
            If no plots are specified or data is missing.
        """
        from figrecipe._api._plot import create_figure_from_spec

        # Validate spec has plots with data
        plots = spec.get("plots", [])
        axes_specs = spec.get("axes") or spec.get("subplots", [])
        if not plots and not axes_specs:
            raise ValueError(
                "No plots specified in spec. Add 'plots' or 'axes' section."
            )
        for i, p in enumerate(plots):
            if not (p.get("x") or p.get("y") or p.get("data") or p.get("z")):
                raise ValueError(f"Plot {i} has no data (x, y, data, or z required)")

        result = create_figure_from_spec(
            spec=spec,
            output_path=output_path,
            dpi=dpi,
            save_recipe=save_recipe,
            show=False,
        )

        return {
            "image_path": str(result["image_path"]) if result["image_path"] else None,
            "recipe_path": str(result["recipe_path"])
            if result.get("recipe_path")
            else None,
            "success": True,
        }

    @mcp.tool()
    def plt_reproduce(
        recipe_path: str,
        output_path: Optional[str] = None,
        format: str = "png",
        dpi: int = 300,
    ) -> Dict[str, Any]:
        """[plt] Reproduce a figure from a saved YAML recipe.

        Parameters
        ----------
        recipe_path : str
            Path to the .yaml recipe file.

        output_path : str, optional
            Output path for the reproduced figure.
            Defaults to recipe_path with .reproduced.{format} suffix.

        format : str
            Output format: png, pdf, or svg.

        dpi : int
            DPI for raster output.

        Returns
        -------
        dict
            Result with 'output_path' and 'success'.
        """
        from pathlib import Path

        fig, axes = fr.reproduce(recipe_path)

        # Determine output path
        if output_path is None:
            recipe_p = Path(recipe_path)
            output_path = str(recipe_p.with_suffix(f".reproduced.{format}"))

        fig.savefig(output_path, dpi=dpi, format=format)
        import matplotlib.pyplot as plt

        plt.close(fig)

        return {"output_path": str(output_path), "success": True}

    @mcp.tool()
    def plt_compose(
        sources: List[str],
        output_path: str,
        layout: str = "horizontal",
        gap_mm: float = 5.0,
        dpi: int = 300,
        panel_labels: bool = True,
        label_style: str = "uppercase",
        caption: Optional[str] = None,
        create_symlinks: bool = True,
        canvas_size_mm: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """[plt] Compose multiple figures into a single figure with panel labels.

        Supports two modes:
        1. Grid-based layout (list sources): automatic arrangement with layout parameter
        2. Free-form positioning (dict sources): precise mm-based positioning

        Parameters
        ----------
        sources : list of str or dict
            Either:
            - List of paths to source images or recipe files (grid-based layout)
            - Dict mapping source paths to positioning specs with 'xy_mm' and 'size_mm':
              {"panel_a.yaml": {"xy_mm": [0, 0], "size_mm": [80, 50]}, ...}
        output_path : str
            Path to save the composed figure.
        layout : str
            Layout mode for list sources: 'horizontal', 'vertical', or 'grid'.
            Ignored when using dict sources with mm positioning.
        gap_mm : float
            Gap between panels in millimeters (for grid-based layout only).
        dpi : int
            DPI for output.
        panel_labels : bool
            If True, add panel labels (A, B, C, D) automatically.
        label_style : str
            Style: 'uppercase' (A,B,C), 'lowercase' (a,b,c), 'numeric' (1,2,3).
        caption : str, optional
            Figure caption to add below.
        create_symlinks : bool
            If True (default), create symlinks to source files for traceability.
        canvas_size_mm : tuple of (float, float), optional
            Canvas size as (width_mm, height_mm) for free-form positioning.
            Required when sources is a dict with mm positioning.

        Returns
        -------
        dict
            Result with 'output_path', 'success', and 'sources_dir' (if symlinks created).
        """
        from figrecipe import compose_figures

        result = compose_figures(
            sources=sources,
            output_path=output_path,
            layout=layout,
            gap_mm=gap_mm,
            dpi=dpi,
            panel_labels=panel_labels,
            label_style=label_style,
            caption=caption,
            create_symlinks=create_symlinks,
            canvas_size_mm=canvas_size_mm,
        )

        return {
            "output_path": str(result.get("output_path", output_path)),
            "success": True,
            "sources_dir": str(result.get("sources_dir"))
            if result.get("sources_dir")
            else None,
        }

    @mcp.tool()
    def plt_info(recipe_path: str, verbose: bool = False) -> Dict[str, Any]:
        """[plt] Get information about a recipe file.

        Parameters
        ----------
        recipe_path : str
            Path to the .yaml recipe file.

        verbose : bool
            If True, include detailed call information.

        Returns
        -------
        dict
            Recipe information including figure dimensions, call counts, etc.
        """
        return fr.info(recipe_path)

    @mcp.tool()
    def plt_validate(
        recipe_path: str,
        mse_threshold: float = 100.0,
    ) -> Dict[str, Any]:
        """[plt] Validate that a recipe can reproduce its original figure.

        Parameters
        ----------
        recipe_path : str
            Path to the .yaml recipe file.

        mse_threshold : float
            Maximum acceptable mean squared error (default: 100).

        Returns
        -------
        dict
            Validation result with 'passed', 'mse', and details.
        """
        result = fr.validate(recipe_path, mse_threshold=mse_threshold)

        return {
            "valid": result.valid,
            "mse": result.mse,
            "message": result.message,
            "recipe_path": str(recipe_path),
        }

    @mcp.tool()
    def plt_crop(
        input_path: str,
        output_path: Optional[str] = None,
        margin_mm: float = 1.0,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """[plt] Crop whitespace from a figure image.

        Parameters
        ----------
        input_path : str
            Path to the input image.

        output_path : str, optional
            Path for cropped output. Defaults to input with .cropped suffix.

        margin_mm : float
            Margin to keep around content in millimeters.

        overwrite : bool
            If True, overwrite the input file.

        Returns
        -------
        dict
            Result with 'output_path' and 'success'.
        """
        result_path = fr.crop(
            input_path,
            output_path=output_path,
            margin_mm=margin_mm,
            overwrite=overwrite,
        )

        return {"output_path": str(result_path), "success": True}

    @mcp.tool()
    def plt_extract_data(recipe_path: str) -> Dict[str, Dict[str, Any]]:
        """[plt] Extract plotted data arrays from a saved recipe.

        Parameters
        ----------
        recipe_path : str
            Path to the .yaml recipe file.

        Returns
        -------
        dict
            Nested dict: {call_id: {'x': list, 'y': list, ...}}
        """
        return fr.extract_data(recipe_path)

    @mcp.tool()
    def plt_list_styles() -> Dict[str, Any]:
        """[plt] List available figure style presets.

        Returns
        -------
        dict
            Dictionary with 'presets' list of available style names.
        """
        presets = fr.list_presets()
        return {"presets": presets, "success": True}

    @mcp.tool()
    def plt_get_plot_types() -> Dict[str, Any]:
        """[plt] Get list of supported plot types.

        Returns
        -------
        dict
            Dictionary with 'plot_types' and their descriptions.
        """
        # figrecipe supports these plot types via spec
        plot_types = {
            "line": "Line plot with optional error bars",
            "scatter": "Scatter plot with optional regression line",
            "bar": "Bar chart with optional error bars",
            "box": "Box plot for distribution visualization",
            "violin": "Violin plot for distribution visualization",
            "hist": "Histogram for distribution",
            "heatmap": "Heatmap/image plot",
            "contour": "Contour plot",
            "errorbar": "Error bar plot",
            "fill_between": "Filled area between curves",
            "imshow": "Image display",
        }
        return {"plot_types": plot_types, "success": True}


# EOF
