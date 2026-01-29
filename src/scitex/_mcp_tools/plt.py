#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/plt.py
"""Plt module tools for FastMCP unified server.

This module delegates to figrecipe's MCP tools for single source of truth.
All plt_* tools are thin wrappers around figrecipe's canonical implementation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple


def register_plt_tools(mcp) -> None:
    """Register plt tools with FastMCP server.

    Delegates to figrecipe's plt tools (canonical source).
    Tools are prefixed with 'plt_' for scitex namespace consistency.
    """
    # Ensure branding is set before any figrecipe imports
    os.environ.setdefault("FIGRECIPE_BRAND", "scitex.plt")
    os.environ.setdefault("FIGRECIPE_ALIAS", "plt")

    # Check if figrecipe is available
    try:
        from figrecipe._mcp import server as fr_mcp

        # Access underlying functions from FunctionTool objects
        # figrecipe tools are named plt_* for proper MCP categorization
        _plot = fr_mcp.plt_plot.fn
        _reproduce = fr_mcp.plt_reproduce.fn
        _compose = fr_mcp.plt_compose.fn
        _info = fr_mcp.plt_info.fn
        _validate = fr_mcp.plt_validate.fn
        _crop = fr_mcp.plt_crop.fn
        _extract_data = fr_mcp.plt_extract_data.fn
        _list_styles = fr_mcp.plt_list_styles.fn
        _get_plot_types = fr_mcp.plt_get_plot_types.fn

        _FIGRECIPE_AVAILABLE = True
    except ImportError:
        _FIGRECIPE_AVAILABLE = False

    if not _FIGRECIPE_AVAILABLE:

        @mcp.tool()
        def plt_not_available() -> str:
            """[plt] figrecipe not installed."""
            return "figrecipe is required for plt tools. Install with: pip install figrecipe"

        return

    # Delegate to figrecipe's MCP tools with plt_ prefix
    # Each wrapper simply calls the figrecipe function

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
        return _plot(spec, output_path, dpi, save_recipe)

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
        return _reproduce(recipe_path, output_path, format, dpi)

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
        facecolor: str = "white",
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
        facecolor : str
            Background color for the composed figure. Default is 'white'.
            All source panels are flattened onto this background to ensure
            consistent appearance regardless of original panel transparency.

        Returns
        -------
        dict
            Result with 'output_path', 'success', and 'sources_dir' (if symlinks created).
        """
        return _compose(
            sources,
            output_path,
            layout,
            gap_mm,
            dpi,
            panel_labels,
            label_style,
            caption,
            create_symlinks,
            canvas_size_mm,
            facecolor,
        )

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
        return _info(recipe_path, verbose)

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
        return _validate(recipe_path, mse_threshold)

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
        return _crop(input_path, output_path, margin_mm, overwrite)

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
        return _extract_data(recipe_path)

    @mcp.tool()
    def plt_list_styles() -> Dict[str, Any]:
        """[plt] List available figure style presets.

        Returns
        -------
        dict
            Dictionary with 'presets' list of available style names.
        """
        return _list_styles()

    @mcp.tool()
    def plt_get_plot_types() -> Dict[str, Any]:
        """[plt] Get list of supported plot types.

        Returns
        -------
        dict
            Dictionary with 'plot_types' and their descriptions.
        """
        return _get_plot_types()


# EOF
