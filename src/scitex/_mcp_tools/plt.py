#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/plt.py
"""Plt module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import Optional


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def register_plt_tools(mcp) -> None:
    """Register plt tools with FastMCP server."""

    # Style tools
    @mcp.tool()
    async def plt_get_style() -> str:
        """[plt] Get current SciTeX publication style configuration."""
        from scitex.plt._mcp._handlers_style import get_style_handler

        result = await get_style_handler()
        return _json(result)

    @mcp.tool()
    async def plt_set_style(
        axes_width_mm: Optional[float] = None,
        axes_height_mm: Optional[float] = None,
        margin_left_mm: Optional[float] = None,
        margin_right_mm: Optional[float] = None,
        margin_top_mm: Optional[float] = None,
        margin_bottom_mm: Optional[float] = None,
        dpi: Optional[int] = None,
        title_font_size_pt: Optional[float] = None,
        axis_font_size_pt: Optional[float] = None,
        tick_font_size_pt: Optional[float] = None,
        legend_font_size_pt: Optional[float] = None,
        trace_thickness_mm: Optional[float] = None,
        reset: Optional[bool] = None,
    ) -> str:
        """[plt] Set global style overrides for publication figures."""
        from scitex.plt._mcp._handlers_style import set_style_handler

        result = await set_style_handler(
            axes_width_mm=axes_width_mm,
            axes_height_mm=axes_height_mm,
            margin_left_mm=margin_left_mm,
            margin_right_mm=margin_right_mm,
            margin_top_mm=margin_top_mm,
            margin_bottom_mm=margin_bottom_mm,
            dpi=dpi,
            title_font_size_pt=title_font_size_pt,
            axis_font_size_pt=axis_font_size_pt,
            tick_font_size_pt=tick_font_size_pt,
            legend_font_size_pt=legend_font_size_pt,
            trace_thickness_mm=trace_thickness_mm,
            reset=reset,
        )
        return _json(result)

    @mcp.tool()
    async def plt_list_presets() -> str:
        """[plt] List available publication style presets."""
        from scitex.plt._mcp._handlers_style import list_presets_handler

        result = await list_presets_handler()
        return _json(result)

    @mcp.tool()
    async def plt_get_dpi_settings() -> str:
        """[plt] Get DPI settings for different output contexts."""
        from scitex.plt._mcp._handlers_style import get_dpi_settings_handler

        result = await get_dpi_settings_handler()
        return _json(result)

    @mcp.tool()
    async def plt_get_color_palette(format: str = "hex") -> str:
        """[plt] Get the SciTeX color palette for consistent figure colors."""
        from scitex.plt._mcp._handlers_style import get_color_palette_handler

        result = await get_color_palette_handler(format=format)
        return _json(result)

    # Figure tools
    @mcp.tool()
    async def plt_create_figure(
        nrows: int = 1,
        ncols: int = 1,
        axes_width_mm: float = 40,
        axes_height_mm: float = 28,
        space_w_mm: Optional[float] = None,
        space_h_mm: Optional[float] = None,
    ) -> str:
        """[plt] Create a multi-panel figure canvas with SciTeX style."""
        from scitex.plt._mcp._handlers_figure import create_figure_handler

        result = await create_figure_handler(
            nrows=nrows,
            ncols=ncols,
            axes_width_mm=axes_width_mm,
            axes_height_mm=axes_height_mm,
            space_w_mm=space_w_mm,
            space_h_mm=space_h_mm,
        )
        return _json(result)

    @mcp.tool()
    async def plt_crop_figure(
        input_path: str,
        output_path: Optional[str] = None,
        margin: int = 12,
        overwrite: bool = False,
    ) -> str:
        """[plt] Auto-crop whitespace from a saved figure image."""
        from scitex.plt._mcp._handlers_figure import crop_figure_handler

        result = await crop_figure_handler(
            input_path=input_path,
            output_path=output_path,
            margin=margin,
            overwrite=overwrite,
        )
        return _json(result)

    @mcp.tool()
    async def plt_save_figure(
        output_path: str,
        figure_id: Optional[str] = None,
        dpi: int = 300,
        crop: bool = True,
    ) -> str:
        """[plt] Save the current figure to file."""
        from scitex.plt._mcp._handlers_figure import save_figure_handler

        result = await save_figure_handler(
            output_path=output_path,
            figure_id=figure_id,
            dpi=dpi,
            crop=crop,
        )
        return _json(result)

    @mcp.tool()
    async def plt_close_figure(figure_id: Optional[str] = None) -> str:
        """[plt] Close a figure and free memory."""
        from scitex.plt._mcp._handlers_figure import close_figure_handler

        result = await close_figure_handler(figure_id=figure_id)
        return _json(result)

    # Plot tools
    @mcp.tool()
    async def plt_plot_bar(
        x: list,
        y: list,
        yerr: Optional[list] = None,
        colors: Optional[list] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        panel: str = "0,0",
        figure_id: Optional[str] = None,
    ) -> str:
        """[plt] Create a bar plot on specified panel."""
        from scitex.plt._mcp._handlers_plot import plot_bar_handler

        result = await plot_bar_handler(
            x=x,
            y=y,
            yerr=yerr,
            colors=colors,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            panel=panel,
            figure_id=figure_id,
        )
        return _json(result)

    @mcp.tool()
    async def plt_plot_scatter(
        x: list,
        y: list,
        color: Optional[str] = None,
        size: Optional[float] = None,
        alpha: Optional[float] = None,
        add_regression: bool = False,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        panel: str = "0,0",
        figure_id: Optional[str] = None,
    ) -> str:
        """[plt] Create a scatter plot on specified panel."""
        from scitex.plt._mcp._handlers_plot import plot_scatter_handler

        result = await plot_scatter_handler(
            x=x,
            y=y,
            color=color,
            size=size,
            alpha=alpha,
            add_regression=add_regression,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            panel=panel,
            figure_id=figure_id,
        )
        return _json(result)

    @mcp.tool()
    async def plt_plot_line(
        x: list,
        y: list,
        yerr: Optional[list] = None,
        color: Optional[str] = None,
        linestyle: str = "-",
        label: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        panel: str = "0,0",
        figure_id: Optional[str] = None,
    ) -> str:
        """[plt] Create a line plot on specified panel."""
        from scitex.plt._mcp._handlers_plot import plot_line_handler

        result = await plot_line_handler(
            x=x,
            y=y,
            yerr=yerr,
            color=color,
            linestyle=linestyle,
            label=label,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            panel=panel,
            figure_id=figure_id,
        )
        return _json(result)

    @mcp.tool()
    async def plt_plot_box(
        data: list,
        labels: Optional[list] = None,
        colors: Optional[list] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        panel: str = "0,0",
        figure_id: Optional[str] = None,
    ) -> str:
        """[plt] Create a box plot on specified panel."""
        from scitex.plt._mcp._handlers_plot import plot_box_handler

        result = await plot_box_handler(
            data=data,
            labels=labels,
            colors=colors,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            panel=panel,
            figure_id=figure_id,
        )
        return _json(result)

    @mcp.tool()
    async def plt_plot_violin(
        data: list,
        labels: Optional[list] = None,
        colors: Optional[list] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        panel: str = "0,0",
        figure_id: Optional[str] = None,
    ) -> str:
        """[plt] Create a violin plot on specified panel."""
        from scitex.plt._mcp._handlers_plot import plot_violin_handler

        result = await plot_violin_handler(
            data=data,
            labels=labels,
            colors=colors,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            panel=panel,
            figure_id=figure_id,
        )
        return _json(result)

    # Annotation tools
    @mcp.tool()
    async def plt_add_significance(
        x1: float,
        x2: float,
        y: float,
        text: str,
        height: Optional[float] = None,
        panel: str = "0,0",
        figure_id: Optional[str] = None,
    ) -> str:
        """[plt] Add significance bracket between two groups."""
        from scitex.plt._mcp._handlers_annotation import add_significance_handler

        result = await add_significance_handler(
            x1=x1,
            x2=x2,
            y=y,
            text=text,
            height=height,
            panel=panel,
            figure_id=figure_id,
        )
        return _json(result)

    @mcp.tool()
    async def plt_add_panel_label(
        label: str,
        x: float = -0.15,
        y: float = 1.1,
        fontsize: float = 10,
        fontweight: str = "bold",
        panel: str = "0,0",
        figure_id: Optional[str] = None,
    ) -> str:
        """[plt] Add panel label (A, B, C, etc.) to a panel."""
        from scitex.plt._mcp._handlers_annotation import add_panel_label_handler

        result = await add_panel_label_handler(
            label=label,
            x=x,
            y=y,
            fontsize=fontsize,
            fontweight=fontweight,
            panel=panel,
            figure_id=figure_id,
        )
        return _json(result)


# EOF
