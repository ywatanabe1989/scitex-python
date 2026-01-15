#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/canvas.py
"""Canvas module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import Optional


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def register_canvas_tools(mcp) -> None:
    """Register canvas tools with FastMCP server."""

    @mcp.tool()
    async def canvas_create_canvas(
        parent_dir: str,
        canvas_name: str,
        width_mm: float = 180,
        height_mm: float = 120,
    ) -> str:
        """[canvas] Create a new paper figure canvas workspace."""
        from scitex.canvas._mcp.handlers import create_canvas_handler

        result = await create_canvas_handler(
            parent_dir=parent_dir,
            canvas_name=canvas_name,
            width_mm=width_mm,
            height_mm=height_mm,
        )
        return _json(result)

    @mcp.tool()
    async def canvas_add_panel(
        parent_dir: str,
        canvas_name: str,
        panel_name: str,
        source: str,
        x_mm: float = 0,
        y_mm: float = 0,
        width_mm: float = 50,
        height_mm: float = 50,
        label: Optional[str] = None,
    ) -> str:
        """[canvas] Add a panel to an existing canvas from an image or plot."""
        from scitex.canvas._mcp.handlers import add_panel_handler

        result = await add_panel_handler(
            parent_dir=parent_dir,
            canvas_name=canvas_name,
            panel_name=panel_name,
            source=source,
            x_mm=x_mm,
            y_mm=y_mm,
            width_mm=width_mm,
            height_mm=height_mm,
            label=label,
        )
        return _json(result)

    @mcp.tool()
    async def canvas_list_panels(parent_dir: str, canvas_name: str) -> str:
        """[canvas] List all panels in a canvas with their properties."""
        from scitex.canvas._mcp.handlers import list_panels_handler

        result = await list_panels_handler(
            parent_dir=parent_dir, canvas_name=canvas_name
        )
        return _json(result)

    @mcp.tool()
    async def canvas_remove_panel(
        parent_dir: str, canvas_name: str, panel_name: str
    ) -> str:
        """[canvas] Remove a panel from a canvas."""
        from scitex.canvas._mcp.handlers import remove_panel_handler

        result = await remove_panel_handler(
            parent_dir=parent_dir,
            canvas_name=canvas_name,
            panel_name=panel_name,
        )
        return _json(result)

    @mcp.tool()
    async def canvas_export_canvas(
        parent_dir: str,
        canvas_name: str,
        output_path: Optional[str] = None,
        format: Optional[str] = None,
        dpi: int = 300,
    ) -> str:
        """[canvas] Export/render canvas to PNG, PDF, or SVG format."""
        from scitex.canvas._mcp.handlers import export_canvas_handler

        result = await export_canvas_handler(
            parent_dir=parent_dir,
            canvas_name=canvas_name,
            output_path=output_path,
            format=format,
            dpi=dpi,
        )
        return _json(result)

    @mcp.tool()
    async def canvas_list_canvases(parent_dir: str) -> str:
        """[canvas] List all canvases in a directory."""
        from scitex.canvas._mcp.handlers import list_canvases_handler

        result = await list_canvases_handler(parent_dir=parent_dir)
        return _json(result)

    @mcp.tool()
    async def canvas_canvas_exists(parent_dir: str, canvas_name: str) -> str:
        """[canvas] Check if a canvas exists."""
        from scitex.canvas._mcp.handlers import canvas_exists_handler

        result = await canvas_exists_handler(
            parent_dir=parent_dir, canvas_name=canvas_name
        )
        return _json(result)


# EOF
