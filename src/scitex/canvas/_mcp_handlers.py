#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/canvas/_mcp_handlers.py
# ----------------------------------------

"""
MCP Handler implementations for SciTeX canvas module.

Provides async handlers for multi-panel figure composition.
"""

from __future__ import annotations

import asyncio
from typing import Optional


async def create_canvas_handler(
    parent_dir: str,
    canvas_name: str,
    width_mm: float = 180,
    height_mm: float = 120,
) -> dict:
    """
    Create a new canvas workspace.

    Parameters
    ----------
    parent_dir : str
        Parent directory for canvas
    canvas_name : str
        Name for the canvas
    width_mm : float
        Canvas width in mm
    height_mm : float
        Canvas height in mm

    Returns
    -------
    dict
        Success status and canvas info
    """
    try:
        import scitex.canvas as canvas_module

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: canvas_module.create_canvas(
                parent_dir,
                canvas_name,
            ),
        )

        canvas_path = canvas_module.get_canvas_path(parent_dir, canvas_name)

        return {
            "success": True,
            "canvas_name": canvas_name,
            "canvas_path": str(canvas_path),
            "size_mm": {"width": width_mm, "height": height_mm},
            "message": f"Created canvas '{canvas_name}' at {canvas_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def add_panel_handler(
    parent_dir: str,
    canvas_name: str,
    panel_name: str,
    source: str,
    x_mm: float = 0,
    y_mm: float = 0,
    width_mm: float = 50,
    height_mm: float = 50,
    label: Optional[str] = None,
) -> dict:
    """
    Add a panel to a canvas.

    Parameters
    ----------
    parent_dir : str
        Parent directory containing canvas
    canvas_name : str
        Canvas name
    panel_name : str
        Name for this panel
    source : str
        Path to source file
    x_mm, y_mm : float
        Position in mm
    width_mm, height_mm : float
        Size in mm
    label : str, optional
        Panel label (A, B, C...)

    Returns
    -------
    dict
        Success status and panel info
    """
    try:
        import scitex.canvas as canvas_module

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: canvas_module.add_panel(
                parent_dir=parent_dir,
                canvas_name=canvas_name,
                panel_name=panel_name,
                source=source,
                xy_mm=(x_mm, y_mm),
                size_mm=(width_mm, height_mm),
                label=label or "",
            ),
        )

        return {
            "success": True,
            "canvas_name": canvas_name,
            "panel_name": panel_name,
            "source": source,
            "position_mm": {"x": x_mm, "y": y_mm},
            "size_mm": {"width": width_mm, "height": height_mm},
            "label": label,
            "message": f"Added panel '{panel_name}' to canvas '{canvas_name}'",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def list_panels_handler(
    parent_dir: str,
    canvas_name: str,
) -> dict:
    """
    List all panels in a canvas.

    Parameters
    ----------
    parent_dir : str
        Parent directory
    canvas_name : str
        Canvas name

    Returns
    -------
    dict
        Success status and panel list
    """
    try:
        import scitex.canvas as canvas_module

        loop = asyncio.get_event_loop()
        panels = await loop.run_in_executor(
            None,
            lambda: canvas_module.list_panels(parent_dir, canvas_name),
        )

        return {
            "success": True,
            "canvas_name": canvas_name,
            "count": len(panels) if panels else 0,
            "panels": panels or [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def remove_panel_handler(
    parent_dir: str,
    canvas_name: str,
    panel_name: str,
) -> dict:
    """
    Remove a panel from a canvas.

    Parameters
    ----------
    parent_dir : str
        Parent directory
    canvas_name : str
        Canvas name
    panel_name : str
        Panel to remove

    Returns
    -------
    dict
        Success status
    """
    try:
        import scitex.canvas as canvas_module

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: canvas_module.remove_panel(parent_dir, canvas_name, panel_name),
        )

        return {
            "success": True,
            "canvas_name": canvas_name,
            "panel_name": panel_name,
            "message": f"Removed panel '{panel_name}' from canvas '{canvas_name}'",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def export_canvas_handler(
    parent_dir: str,
    canvas_name: str,
    output_path: Optional[str] = None,
    format: Optional[str] = None,
    dpi: int = 300,
) -> dict:
    """
    Export canvas to file.

    Parameters
    ----------
    parent_dir : str
        Parent directory
    canvas_name : str
        Canvas name
    output_path : str, optional
        Output file path
    format : str, optional
        Output format (png, pdf, svg)
    dpi : int
        Output DPI

    Returns
    -------
    dict
        Success status and output path
    """
    try:
        from pathlib import Path

        import scitex.canvas as canvas_module

        # Build output path if not provided
        if output_path is None:
            canvas_path = canvas_module.get_canvas_path(parent_dir, canvas_name)
            fmt = format or "png"
            output_path = str(Path(canvas_path) / "exports" / f"{canvas_name}.{fmt}")

        loop = asyncio.get_event_loop()
        result_path = await loop.run_in_executor(
            None,
            lambda: canvas_module.export_canvas(
                parent_dir,
                canvas_name,
                output_path,
            ),
        )

        return {
            "success": True,
            "canvas_name": canvas_name,
            "output_path": str(result_path) if result_path else output_path,
            "format": format or Path(output_path).suffix.lstrip("."),
            "dpi": dpi,
            "message": f"Exported canvas to {output_path}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def list_canvases_handler(parent_dir: str) -> dict:
    """
    List all canvases in a directory.

    Parameters
    ----------
    parent_dir : str
        Directory to search

    Returns
    -------
    dict
        Success status and canvas list
    """
    try:
        import scitex.canvas as canvas_module

        loop = asyncio.get_event_loop()
        canvases = await loop.run_in_executor(
            None,
            lambda: canvas_module.list_canvases(parent_dir),
        )

        return {
            "success": True,
            "parent_dir": parent_dir,
            "count": len(canvases) if canvases else 0,
            "canvases": canvases or [],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def canvas_exists_handler(
    parent_dir: str,
    canvas_name: str,
) -> dict:
    """
    Check if a canvas exists.

    Parameters
    ----------
    parent_dir : str
        Parent directory
    canvas_name : str
        Canvas name

    Returns
    -------
    dict
        Success status and existence flag
    """
    try:
        import scitex.canvas as canvas_module

        exists = canvas_module.canvas_exists(parent_dir, canvas_name)

        return {
            "success": True,
            "canvas_name": canvas_name,
            "exists": exists,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


__all__ = [
    "create_canvas_handler",
    "add_panel_handler",
    "list_panels_handler",
    "remove_panel_handler",
    "export_canvas_handler",
    "list_canvases_handler",
    "canvas_exists_handler",
]

# EOF
