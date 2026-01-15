#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/diagram.py
"""Diagram module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import Optional


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def register_diagram_tools(mcp) -> None:
    """Register diagram tools with FastMCP server."""

    @mcp.tool()
    async def diagram_create_diagram(
        spec_path: Optional[str] = None,
        spec_dict: Optional[dict] = None,
    ) -> str:
        """[diagram] Create a diagram from a YAML specification file or dictionary."""
        from scitex.diagram._mcp.handlers import create_diagram_handler

        result = await create_diagram_handler(
            spec_path=spec_path,
            spec_dict=spec_dict,
        )
        return _json(result)

    @mcp.tool()
    async def diagram_compile_mermaid(
        spec_path: Optional[str] = None,
        output_path: Optional[str] = None,
        spec_dict: Optional[dict] = None,
    ) -> str:
        """[diagram] Compile diagram specification to Mermaid format."""
        from scitex.diagram._mcp.handlers import compile_mermaid_handler

        result = await compile_mermaid_handler(
            spec_path=spec_path,
            output_path=output_path,
            spec_dict=spec_dict,
        )
        return _json(result)

    @mcp.tool()
    async def diagram_compile_graphviz(
        spec_path: Optional[str] = None,
        output_path: Optional[str] = None,
        spec_dict: Optional[dict] = None,
    ) -> str:
        """[diagram] Compile diagram specification to Graphviz DOT format."""
        from scitex.diagram._mcp.handlers import compile_graphviz_handler

        result = await compile_graphviz_handler(
            spec_path=spec_path,
            output_path=output_path,
            spec_dict=spec_dict,
        )
        return _json(result)

    @mcp.tool()
    async def diagram_list_presets() -> str:
        """[diagram] List available diagram presets (workflow, decision, pipeline)."""
        from scitex.diagram._mcp.handlers import list_presets_handler

        result = await list_presets_handler()
        return _json(result)

    @mcp.tool()
    async def diagram_get_preset(preset_name: str) -> str:
        """[diagram] Get a diagram preset configuration by name."""
        from scitex.diagram._mcp.handlers import get_preset_handler

        result = await get_preset_handler(preset_name=preset_name)
        return _json(result)

    @mcp.tool()
    async def diagram_split_diagram(
        spec_path: str,
        strategy: str = "horizontal",
        max_nodes_per_part: int = 10,
    ) -> str:
        """[diagram] Split a large diagram into smaller parts for multi-column layouts."""
        from scitex.diagram._mcp.handlers import split_diagram_handler

        result = await split_diagram_handler(
            spec_path=spec_path,
            strategy=strategy,
            max_nodes_per_part=max_nodes_per_part,
        )
        return _json(result)

    @mcp.tool()
    async def diagram_get_paper_modes() -> str:
        """[diagram] Get available paper layout modes and their constraints."""
        from scitex.diagram._mcp.handlers import get_paper_modes_handler

        result = await get_paper_modes_handler()
        return _json(result)


# EOF
