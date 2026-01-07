#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/diagram/_mcp_handlers.py
# ----------------------------------------

"""
MCP Handler implementations for SciTeX diagram module.

Provides async handlers for paper-optimized diagram generation.
"""

from __future__ import annotations

import asyncio
from typing import Optional


async def create_diagram_handler(
    spec_path: Optional[str] = None,
    spec_dict: Optional[dict] = None,
) -> dict:
    """
    Create a diagram from specification.

    Parameters
    ----------
    spec_path : str, optional
        Path to YAML spec file
    spec_dict : dict, optional
        Spec as dictionary

    Returns
    -------
    dict
        Success status and diagram info
    """
    try:
        from scitex.diagram import Diagram

        if spec_path:
            loop = asyncio.get_event_loop()
            diagram = await loop.run_in_executor(
                None,
                lambda: Diagram.from_yaml(spec_path),
            )
        elif spec_dict:
            diagram = Diagram(spec_dict)
        else:
            return {
                "success": False,
                "error": "Either spec_path or spec_dict must be provided",
            }

        # Get diagram info
        spec = diagram.spec if hasattr(diagram, "spec") else {}

        return {
            "success": True,
            "diagram_type": spec.get("type", "unknown"),
            "node_count": len(spec.get("nodes", [])),
            "edge_count": len(spec.get("edges", [])),
            "message": "Diagram created successfully",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def compile_mermaid_handler(
    spec_path: Optional[str] = None,
    output_path: Optional[str] = None,
    spec_dict: Optional[dict] = None,
) -> dict:
    """
    Compile diagram to Mermaid format.

    Parameters
    ----------
    spec_path : str, optional
        Path to YAML spec file
    output_path : str, optional
        Output .mmd file path
    spec_dict : dict, optional
        Spec as dictionary

    Returns
    -------
    dict
        Success status and Mermaid output
    """
    try:
        from scitex.diagram import Diagram, compile_to_mermaid

        if spec_path:
            loop = asyncio.get_event_loop()
            diagram = await loop.run_in_executor(
                None,
                lambda: Diagram.from_yaml(spec_path),
            )
        elif spec_dict:
            diagram = Diagram(spec_dict)
        else:
            return {
                "success": False,
                "error": "Either spec_path or spec_dict must be provided",
            }

        # Compile to Mermaid
        loop = asyncio.get_event_loop()
        mermaid_code = await loop.run_in_executor(
            None,
            lambda: compile_to_mermaid(diagram.spec),
        )

        # Save if output path provided
        if output_path:
            from pathlib import Path

            Path(output_path).write_text(mermaid_code)

        return {
            "success": True,
            "mermaid_code": mermaid_code,
            "output_path": output_path,
            "message": f"Compiled to Mermaid{f' and saved to {output_path}' if output_path else ''}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def compile_graphviz_handler(
    spec_path: Optional[str] = None,
    output_path: Optional[str] = None,
    spec_dict: Optional[dict] = None,
) -> dict:
    """
    Compile diagram to Graphviz DOT format.

    Parameters
    ----------
    spec_path : str, optional
        Path to YAML spec file
    output_path : str, optional
        Output .dot file path
    spec_dict : dict, optional
        Spec as dictionary

    Returns
    -------
    dict
        Success status and DOT output
    """
    try:
        from scitex.diagram import Diagram, compile_to_graphviz

        if spec_path:
            loop = asyncio.get_event_loop()
            diagram = await loop.run_in_executor(
                None,
                lambda: Diagram.from_yaml(spec_path),
            )
        elif spec_dict:
            diagram = Diagram(spec_dict)
        else:
            return {
                "success": False,
                "error": "Either spec_path or spec_dict must be provided",
            }

        # Compile to Graphviz
        loop = asyncio.get_event_loop()
        dot_code = await loop.run_in_executor(
            None,
            lambda: compile_to_graphviz(diagram.spec),
        )

        # Save if output path provided
        if output_path:
            from pathlib import Path

            Path(output_path).write_text(dot_code)

        return {
            "success": True,
            "dot_code": dot_code,
            "output_path": output_path,
            "message": f"Compiled to Graphviz{f' and saved to {output_path}' if output_path else ''}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def list_presets_handler() -> dict:
    """
    List available diagram presets.

    Returns
    -------
    dict
        Success status and preset list
    """
    try:
        presets = [
            {
                "name": "workflow",
                "description": "Linear workflow diagrams (step1 → step2 → step3)",
                "use_case": "Methods section, data processing pipelines",
                "direction": "left-to-right",
            },
            {
                "name": "decision",
                "description": "Decision tree/flowchart diagrams",
                "use_case": "Algorithm flowcharts, decision processes",
                "direction": "top-to-bottom",
            },
            {
                "name": "pipeline",
                "description": "Data pipeline with parallel branches",
                "use_case": "Complex data flows, parallel processing",
                "direction": "left-to-right",
            },
        ]

        return {
            "success": True,
            "count": len(presets),
            "presets": presets,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_preset_handler(preset_name: str) -> dict:
    """
    Get a specific preset configuration.

    Parameters
    ----------
    preset_name : str
        Preset name (workflow, decision, pipeline)

    Returns
    -------
    dict
        Success status and preset config
    """
    try:
        from scitex.diagram import DECISION_PRESET, PIPELINE_PRESET, WORKFLOW_PRESET

        presets = {
            "workflow": WORKFLOW_PRESET,
            "decision": DECISION_PRESET,
            "pipeline": PIPELINE_PRESET,
        }

        if preset_name not in presets:
            return {
                "success": False,
                "error": f"Unknown preset: {preset_name}",
                "available": list(presets.keys()),
            }

        preset = presets[preset_name]

        return {
            "success": True,
            "preset_name": preset_name,
            "config": preset,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def split_diagram_handler(
    spec_path: str,
    strategy: str = "horizontal",
    max_nodes_per_part: int = 10,
) -> dict:
    """
    Split a large diagram into smaller parts.

    Parameters
    ----------
    spec_path : str
        Path to YAML spec file
    strategy : str
        Split strategy (horizontal, vertical, semantic)
    max_nodes_per_part : int
        Max nodes per part

    Returns
    -------
    dict
        Success status and split results
    """
    try:
        from scitex.diagram import Diagram, SplitConfig, SplitStrategy, split_diagram

        # Map strategy string to enum
        strategy_map = {
            "horizontal": SplitStrategy.HORIZONTAL,
            "vertical": SplitStrategy.VERTICAL,
            "semantic": SplitStrategy.SEMANTIC,
        }

        if strategy not in strategy_map:
            return {
                "success": False,
                "error": f"Unknown strategy: {strategy}",
                "available": list(strategy_map.keys()),
            }

        loop = asyncio.get_event_loop()
        diagram = await loop.run_in_executor(
            None,
            lambda: Diagram.from_yaml(spec_path),
        )

        config = SplitConfig(
            strategy=strategy_map[strategy],
            max_nodes_per_part=max_nodes_per_part,
        )

        result = await loop.run_in_executor(
            None,
            lambda: split_diagram(diagram.spec, config),
        )

        return {
            "success": True,
            "strategy": strategy,
            "parts_count": len(result.parts) if hasattr(result, "parts") else 0,
            "result": str(result),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def get_paper_modes_handler() -> dict:
    """
    Get available paper layout modes.

    Returns
    -------
    dict
        Success status and paper modes
    """
    try:
        from scitex.diagram import PaperMode

        modes = []
        for mode in PaperMode:
            modes.append(
                {
                    "name": mode.name,
                    "value": mode.value,
                }
            )

        return {
            "success": True,
            "count": len(modes),
            "modes": modes,
            "description": "Paper modes control diagram sizing for publication layouts",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


__all__ = [
    "create_diagram_handler",
    "compile_mermaid_handler",
    "compile_graphviz_handler",
    "list_presets_handler",
    "get_preset_handler",
    "split_diagram_handler",
    "get_paper_modes_handler",
]

# EOF
