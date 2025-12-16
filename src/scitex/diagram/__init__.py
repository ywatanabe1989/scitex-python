#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-15
# Author: ywatanabe / Claude
# File: scitex/diagram/__init__.py

"""
SciTeX Diagram - Paper-optimized diagram generation.

This module provides a semantic layer above Mermaid/Graphviz/D2 that
understands paper constraints (column width, reading direction, emphasis)
and compiles to backend formats with appropriate layout hints.

Key insight: LLMs are good at generating CONSTRAINTS, not pixel layouts.
SciTeX Diagram defines "what this diagram means for a paper" and compiles
that to backend-specific layout directives.

Example
-------
>>> from scitex.diagram import Diagram
>>>
>>> diagram = Diagram.from_yaml("workflow.diagram.yaml")
>>> diagram.to_mermaid("workflow.mmd")
>>> diagram.to_graphviz("workflow.dot")
"""

from scitex.diagram._schema import DiagramSpec, PaperConstraints, LayoutHints, PaperMode
from scitex.diagram._diagram import Diagram
from scitex.diagram._compile import compile_to_mermaid, compile_to_graphviz
from scitex.diagram._presets import WORKFLOW_PRESET, DECISION_PRESET, PIPELINE_PRESET
from scitex.diagram._split import split_diagram, SplitConfig, SplitStrategy, SplitResult

__all__ = [
    "Diagram",
    "DiagramSpec",
    "PaperConstraints",
    "LayoutHints",
    "PaperMode",
    "compile_to_mermaid",
    "compile_to_graphviz",
    "WORKFLOW_PRESET",
    "DECISION_PRESET",
    "PIPELINE_PRESET",
    "split_diagram",
    "SplitConfig",
    "SplitStrategy",
    "SplitResult",
]
