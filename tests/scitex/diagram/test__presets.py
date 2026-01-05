#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for scitex.diagram._presets"""

import pytest
from scitex.diagram._presets import (
    DiagramPreset, WORKFLOW_PRESET, DECISION_PRESET, PIPELINE_PRESET, get_preset
)


class TestDiagramPreset:
    def test_preset_has_required_attributes(self):
        preset = WORKFLOW_PRESET
        assert hasattr(preset, "mermaid_direction")
        assert hasattr(preset, "mermaid_theme")
        assert hasattr(preset, "graphviz_rankdir")
        assert hasattr(preset, "graphviz_ranksep")
        assert hasattr(preset, "graphviz_nodesep")
        assert hasattr(preset, "spacing_map")
        assert hasattr(preset, "mermaid_shapes")
        assert hasattr(preset, "graphviz_shapes")
        assert hasattr(preset, "emphasis_styles")


class TestWorkflowPreset:
    def test_direction_is_lr(self):
        assert WORKFLOW_PRESET.mermaid_direction == "LR"
        assert WORKFLOW_PRESET.graphviz_rankdir == "LR"

    def test_has_tight_spacing(self):
        assert "tight" in WORKFLOW_PRESET.spacing_map
        tight = WORKFLOW_PRESET.spacing_map["tight"]
        assert tight["ranksep"] == 0.3
        assert tight["nodesep"] == 0.2

    def test_has_all_shapes(self):
        shapes = ["box", "rounded", "diamond", "circle", "stadium"]
        for shape in shapes:
            assert shape in WORKFLOW_PRESET.mermaid_shapes
            assert shape in WORKFLOW_PRESET.graphviz_shapes

    def test_has_emphasis_styles(self):
        styles = ["primary", "success", "warning", "muted", "normal"]
        for style in styles:
            assert style in WORKFLOW_PRESET.emphasis_styles


class TestDecisionPreset:
    def test_direction_is_tb(self):
        assert DECISION_PRESET.mermaid_direction == "TB"
        assert DECISION_PRESET.graphviz_rankdir == "TB"

    def test_wider_spacing_than_workflow(self):
        assert DECISION_PRESET.graphviz_ranksep > WORKFLOW_PRESET.graphviz_ranksep


class TestPipelinePreset:
    def test_direction_is_lr(self):
        assert PIPELINE_PRESET.mermaid_direction == "LR"
        assert PIPELINE_PRESET.graphviz_rankdir == "LR"

    def test_larger_ranksep(self):
        assert PIPELINE_PRESET.graphviz_ranksep > WORKFLOW_PRESET.graphviz_ranksep


class TestGetPreset:
    def test_get_workflow_preset(self):
        preset = get_preset("workflow")
        assert preset is WORKFLOW_PRESET

    def test_get_decision_preset(self):
        preset = get_preset("decision")
        assert preset is DECISION_PRESET

    def test_get_pipeline_preset(self):
        preset = get_preset("pipeline")
        assert preset is PIPELINE_PRESET

    def test_get_hierarchy_returns_decision(self):
        preset = get_preset("hierarchy")
        assert preset is DECISION_PRESET

    def test_get_comparison_returns_workflow(self):
        preset = get_preset("comparison")
        assert preset is WORKFLOW_PRESET

    def test_case_insensitive(self):
        assert get_preset("WORKFLOW") is WORKFLOW_PRESET
        assert get_preset("Decision") is DECISION_PRESET

    def test_unknown_type_returns_workflow(self):
        preset = get_preset("unknown_type")
        assert preset is WORKFLOW_PRESET


class TestMermaidShapes:
    def test_label_placeholder(self):
        for shape, template in WORKFLOW_PRESET.mermaid_shapes.items():
            assert "__LABEL__" in template

    def test_box_shape(self):
        assert WORKFLOW_PRESET.mermaid_shapes["box"] == '["__LABEL__"]'

    def test_rounded_shape(self):
        assert WORKFLOW_PRESET.mermaid_shapes["rounded"] == '("__LABEL__")'

    def test_diamond_shape(self):
        assert WORKFLOW_PRESET.mermaid_shapes["diamond"] == '{"__LABEL__"}'


class TestEmphasisStyles:
    def test_primary_has_fill(self):
        assert "fill" in WORKFLOW_PRESET.emphasis_styles["primary"]

    def test_primary_has_stroke(self):
        assert "stroke" in WORKFLOW_PRESET.emphasis_styles["primary"]

    def test_all_presets_have_same_emphasis_keys(self):
        workflow_keys = set(WORKFLOW_PRESET.emphasis_styles.keys())
        decision_keys = set(DECISION_PRESET.emphasis_styles.keys())
        pipeline_keys = set(PIPELINE_PRESET.emphasis_styles.keys())
        assert workflow_keys == decision_keys == pipeline_keys

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_presets.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: 2025-12-15
# # Author: ywatanabe / Claude
# # File: scitex/diagram/_presets.py
# 
# """
# Presets for common diagram types in scientific papers.
# 
# Each preset defines rules for compiling the semantic spec to backend formats.
# These encode domain knowledge about "what makes a good paper figure."
# """
# 
# from dataclasses import dataclass
# from typing import Dict, Any, List
# 
# 
# @dataclass
# class DiagramPreset:
#     """Rules for compiling a diagram type."""
# 
#     # Mermaid settings
#     mermaid_direction: str  # TB, LR, RL, BT
#     mermaid_theme: Dict[str, str]
# 
#     # Graphviz settings
#     graphviz_rankdir: str  # TB, LR, RL, BT
#     graphviz_ranksep: float
#     graphviz_nodesep: float
# 
#     # Spacing mappings
#     spacing_map: Dict[str, Dict[str, float]]
# 
#     # Shape mappings (semantic -> backend)
#     mermaid_shapes: Dict[str, str]
#     graphviz_shapes: Dict[str, str]
# 
#     # Emphasis styles (colors, borders)
#     emphasis_styles: Dict[str, Dict[str, str]]
# 
# 
# # Workflow preset: sequential processes, emphasize flow
# WORKFLOW_PRESET = DiagramPreset(
#     mermaid_direction="LR",  # Left-to-right for workflows
#     mermaid_theme={
#         "primaryColor": "#1a2634",
#         "primaryTextColor": "#e0e0e0",
#         "primaryBorderColor": "#3a4a5a",
#         "lineColor": "#5a9fcf",
#     },
#     graphviz_rankdir="LR",
#     graphviz_ranksep=0.8,
#     graphviz_nodesep=0.5,
#     spacing_map={
#         "tight": {"ranksep": 0.3, "nodesep": 0.2},      # Publication: minimal
#         "compact": {"ranksep": 0.4, "nodesep": 0.3},
#         "medium": {"ranksep": 0.8, "nodesep": 0.5},
#         "large": {"ranksep": 1.2, "nodesep": 0.8},
#     },
#     mermaid_shapes={
#         "box": '["__LABEL__"]',
#         "rounded": '("__LABEL__")',
#         "diamond": '{"__LABEL__"}',
#         "circle": '(("__LABEL__"))',
#         "stadium": '(["__LABEL__"])',
#     },
#     graphviz_shapes={
#         "box": "box",
#         "rounded": "box",  # with style=rounded
#         "diamond": "diamond",
#         "circle": "circle",
#         "stadium": "box",  # with style=rounded
#     },
#     emphasis_styles={
#         "primary": {"fill": "#0d4a6b", "stroke": "#5a9fcf", "stroke-width": "2px"},
#         "success": {"fill": "#ccffcc", "stroke": "#00cc00"},
#         "warning": {"fill": "#ffcccc", "stroke": "#cc0000"},
#         "muted": {"fill": "#f0f0f0", "stroke": "#999999"},
#         "normal": {"fill": "#1a2634", "stroke": "#3a4a5a"},
#     },
# )
# 
# # Decision tree preset: top-to-bottom with branches
# DECISION_PRESET = DiagramPreset(
#     mermaid_direction="TB",
#     mermaid_theme={
#         "primaryColor": "#f5f5f5",
#         "primaryTextColor": "#333333",
#         "primaryBorderColor": "#666666",
#         "lineColor": "#666666",
#     },
#     graphviz_rankdir="TB",
#     graphviz_ranksep=1.0,
#     graphviz_nodesep=0.6,
#     spacing_map={
#         "compact": {"ranksep": 0.6, "nodesep": 0.4},
#         "medium": {"ranksep": 1.0, "nodesep": 0.6},
#         "large": {"ranksep": 1.5, "nodesep": 1.0},
#     },
#     mermaid_shapes={
#         "box": '["__LABEL__"]',
#         "rounded": '("__LABEL__")',
#         "diamond": '{"__LABEL__"}',
#         "circle": '(("__LABEL__"))',
#         "stadium": '(["__LABEL__"])',
#     },
#     graphviz_shapes={
#         "box": "box",
#         "rounded": "box",
#         "diamond": "diamond",
#         "circle": "circle",
#         "stadium": "box",
#     },
#     emphasis_styles={
#         "primary": {"fill": "#e6f3ff", "stroke": "#0066cc", "stroke-width": "2px"},
#         "success": {"fill": "#e6ffe6", "stroke": "#00aa00"},
#         "warning": {"fill": "#ffe6e6", "stroke": "#cc0000"},
#         "muted": {"fill": "#f0f0f0", "stroke": "#aaaaaa"},
#         "normal": {"fill": "#ffffff", "stroke": "#666666"},
#     },
# )
# 
# # Pipeline preset: strict horizontal stages
# PIPELINE_PRESET = DiagramPreset(
#     mermaid_direction="LR",
#     mermaid_theme={
#         "primaryColor": "#ffffff",
#         "primaryTextColor": "#333333",
#         "primaryBorderColor": "#0066cc",
#         "lineColor": "#0066cc",
#     },
#     graphviz_rankdir="LR",
#     graphviz_ranksep=1.2,
#     graphviz_nodesep=0.4,
#     spacing_map={
#         "compact": {"ranksep": 0.8, "nodesep": 0.3},
#         "medium": {"ranksep": 1.2, "nodesep": 0.4},
#         "large": {"ranksep": 1.8, "nodesep": 0.6},
#     },
#     mermaid_shapes={
#         "box": '["__LABEL__"]',
#         "rounded": '("__LABEL__")',
#         "diamond": '{"__LABEL__"}',
#         "circle": '(("__LABEL__"))',
#         "stadium": '(["__LABEL__"])',
#     },
#     graphviz_shapes={
#         "box": "box",
#         "rounded": "box",
#         "diamond": "diamond",
#         "circle": "circle",
#         "stadium": "box",
#     },
#     emphasis_styles={
#         "primary": {"fill": "#e6f0ff", "stroke": "#0044aa", "stroke-width": "2px"},
#         "success": {"fill": "#e6ffe6", "stroke": "#00aa00"},
#         "warning": {"fill": "#fff3e6", "stroke": "#ff8800"},
#         "muted": {"fill": "#f5f5f5", "stroke": "#cccccc"},
#         "normal": {"fill": "#ffffff", "stroke": "#0066cc"},
#     },
# )
# 
# 
# def get_preset(diagram_type: str) -> DiagramPreset:
#     """Get preset for diagram type."""
#     presets = {
#         "workflow": WORKFLOW_PRESET,
#         "decision": DECISION_PRESET,
#         "pipeline": PIPELINE_PRESET,
#         "hierarchy": DECISION_PRESET,  # Same as decision for now
#         "comparison": WORKFLOW_PRESET,  # Override direction in compiler
#     }
#     return presets.get(diagram_type.lower(), WORKFLOW_PRESET)

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/diagram/_presets.py
# --------------------------------------------------------------------------------
