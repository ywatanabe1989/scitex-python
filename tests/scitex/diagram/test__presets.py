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
    pytest.main([__file__, "-v"])
