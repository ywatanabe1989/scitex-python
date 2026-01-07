#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/io/bundle/_dataclasses/test__Spec.py

"""Tests for Spec dataclass."""

import json
from datetime import datetime

import pytest


class TestSpecCreation:
    """Test Spec instantiation."""

    def test_create_minimal_spec(self):
        """Test creating spec with minimal required fields."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="test-1", kind="plot")

        assert spec.id == "test-1"
        assert spec.kind == "plot"
        assert spec.bbox_norm is not None
        assert spec.created_at is not None
        assert spec.modified_at is not None

    def test_create_plot_spec(self):
        """Test creating a plot-type spec."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="plot-1", kind="plot", name="Line Plot")

        assert spec.kind == "plot"
        assert spec.name == "Line Plot"

    def test_create_table_spec(self):
        """Test creating a table-type spec."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="table-1", kind="table", name="Demographics Table")

        assert spec.kind == "table"
        assert spec.name == "Demographics Table"

    def test_create_figure_spec(self):
        """Test creating a figure-type spec (container)."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(
            id="fig-1",
            kind="figure",
            name="Multi-panel Figure",
            children=["panel-a", "panel-b"],
        )

        assert spec.kind == "figure"
        assert len(spec.children) == 2
        assert "panel-a" in spec.children

    def test_create_with_size_mm(self):
        """Test creating spec with size specification."""
        from scitex.io.bundle._dataclasses import SizeMM, Spec

        size = SizeMM(width=80, height=60)
        spec = Spec(id="sized-1", kind="plot", size_mm=size)

        assert spec.size_mm is not None
        assert spec.size_mm.width == 80
        assert spec.size_mm.height == 60

    def test_timestamp_auto_generation(self):
        """Test that timestamps are auto-generated."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="ts-test", kind="plot")

        assert spec.created_at is not None
        assert spec.modified_at is not None
        assert spec.created_at == spec.modified_at


class TestSpecSerialization:
    """Test Spec to_dict and from_dict."""

    def test_to_dict_minimal(self):
        """Test to_dict with minimal spec."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="test-1", kind="plot")
        d = spec.to_dict()

        assert d["id"] == "test-1"
        assert d["kind"] == "plot"
        assert "bbox_norm" in d
        assert "created_at" in d
        assert "modified_at" in d

    def test_to_dict_with_name(self):
        """Test to_dict includes name when set."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="test-1", kind="table", name="Test Table")
        d = spec.to_dict()

        assert d["name"] == "Test Table"

    def test_to_dict_with_children(self):
        """Test to_dict includes children."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="fig-1", kind="figure", children=["a", "b", "c"])
        d = spec.to_dict()

        assert d["children"] == ["a", "b", "c"]

    def test_to_dict_with_size_mm(self):
        """Test to_dict includes size_mm."""
        from scitex.io.bundle._dataclasses import SizeMM, Spec

        spec = Spec(id="test-1", kind="plot", size_mm=SizeMM(width=100, height=80))
        d = spec.to_dict()

        assert "size_mm" in d
        assert d["size_mm"]["width"] == 100
        assert d["size_mm"]["height"] == 80

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        from scitex.io.bundle._dataclasses import Spec

        data = {"id": "from-dict-1", "kind": "plot"}
        spec = Spec.from_dict(data)

        assert spec.id == "from-dict-1"
        assert spec.kind == "plot"

    def test_from_dict_with_all_fields(self):
        """Test from_dict with all fields."""
        from scitex.io.bundle._dataclasses import Spec

        data = {
            "id": "full-spec",
            "kind": "figure",
            "name": "Full Table",
            "bbox_norm": {"x": 0.1, "y": 0.1, "width": 0.8, "height": 0.8},
            "size_mm": {"width": 170, "height": 120},
            "children": ["child-1", "child-2"],
            "refs": {"data": "data.csv", "stats": "stats.json"},
            "created_at": "2025-12-20T00:00:00Z",
            "modified_at": "2025-12-20T01:00:00Z",
        }
        spec = Spec.from_dict(data)

        assert spec.id == "full-spec"
        assert spec.kind == "figure"
        assert spec.name == "Full Table"
        assert spec.children == ["child-1", "child-2"]
        assert spec.created_at == "2025-12-20T00:00:00Z"

    def test_roundtrip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        from scitex.io.bundle._dataclasses import SizeMM, Spec

        original = Spec(
            id="roundtrip-1",
            kind="figure",
            name="Roundtrip Test",
            size_mm=SizeMM(width=80, height=60),
            children=["a", "b"],
        )

        d = original.to_dict()
        restored = Spec.from_dict(d)

        assert restored.id == original.id
        assert restored.kind == original.kind
        assert restored.name == original.name
        assert restored.children == original.children


class TestSpecTouch:
    """Test Spec touch method."""

    def test_touch_updates_modified(self):
        """Test that touch updates modified_at."""
        import time

        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="touch-test", kind="plot")
        original_modified = spec.modified_at

        time.sleep(0.01)  # Small delay to ensure different timestamp
        spec.touch()

        assert spec.modified_at != original_modified
        assert spec.created_at != spec.modified_at


class TestSpecKinds:
    """Test various spec kinds."""

    def test_all_valid_kinds(self):
        """Test creating specs with all valid kinds."""
        from scitex.io.bundle._dataclasses import Spec

        valid_kinds = [
            "figure",
            "plot",
            "table",
            "stats",
            "image",
            "text",
            "shape",
        ]

        for spec_kind in valid_kinds:
            spec = Spec(id=f"{spec_kind}-test", kind=spec_kind)
            assert spec.kind == spec_kind

    def test_table_kind_integration(self):
        """Test table kind works with full workflow."""
        from scitex.io.bundle._dataclasses import Spec

        # Create table spec
        table_spec = Spec(
            id="demographics",
            kind="table",
            name="Subject Demographics",
        )

        # Serialize
        d = table_spec.to_dict()
        assert d["kind"] == "table"

        # Deserialize
        restored = Spec.from_dict(d)
        assert restored.kind == "table"
        assert restored.name == "Subject Demographics"


class TestSpecBBox:
    """Test Spec bounding box handling."""

    def test_default_bbox(self):
        """Test default bounding box values."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="bbox-test", kind="plot")

        assert spec.bbox_norm is not None
        # BBox uses x0, y0, x1, y1 format
        assert hasattr(spec.bbox_norm, "x0")
        assert hasattr(spec.bbox_norm, "y0")
        assert hasattr(spec.bbox_norm, "x1")
        assert hasattr(spec.bbox_norm, "y1")

    def test_custom_bbox(self):
        """Test custom bounding box."""
        from scitex.io.bundle._dataclasses import BBox, Spec

        # BBox uses x0, y0, x1, y1 format
        bbox = BBox(x0=0.1, y0=0.2, x1=0.7, y1=0.7)
        spec = Spec(id="bbox-test", kind="plot", bbox_norm=bbox)

        assert spec.bbox_norm.x0 == 0.1
        assert spec.bbox_norm.y0 == 0.2
        assert spec.bbox_norm.x1 == 0.7
        assert spec.bbox_norm.y1 == 0.7


class TestSpecRefs:
    """Test Spec refs handling."""

    def test_default_refs(self):
        """Test default refs."""
        from scitex.io.bundle._dataclasses import Spec

        spec = Spec(id="refs-test", kind="plot")

        assert spec.refs is not None

    def test_refs_from_dict(self):
        """Test refs from dictionary."""
        from scitex.io.bundle._dataclasses import Spec

        data = {
            "id": "refs-test",
            "kind": "table",
            "refs": {
                "data": "data/table1.csv",
                "stats": "stats/table1_stats.json",
            },
        }
        spec = Spec.from_dict(data)

        # Refs should be parsed
        assert spec.refs is not None


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF
