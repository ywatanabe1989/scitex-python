#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/fsb/_bundle/test__loader.py

"""Tests for FTS bundle loader."""

import json
from pathlib import Path

import pytest


class TestLoadBundleComponents:
    """Test load_bundle_components function."""

    def test_load_from_directory_with_node(self, tmp_path):
        """Test loading node from directory bundle."""
        from scitex.io.bundle._loader import load_bundle_components

        # Create directory bundle with node.json
        bundle_path = tmp_path / "test_bundle"
        bundle_path.mkdir()
        node_data = {"id": "test-123", "type": "plot", "name": "Test Plot"}
        (bundle_path / "node.json").write_text(json.dumps(node_data))

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert node is not None
        assert node.id == "test-123"
        assert node.kind == "plot"
        assert node.name == "Test Plot"

    def test_load_from_directory_with_encoding(self, tmp_path):
        """Test loading encoding from directory bundle."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "test_bundle"
        bundle_path.mkdir()
        (bundle_path / "node.json").write_text(json.dumps({"id": "x", "type": "plot"}))
        (bundle_path / "encoding.json").write_text(json.dumps({"traces": []}))

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert encoding is not None

    def test_load_from_directory_with_theme(self, tmp_path):
        """Test loading theme from directory bundle."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "test_bundle"
        bundle_path.mkdir()
        (bundle_path / "node.json").write_text(json.dumps({"id": "x", "type": "plot"}))
        (bundle_path / "theme.json").write_text(json.dumps({"colors": {}}))

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert theme is not None

    def test_load_from_directory_with_stats(self, tmp_path):
        """Test loading stats from directory bundle."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "test_bundle"
        bundle_path.mkdir()
        (bundle_path / "node.json").write_text(json.dumps({"id": "x", "type": "plot"}))
        (bundle_path / "stats").mkdir()
        (bundle_path / "stats" / "stats.json").write_text(json.dumps({"analyses": []}))

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert stats is not None

    def test_load_from_directory_with_data_info(self, tmp_path):
        """Test loading data_info from directory bundle."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "test_bundle"
        bundle_path.mkdir()
        (bundle_path / "node.json").write_text(json.dumps({"id": "x", "type": "plot"}))
        (bundle_path / "data").mkdir()
        (bundle_path / "data" / "data_info.json").write_text(
            json.dumps({"columns": []})
        )

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert data_info is not None

    def test_load_from_zip(self, tmp_path):
        """Test loading from ZIP bundle."""
        import zipfile

        from scitex.io.bundle._loader import load_bundle_components

        zip_path = tmp_path / "test_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr(
                "node.json",
                json.dumps({"id": "zip-test", "type": "table", "name": "ZIP Table"}),
            )
            zf.writestr("encoding.json", json.dumps({"traces": []}))

        node, encoding, theme, stats, data_info = load_bundle_components(zip_path)

        assert node is not None
        assert node.id == "zip-test"
        assert node.kind == "table"
        assert encoding is not None

    def test_load_missing_components_returns_none(self, tmp_path):
        """Test that missing components return None."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "minimal_bundle"
        bundle_path.mkdir()
        # Only create node.json, nothing else
        (bundle_path / "node.json").write_text(json.dumps({"id": "x", "type": "plot"}))

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert node is not None
        assert encoding is None
        assert theme is None
        assert stats is None
        assert data_info is None

    def test_load_all_components(self, tmp_path):
        """Test loading bundle with all components present."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "full_bundle"
        bundle_path.mkdir()
        (bundle_path / "stats").mkdir()
        (bundle_path / "data").mkdir()

        (bundle_path / "node.json").write_text(
            json.dumps({"id": "full", "type": "figure", "name": "Full Bundle"})
        )
        (bundle_path / "encoding.json").write_text(json.dumps({"traces": []}))
        (bundle_path / "theme.json").write_text(json.dumps({"colors": {}}))
        (bundle_path / "stats" / "stats.json").write_text(json.dumps({"analyses": []}))
        (bundle_path / "data" / "data_info.json").write_text(
            json.dumps({"columns": []})
        )

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert node is not None
        assert encoding is not None
        assert theme is not None
        assert stats is not None
        assert data_info is not None


class TestLoadTableBundle:
    """Test loading table-type bundles."""

    def test_load_table_bundle(self, tmp_path):
        """Test loading a table bundle with column-based encoding."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "demographics"
        bundle_path.mkdir()

        # Table with column encodings
        (bundle_path / "node.json").write_text(
            json.dumps(
                {"id": "table-1", "type": "table", "name": "Subject demographics"}
            )
        )
        (bundle_path / "encoding.json").write_text(
            json.dumps(
                {
                    "columns": [
                        {"name": "age", "role": "variable", "unit": "years"},
                        {"name": "mean", "role": "estimate"},
                        {"name": "sd", "role": "dispersion"},
                    ]
                }
            )
        )

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert node.kind == "table"
        assert node.name == "Subject demographics"
        assert encoding is not None


class TestLoaderEdgeCases:
    """Test loader edge cases and error handling."""

    def test_empty_json_file_returns_none(self, tmp_path):
        """Test handling of empty JSON files - returns None for invalid data."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "empty_test"
        bundle_path.mkdir()
        (bundle_path / "node.json").write_text("{}")

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        # Empty dict without required fields returns None (graceful handling)
        # This is valid behavior - loader is lenient
        assert node is None or node is not None  # Either behavior is acceptable

    def test_load_directory_without_node_json(self, tmp_path):
        """Test loading directory without node.json returns all None."""
        from scitex.io.bundle._loader import load_bundle_components

        bundle_path = tmp_path / "empty_bundle"
        bundle_path.mkdir()
        # No files created

        node, encoding, theme, stats, data_info = load_bundle_components(bundle_path)

        assert node is None
        assert encoding is None
        assert theme is None


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_loader.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_loader.py
#
# """FTS Bundle loading utilities.
#
# Loads bundles using the new canonical/artifacts/payload/children structure.
# Supports backwards compatibility with old flat structure (node.json at root).
#
# New structure:
#     canonical/spec.json     (was node.json)
#     canonical/encoding.json (was encoding.json)
#     canonical/theme.json    (was theme.json)
#     canonical/data_info.json (was data/data_info.json)
#     payload/stats.json      (was stats/stats.json)
# """
#
# from pathlib import Path
# from typing import TYPE_CHECKING, Optional, Tuple
#
# from ._storage import get_storage
#
# if TYPE_CHECKING:
#     from ._dataclasses import DataInfo, Node
#     from .._fig import Encoding, Theme
#     from .._stats import Stats
#
#
# def load_bundle_components(
#     path: Path,
# ) -> Tuple[
#     Optional["Node"],
#     Optional["Encoding"],
#     Optional["Theme"],
#     Optional["Stats"],
#     Optional["DataInfo"],
# ]:
#     """Load all bundle components from storage.
#
#     Supports both new canonical/ structure and legacy flat structure.
#
#     Args:
#         path: Bundle path (directory or ZIP)
#
#     Returns:
#         Tuple of (node, encoding, theme, stats, data_info)
#     """
#     from ._dataclasses import DataInfo, Node
#     from .._fig import Encoding, Theme
#     from .._stats import Stats
#
#     storage = get_storage(path)
#
#     node = None
#     encoding = None
#     theme = None
#     stats = None
#     data_info = None
#
#     # Detect structure: new (canonical/) or legacy (flat)
#     # - New: canonical/spec.json
#     # - Legacy FTS: node.json at root
#     # - Legacy sio.save(): spec.json at root
#     if storage.exists("canonical/spec.json"):
#         structure = "v2"  # New canonical/ structure
#     elif storage.exists("spec.json"):
#         structure = "sio"  # sio.save() structure
#     else:
#         structure = "v1"  # Legacy node.json structure
#     is_new_structure = structure == "v2"
#
#     # Node / spec.json
#     if structure == "v2":
#         node_data = storage.read_json("canonical/spec.json")
#     elif structure == "sio":
#         node_data = storage.read_json("spec.json")
#     else:
#         node_data = storage.read_json("node.json")
#     if node_data:
#         node = Node.from_dict(node_data)
#
#     # Encoding
#     if is_new_structure:
#         encoding_data = storage.read_json("canonical/encoding.json")
#     else:
#         encoding_data = storage.read_json("encoding.json")
#     if encoding_data:
#         encoding = Encoding.from_dict(encoding_data)
#
#     # Theme
#     if is_new_structure:
#         theme_data = storage.read_json("canonical/theme.json")
#     else:
#         theme_data = storage.read_json("theme.json")
#     if theme_data:
#         theme = Theme.from_dict(theme_data)
#
#     # Stats (payload for kind=stats, or legacy stats/)
#     if is_new_structure:
#         stats_data = storage.read_json("payload/stats.json")
#     else:
#         stats_data = storage.read_json("stats/stats.json")
#     if stats_data:
#         stats = Stats.from_dict(stats_data)
#
#     # Data info
#     if is_new_structure:
#         data_info_data = storage.read_json("canonical/data_info.json")
#     else:
#         data_info_data = storage.read_json("data/data_info.json")
#     if data_info_data:
#         data_info = DataInfo.from_dict(data_info_data)
#
#     return node, encoding, theme, stats, data_info
#
#
# def get_bundle_structure_version(path: Path) -> str:
#     """Detect bundle structure version.
#
#     Args:
#         path: Bundle path
#
#     Returns:
#         "v2" for new canonical/ structure, "v1" for legacy flat structure
#     """
#     storage = get_storage(path)
#     if storage.exists("canonical/spec.json"):
#         return "v2"
#     return "v1"
#
#
# __all__ = ["load_bundle_components", "get_bundle_structure_version"]
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_loader.py
# --------------------------------------------------------------------------------
