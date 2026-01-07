#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: tests/scitex/compat/test_deprecation_warnings.py

"""Tests for deprecated module imports and backward compatibility."""

import importlib
import sys
import warnings

import pytest


class TestFigDeprecation:
    """Test scitex.fig deprecation (-> scitex.canvas)."""

    def test_fig_import_shows_deprecation_warning(self):
        """Importing scitex.fig should show DeprecationWarning."""
        # Remove from cache to ensure fresh import
        modules_to_remove = [k for k in sys.modules if k.startswith("scitex.fig")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import scitex.fig

            # Check that a deprecation warning was issued
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "scitex.fig is deprecated" in str(deprecation_warnings[0].message)
            assert "scitex.canvas" in str(deprecation_warnings[0].message)

    def test_fig_exports_canvas_classes(self):
        """scitex.fig should export Canvas class from scitex.canvas."""
        import scitex.fig

        assert hasattr(scitex.fig, "Canvas")

    def test_fig_canvas_is_same_class(self):
        """scitex.fig.Canvas should be same as scitex.canvas.Canvas."""
        import scitex.canvas
        import scitex.fig

        assert scitex.fig.Canvas is scitex.canvas.Canvas


class TestFtsDeprecation:
    """Test scitex.fts deprecation (-> scitex.io.bundle)."""

    def test_fts_import_shows_deprecation_warning(self):
        """Importing scitex.fts should show DeprecationWarning."""
        # Remove from cache to ensure fresh import
        modules_to_remove = [k for k in sys.modules if k.startswith("scitex.fts")]
        for mod in modules_to_remove:
            del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import scitex.fts

            # Check that a deprecation warning was issued
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "scitex.fts is deprecated" in str(deprecation_warnings[0].message)
            assert "scitex.io.bundle" in str(deprecation_warnings[0].message)

    def test_fts_exports_fts_class(self):
        """scitex.fts should export FTS class."""
        import scitex.fts

        assert hasattr(scitex.fts, "FTS")
        assert scitex.fts.FTS_AVAILABLE is True

    def test_fts_exports_dataclasses(self):
        """scitex.fts should export Node, BBox, etc."""
        import scitex.fts

        assert hasattr(scitex.fts, "Node")
        assert hasattr(scitex.fts, "BBox")
        assert hasattr(scitex.fts, "SizeMM")
        assert hasattr(scitex.fts, "DataInfo")

    def test_fts_class_is_same_as_bundle(self):
        """scitex.fts.FTS should be same as scitex.io.bundle.FTS."""
        import scitex.fts
        import scitex.io.bundle

        assert scitex.fts.FTS is scitex.io.bundle.FTS

    def test_fts_legacy_aliases(self):
        """scitex.fts should have legacy FSB aliases."""
        import scitex.fts

        assert hasattr(scitex.fts, "FSB")
        assert hasattr(scitex.fts, "FSB_AVAILABLE")
        assert hasattr(scitex.fts, "FSB_VERSION")
        assert scitex.fts.FSB is scitex.fts.FTS


class TestNewImports:
    """Test that new import paths work correctly."""

    def test_canvas_import(self):
        """scitex.canvas should import without deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from scitex.canvas import Canvas

            # Filter for our deprecation warnings (not third-party)
            our_warnings = [
                x
                for x in w
                if "scitex" in str(x.message)
                and issubclass(x.category, DeprecationWarning)
            ]
            assert len(our_warnings) == 0

    def test_io_bundle_import(self):
        """scitex.io.bundle should import without deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from scitex.io.bundle import FTS, Node, load, save

            # Filter for our deprecation warnings (not third-party)
            our_warnings = [
                x
                for x in w
                if "scitex" in str(x.message)
                and issubclass(x.category, DeprecationWarning)
            ]
            assert len(our_warnings) == 0

    def test_io_bundle_exports_fts_class(self):
        """scitex.io.bundle should export FTS class and related items."""
        from scitex.io.bundle import (
            FTS,
            BBox,
            DataInfo,
            Node,
            NodeType,
            SizeMM,
            create_bundle,
            from_matplotlib,
            load_bundle,
        )

        assert FTS is not None
        assert Node is not None
        assert BBox is not None


class TestExtensionAliases:
    """Test new extension format support."""

    def test_extension_constants(self):
        """io.bundle should have extension constants."""
        from scitex.io.bundle import (
            EXTENSION_MAP,
            EXTENSIONS,
            EXTENSIONS_LEGACY,
            EXTENSIONS_NEW,
        )

        # Legacy extensions
        assert ".figure" in EXTENSIONS_LEGACY
        assert ".plot" in EXTENSIONS_LEGACY
        assert ".stats" in EXTENSIONS_LEGACY

        # New extensions
        assert ".figure.zip" in EXTENSIONS_NEW
        assert ".plot.zip" in EXTENSIONS_NEW
        assert ".stats.zip" in EXTENSIONS_NEW

        # All extensions combined
        assert ".figure" in EXTENSIONS
        assert ".figure.zip" in EXTENSIONS

        # Extension mapping
        assert EXTENSION_MAP[".figure"] == ".figure.zip"
        assert EXTENSION_MAP[".plot"] == ".plot.zip"
        assert EXTENSION_MAP[".stats"] == ".stats.zip"

    def test_get_type_legacy_extensions(self):
        """get_type should work with legacy extensions."""
        from scitex.io.bundle import get_type

        # Legacy extensions return short type names
        assert get_type("test.figure") == "figz"
        assert get_type("test.plot") == "pltz"
        assert get_type("test.stats") == "statsz"

    def test_get_type_new_extensions(self):
        """get_type should work with new .zip extensions."""
        from scitex.io.bundle import get_type

        assert get_type("test.figure.zip") == "figure"
        assert get_type("test.plot.zip") == "plot"
        assert get_type("test.stats.zip") == "stats"


# EOF
