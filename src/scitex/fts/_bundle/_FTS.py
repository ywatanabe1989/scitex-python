#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_FTS.py

"""
FTS Bundle Class - The main entry point for working with FTS bundles.

A FSB is a self-contained unit containing:
- node.json: Structural metadata (id, type, bbox, children)
- encoding.json: Data-to-visual mappings
- theme.json: Visual aesthetics
- data/: Data files with data_info.json
- stats/: Statistical results
- exports/: Rendered outputs (PNG, SVG, PDF)

Usage:
    from scitex.fts import FTS

    # Create new bundle
    bundle = FTS("my_plot", create=True, node_type="plot")
    bundle.encoding = {"traces": [...]}
    bundle.save()

    # Load existing bundle
    bundle = FTS("my_plot.zip")
    print(bundle.node.id)

    # Context manager (auto-save on exit)
    with FTS("my_plot.zip") as bundle:
        bundle.theme = {"colors": {...}}
    # Automatically saved
"""

import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ._dataclasses import DataInfo, Node, SizeMM
from ._loader import load_bundle_components
from ._saver import save_bundle_components
from .._fig import Encoding, Theme
from .._stats import Stats


class FTS:
    """Figure-Statistics Bundle - Self-contained figure/plot/stats package."""

    def __init__(
        self,
        path: Union[str, Path],
        create: bool = False,
        node_type: str = "plot",
        name: Optional[str] = None,
        size_mm: Optional[Dict[str, float]] = None,
    ):
        self._path = Path(path)
        self._is_zip = self._path.suffix == ".zip"
        self._node: Optional[Node] = None
        self._encoding: Optional[Encoding] = None
        self._theme: Optional[Theme] = None
        self._stats: Optional[Stats] = None
        self._data_info: Optional[DataInfo] = None
        self._dirty = False

        if create:
            self._create_new(node_type, name, size_mm)
        else:
            self._load()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def path(self) -> Path:
        """FSB path (directory or ZIP)."""
        return self._path

    @property
    def is_zip(self) -> bool:
        """Whether bundle is a ZIP file."""
        return self._is_zip

    @property
    def bundle_type(self) -> str:
        """FSB type (figure, plot, text, etc.)."""
        return self._node.type if self._node else "unknown"

    @property
    def is_dirty(self) -> bool:
        """Whether bundle has unsaved changes."""
        return self._dirty

    @property
    def node(self) -> Optional[Node]:
        """Node metadata."""
        return self._node

    @node.setter
    def node(self, value: Union[Node, Dict[str, Any]]):
        if isinstance(value, dict):
            self._node = Node.from_dict(value)
        else:
            self._node = value
        self._dirty = True

    @property
    def encoding(self) -> Optional[Encoding]:
        """Encoding specification (typed object)."""
        return self._encoding

    @encoding.setter
    def encoding(self, value: Union[Encoding, Dict[str, Any]]):
        if isinstance(value, dict):
            self._encoding = Encoding.from_dict(value)
        else:
            self._encoding = value
        self._dirty = True

    @property
    def encoding_dict(self) -> Optional[Dict[str, Any]]:
        """Encoding as dictionary (for serialization)."""
        return self._encoding.to_dict() if self._encoding else None

    @property
    def theme(self) -> Optional[Theme]:
        """Theme specification (typed object)."""
        return self._theme

    @theme.setter
    def theme(self, value: Union[Theme, Dict[str, Any]]):
        if isinstance(value, dict):
            self._theme = Theme.from_dict(value)
        else:
            self._theme = value
        self._dirty = True

    @property
    def theme_dict(self) -> Optional[Dict[str, Any]]:
        """Theme as dictionary (for serialization)."""
        return self._theme.to_dict() if self._theme else None

    @property
    def stats(self) -> Optional[Stats]:
        """Statistics."""
        return self._stats

    @stats.setter
    def stats(self, value: Union[Stats, Dict[str, Any]]):
        if isinstance(value, dict):
            self._stats = Stats.from_dict(value)
        else:
            self._stats = value
        self._dirty = True

    @property
    def data_info(self) -> Optional[DataInfo]:
        """Data info metadata."""
        return self._data_info

    @data_info.setter
    def data_info(self, value: Union[DataInfo, Dict[str, Any]]):
        if isinstance(value, dict):
            self._data_info = DataInfo.from_dict(value)
        else:
            self._data_info = value
        self._dirty = True

    # =========================================================================
    # Core Methods
    # =========================================================================

    def _create_new(
        self,
        node_type: str,
        name: Optional[str],
        size_mm: Optional[Dict[str, float]],
    ):
        """Create a new bundle."""
        bundle_id = str(uuid.uuid4())
        if name is None:
            name = self._path.stem

        self._node = Node(
            id=bundle_id,
            type=node_type,
            name=name,
            size_mm=SizeMM.from_dict(size_mm) if size_mm else None,
        )
        self._encoding = Encoding()
        self._theme = Theme()
        self._stats = Stats()
        self._dirty = True

    def _load(self):
        """Load existing bundle."""
        if not self._path.exists():
            raise FileNotFoundError(f"FSB not found: {self._path}")

        (
            self._node,
            self._encoding,
            self._theme,
            self._stats,
            self._data_info,
        ) = load_bundle_components(self._path)

    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        validate: bool = True,
        validation_level: str = "schema",
    ):
        """Save bundle to disk."""
        if path:
            self._path = Path(path)
            self._is_zip = self._path.suffix == ".zip"

        # Validate before saving
        if validate:
            result = self.validate(level=validation_level)
            result.raise_if_invalid()

        # Update modified timestamp
        if self._node:
            self._node.touch()

        save_bundle_components(
            self._path,
            node=self._node,
            encoding=self._encoding,
            theme=self._theme,
            stats=self._stats,
            data_info=self._data_info,
        )
        self._dirty = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert bundle to dictionary."""
        result = {
            "path": str(self._path),
            "is_zip": self._is_zip,
            "type": self.bundle_type,
        }
        if self._node:
            result["node"] = self._node.to_dict()
        if self._encoding:
            result["encoding"] = self._encoding.to_dict()
        if self._theme:
            result["theme"] = self._theme.to_dict()
        if self._stats:
            result["stats"] = self._stats.to_dict()
        if self._data_info:
            result["data_info"] = self._data_info.to_dict()
        return result

    def validate(self, level: str = "schema") -> "ValidationResult":
        """Validate bundle at specified level."""
        from ._validation import (
            ValidationResult,
            validate_data_info,
            validate_encoding,
            validate_node,
            validate_semantic,
            validate_stats,
            validate_strict,
            validate_theme,
        )

        result = ValidationResult(level=level)

        # Level 1: Schema validation
        if self._node:
            result.errors.extend(validate_node(self._node.to_dict()))
        if self._encoding:
            result.errors.extend(validate_encoding(self._encoding.to_dict()))
        if self._theme:
            result.errors.extend(validate_theme(self._theme.to_dict()))
        if self._stats:
            result.errors.extend(validate_stats(self._stats.to_dict()))
        if self._data_info:
            result.errors.extend(validate_data_info(self._data_info.to_dict()))

        # Level 2+: Semantic validation
        if level in ("semantic", "strict"):
            result.errors.extend(
                validate_semantic(
                    node=self._node.to_dict() if self._node else None,
                    encoding=self._encoding.to_dict() if self._encoding else None,
                    theme=self._theme.to_dict() if self._theme else None,
                    stats=self._stats.to_dict() if self._stats else None,
                    data_info=self._data_info.to_dict() if self._data_info else None,
                )
            )

        # Level 3: Strict validation
        if level == "strict":
            result.errors.extend(
                validate_strict(
                    node=self._node.to_dict() if self._node else None,
                    encoding=self._encoding.to_dict() if self._encoding else None,
                    theme=self._theme.to_dict() if self._theme else None,
                    stats=self._stats.to_dict() if self._stats else None,
                    data_info=self._data_info.to_dict() if self._data_info else None,
                )
            )

        return result

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_latex(
        self,
        output_path: Optional[Union[str, Path]] = None,
        validate: bool = True,
        validation_level: str = "syntax",
        include_preamble: bool = False,
        open_editor: bool = False,
    ) -> "LaTeXResult":
        """Export bundle to LaTeX.

        Args:
            output_path: Where to save .tex file (default: exports/)
            validate: Whether to validate generated LaTeX
            validation_level: Validation level ("syntax", "semantic", "compile")
            include_preamble: Include required packages in output
            open_editor: Open Flask editor for manual fixes

        Returns:
            LaTeXResult with code, validation, and metadata
        """
        from .._tables._latex import (
            LaTeXExportOptions,
            LaTeXResult,
            export_to_latex,
            launch_editor,
        )

        options = LaTeXExportOptions(
            validate=validate,
            validation_level=validation_level,
            include_preamble=include_preamble,
            output_path=Path(output_path) if output_path else None,
        )

        result = export_to_latex(self, options)

        if open_editor:
            launch_editor(result.latex_code, bundle=self, output_path=result.output_path)

        return result

    # =========================================================================
    # Context Manager
    # =========================================================================

    def __enter__(self) -> "FTS":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager, auto-saving if dirty and no exception."""
        if exc_type is None and self._dirty:
            self.save()
        return False

    def __repr__(self) -> str:
        dirty_marker = "*" if self._dirty else ""
        return f"FTS({self._path!r}, type={self.bundle_type!r}){dirty_marker}"


# =============================================================================
# Factory Functions
# =============================================================================


def load_bundle(path: Union[str, Path]) -> FTS:
    """Load an existing FTS bundle."""
    return FTS(path)


def create_bundle(
    path: Union[str, Path],
    node_type: str = "plot",
    name: Optional[str] = None,
    size_mm: Optional[Dict[str, float]] = None,
) -> FTS:
    """Create a new FTS bundle."""
    return FTS(path, create=True, node_type=node_type, name=name, size_mm=size_mm)


__all__ = ["FTS", "load_bundle", "create_bundle"]

# EOF
