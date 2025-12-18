#!/usr/bin/env python3
# Timestamp: "2025-12-18 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_bundle.py

"""
Pltz - Object-oriented API for plot bundles (.stx and legacy .pltz).

Supports unified .stx format (v2.0.0) with type="plot".

Usage:
    from scitex.plt import Pltz

    # Create (defaults to .stx)
    pltz = Pltz.create("plot.stx", plot_type="line", data=df)
    pltz.spec["axes"]["xlabel"] = "Time (s)"
    pltz.save()

    # Load (auto-detects format)
    pltz = Pltz("plot.stx")    # Native .stx
    pltz = Pltz("plot.pltz")   # Legacy (auto-normalized)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from scitex.io.bundle import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ZipBundle,
)

__all__ = ["Pltz"]


class Pltz:
    """High-level API for plot bundles (.stx and legacy .pltz).

    Supports both unified .stx format (v2.0.0) and legacy .pltz format.
    New bundles are created as .stx by default.
    """

    # v2.0.0 unified schema
    SCHEMA = {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}
    # Legacy schema (for backward compatibility)
    LEGACY_SCHEMA = {"name": "scitex.plt.plot", "version": "1.0.0"}
    DEFAULT_CONSTRAINTS = {"allow_children": False, "max_depth": 1}

    def __init__(self, path: Union[str, Path]):
        """Load an existing plot bundle (.stx or .pltz)."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Bundle not found: {self.path}")
        self._spec: Optional[Dict[str, Any]] = None
        self._style: Optional[Dict[str, Any]] = None
        self._data: Optional[pd.DataFrame] = None
        self._modified = False
        self._is_stx = self.path.suffix == ".stx"
        self._load()

    def _load(self) -> None:
        """Load bundle contents into memory, normalizing to v2.0.0 format."""
        from scitex.io.bundle import normalize_spec

        with ZipBundle(self.path, mode="r") as zb:
            try:
                spec = zb.read_json("spec.json")
                # Normalize to v2.0.0 format
                self._spec = normalize_spec(spec, "plot")
            except FileNotFoundError:
                self._spec = self._create_default_spec("line")
            try:
                self._style = zb.read_json("style.json")
            except FileNotFoundError:
                self._style = None
            try:
                self._data = zb.read_csv("data.csv")
            except FileNotFoundError:
                self._data = None

    def _create_default_spec(self, plot_type: str) -> Dict[str, Any]:
        """Create a default v2.0.0 spec."""
        from scitex.io.bundle import generate_bundle_id

        return {
            "schema": self.SCHEMA,
            "type": "plot",
            "bundle_id": generate_bundle_id(),
            "constraints": self.DEFAULT_CONSTRAINTS,
            "plot_type": plot_type,
            "axes": {"xlabel": "", "ylabel": ""},
        }

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        plot_type: str,
        data: Optional[pd.DataFrame] = None,
        spec_overrides: Optional[Dict[str, Any]] = None,
        use_stx: bool = True,
    ) -> Pltz:
        """Create a new plot bundle.

        Args:
            path: Output path (extension auto-adjusted)
            plot_type: Type of plot (e.g., "line", "scatter")
            data: Optional DataFrame with plot data
            spec_overrides: Additional spec fields to merge
            use_stx: If True, create .stx format; if False, create legacy .pltz
                     (deprecated, will be removed in v3.0.0)

        Returns:
            New Pltz instance
        """
        import warnings

        from scitex.io.bundle import generate_bundle_id

        path = Path(path)

        # Deprecation warnings for legacy format
        if not use_stx:
            warnings.warn(
                "use_stx=False is deprecated. Legacy .pltz format will be "
                "removed in v3.0.0. Use .stx format instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif path.suffix == ".pltz":
            warnings.warn(
                ".pltz extension is deprecated. Use .stx extension instead. "
                "Legacy format support will be removed in v3.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Determine extension
        if use_stx:
            if path.suffix not in (".stx", ".pltz"):
                path = path.with_suffix(".stx")
            elif path.suffix == ".pltz":
                # User explicitly wants .pltz
                pass
        else:
            if path.suffix != ".pltz":
                path = path.with_suffix(".pltz")

        path.parent.mkdir(parents=True, exist_ok=True)

        spec = {
            "schema": cls.SCHEMA,
            "type": "plot",
            "bundle_id": generate_bundle_id(),
            "constraints": cls.DEFAULT_CONSTRAINTS,
            "plot_type": plot_type,
            "axes": {"xlabel": "", "ylabel": ""},
        }
        if spec_overrides:
            spec.update(spec_overrides)

        with ZipBundle(path, mode="w") as zb:
            zb.write_json("spec.json", spec)
            if data is not None:
                zb.write_csv("data.csv", data)
        return cls(path)

    @classmethod
    def create_from_gallery(
        cls,
        path: Union[str, Path],
        category: str,
        plot_name: str,
        use_stx: bool = True,
    ) -> Pltz:
        """Create a plot bundle from gallery template.

        Args:
            path: Output path (extension auto-adjusted)
            category: Gallery category
            plot_name: Name of plot in gallery
            use_stx: If True, create .stx format; if False, create legacy .pltz

        Returns:
            New Pltz instance
        """
        from scitex.io.bundle import generate_bundle_id
        from scitex.plt.gallery import get_plot_data, get_plot_spec

        path = Path(path)

        # Determine extension
        if use_stx:
            if path.suffix not in (".stx", ".pltz"):
                path = path.with_suffix(".stx")
        else:
            if path.suffix != ".pltz":
                path = path.with_suffix(".pltz")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Get gallery template and enhance with v2.0.0 fields
        spec = get_plot_spec(category, plot_name)
        spec["schema"] = cls.SCHEMA
        spec["type"] = "plot"
        spec["bundle_id"] = generate_bundle_id()
        spec["constraints"] = cls.DEFAULT_CONSTRAINTS

        data = get_plot_data(category, plot_name)
        with ZipBundle(path, mode="w") as zb:
            zb.write_json("spec.json", spec)
            if data is not None:
                zb.write_csv("data.csv", data)
        return cls(path)

    @property
    def spec(self) -> Dict[str, Any]:
        """Plot specification dictionary."""
        return self._spec or {}

    @property
    def bundle_id(self) -> Optional[str]:
        """Unique bundle identifier (UUID)."""
        return self.spec.get("bundle_id")

    @spec.setter
    def spec(self, value: Dict[str, Any]) -> None:
        self._spec = value
        self._modified = True

    @property
    def style(self) -> Optional[Dict[str, Any]]:
        """Style dictionary (appearance)."""
        return self._style

    @style.setter
    def style(self, value: Dict[str, Any]) -> None:
        self._style = value
        self._modified = True

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Plot data as DataFrame."""
        return self._data

    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        self._data = value
        self._modified = True

    @property
    def plot_type(self) -> str:
        """Plot type from spec."""
        return self.spec.get("plot_type", "unknown")

    def save(self) -> None:
        """Save changes to bundle atomically."""
        with ZipBundle(self.path, mode="a") as zb:
            if self._spec is not None:
                zb.write_json("spec.json", self._spec)
            if self._style is not None:
                zb.write_json("style.json", self._style)
            if self._data is not None:
                zb.write_csv("data.csv", self._data)
        self._modified = False

    def render_preview(self, format: str = "png", dpi: int = 150) -> bytes:
        """Render plot preview and return image bytes."""
        import matplotlib.pyplot as plt_mpl

        import scitex as stx
        from scitex.plt.gallery._plots import PLOT_FUNCTIONS
        from scitex.plt.styles.presets import SCITEX_STYLE

        plot_type = self.spec.get("plot_type", "plot")
        category = self.spec.get("category", "line")

        # Create figure with SciTeX style
        style = SCITEX_STYLE.copy()
        style["figsize"] = (4, 3)
        fig, ax = stx.plt.subplots(**style)

        # Try to render using gallery plot function
        if plot_type in PLOT_FUNCTIONS:
            plot_func = PLOT_FUNCTIONS[plot_type]
            fig, ax = plot_func(fig, ax, stx)
        else:
            # Fallback: basic line plot if data available
            if self._data is not None and not self._data.empty:
                # Find numeric columns for simple plot
                numeric_cols = self._data.select_dtypes(include=["number"]).columns
                if len(numeric_cols) >= 2:
                    ax.plot(self._data[numeric_cols[0]], self._data[numeric_cols[1]])
                elif len(numeric_cols) == 1:
                    ax.plot(self._data[numeric_cols[0]])

        # Render to bytes
        buffer = io.BytesIO()
        fig_mpl = fig._fig_mpl if hasattr(fig, "_fig_mpl") else fig
        fig_mpl.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        plt_mpl.close(fig_mpl)
        return buffer.getvalue()

    def get_preview(self) -> Optional[bytes]:
        """Get cached preview from bundle if available."""
        try:
            with ZipBundle(self.path, mode="r") as zb:
                # Try multiple possible preview locations
                preview_files = [
                    "exports/preview.png",
                    f"exports/{self.path.stem}.png",  # e.g., exports/plot.png
                ]
                # Also check for any .png in exports/
                for name in zb.namelist():
                    # Handle nested .d structure: temp.pltz.d/exports/temp.png
                    if "exports/" in name and name.endswith(".png"):
                        # Exclude hitmap and overview files
                        if "_hitmap" not in name and "_overview" not in name:
                            preview_files.append(name)

                for preview_file in preview_files:
                    try:
                        return zb.read_bytes(preview_file)
                    except FileNotFoundError:
                        continue
                return None
        except FileNotFoundError:
            return None

    def update_preview(self, dpi: int = 150) -> None:
        """Update cached preview in bundle."""
        preview = self.render_preview(dpi=dpi)
        with ZipBundle(self.path, mode="a") as zb:
            zb.write_bytes("exports/preview.png", preview)

    def __repr__(self) -> str:
        return f"Pltz({self.path.name!r}, plot_type={self.plot_type!r})"


# EOF
