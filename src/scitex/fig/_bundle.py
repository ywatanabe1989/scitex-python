#!/usr/bin/env python3
# Timestamp: "2025-12-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_bundle.py

"""
Figz - Object-oriented API for figure bundles (.stx and legacy .figz).

Supports unified .stx format (v2.0.0) with self-recursive figures.

Usage:
    from scitex.fig import Figz

    # Create (defaults to .stx)
    figz = Figz.create("figure.stx", "Figure1")
    figz.add_panel("A", pltz_bytes, position={"x_mm": 10, "y_mm": 10})
    figz.save()

    # Load (auto-detects format)
    figz = Figz("figure.stx")   # Native .stx
    figz = Figz("figure.figz")  # Legacy (auto-migrated)

    # Self-recursive: add sub-figure
    figz.add_child_figure("inset", sub_figz_bytes, position={...})
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scitex.io.bundle import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    ZipBundle,
    generate_bundle_id,
    normalize_spec,
)

__all__ = ["Figz"]


class Figz:
    """High-level API for figure bundles (.stx and legacy .figz).

    Supports both unified .stx format (v2.0.0) and legacy .figz format.
    New bundles are created as .stx by default.
    """

    # v2.0.0 unified schema
    SCHEMA = {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}
    # Legacy schema (for backward compatibility)
    LEGACY_SCHEMA = {"name": "scitex.fig.figure", "version": "1.0.0"}
    DEFAULT_SIZE_MM = {"width": 170, "height": 120}
    DEFAULT_CONSTRAINTS = {"allow_children": True, "max_depth": 3}

    def __init__(self, path: Union[str, Path]):
        """Load an existing figure bundle (.stx or .figz)."""
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Bundle not found: {self.path}")
        self._spec: Optional[Dict[str, Any]] = None
        self._style: Optional[Dict[str, Any]] = None
        self._modified = False
        self._is_stx = self.path.suffix == ".stx"
        self._load()

    def _load(self) -> None:
        """Load bundle contents into memory, normalizing to v2.0.0 format."""
        with ZipBundle(self.path, mode="r") as zb:
            try:
                spec = zb.read_json("spec.json")
                # Normalize to v2.0.0 format
                self._spec = normalize_spec(spec, "figure")
            except FileNotFoundError:
                self._spec = self._create_default_spec("Untitled")
            try:
                self._style = zb.read_json("style.json")
            except FileNotFoundError:
                self._style = None

    def _create_default_spec(
        self, figure_name: str, size_mm: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a default v2.0.0 spec."""
        size = size_mm or self.DEFAULT_SIZE_MM
        return {
            "schema": self.SCHEMA,
            "type": "figure",
            "bundle_id": generate_bundle_id(),
            "constraints": self.DEFAULT_CONSTRAINTS,
            "title": figure_name,
            "size_mm": size,
            "elements": [],
            # Legacy compatibility
            "panels": [],
        }

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        figure_name: str,
        size_mm: Optional[Dict[str, float]] = None,
        use_stx: bool = True,
    ) -> Figz:
        """Create a new figure bundle.

        Args:
            path: Output path (extension auto-adjusted)
            figure_name: Name/ID of the figure
            size_mm: Canvas size {"width": mm, "height": mm}
            use_stx: If True, create .stx format; if False, create legacy .figz

        Returns:
            New Figz instance
        """
        path = Path(path)

        # Determine extension
        if use_stx:
            if path.suffix not in (".stx", ".figz"):
                path = path.with_suffix(".stx")
            elif path.suffix == ".figz":
                # User explicitly wants .figz
                pass
        else:
            if path.suffix != ".figz":
                path = path.with_suffix(".figz")

        path.parent.mkdir(parents=True, exist_ok=True)

        size = size_mm or cls.DEFAULT_SIZE_MM
        spec = {
            "schema": cls.SCHEMA,
            "type": "figure",
            "bundle_id": generate_bundle_id(),
            "constraints": cls.DEFAULT_CONSTRAINTS,
            "title": figure_name,
            "figure": {"id": figure_name, "title": figure_name},
            "size_mm": size,
            "elements": [],
            "panels": [],  # Legacy compatibility
        }
        with ZipBundle(path, mode="w") as zb:
            zb.write_json("spec.json", spec)
        return cls(path)

    @property
    def spec(self) -> Dict[str, Any]:
        """Figure specification dictionary."""
        return self._spec or {}

    @property
    def bundle_id(self) -> Optional[str]:
        """Unique bundle identifier (UUID)."""
        return self.spec.get("bundle_id")

    @property
    def elements(self) -> List[Dict[str, Any]]:
        """List of element specifications (v2.0.0 format)."""
        return self.spec.get("elements", [])

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
    def panels(self) -> List[Dict[str, Any]]:
        """List of panel specifications."""
        return self.spec.get("panels", [])

    @property
    def figure_name(self) -> str:
        """Figure name/id."""
        return self.spec.get("figure", {}).get("id", "Figure")

    @property
    def size_mm(self) -> Dict[str, float]:
        """Canvas size in mm."""
        return self.spec.get("size_mm", self.DEFAULT_SIZE_MM)

    def save(self) -> None:
        """Save changes to bundle atomically."""
        with ZipBundle(self.path, mode="a") as zb:
            if self._spec is not None:
                zb.write_json("spec.json", self._spec)
            if self._style is not None:
                zb.write_json("style.json", self._style)
        self._modified = False

    def add_panel(
        self,
        panel_id: str,
        pltz_bytes: bytes,
        position: Optional[Dict[str, float]] = None,
        size: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a panel to the figure.

        Args:
            panel_id: Panel identifier (e.g., "A", "B")
            pltz_bytes: Content of .pltz file as bytes
            position: Position in mm {"x_mm": float, "y_mm": float}
            size: Size in mm {"width_mm": float, "height_mm": float}
        """
        pos = position or {"x_mm": 10, "y_mm": 10}
        sz = size or {"width_mm": 80, "height_mm": 60}

        pltz_filename = f"{panel_id}.pltz"
        panel_spec = {
            "id": panel_id,
            "label": panel_id,
            "plot": pltz_filename,
            "position": pos,
            "size": sz,
        }

        # Update spec
        if self._spec is None:
            self._spec = {"schema": self.SCHEMA, "panels": []}

        # Remove existing panel with same id
        self._spec["panels"] = [
            p for p in self._spec.get("panels", []) if p["id"] != panel_id
        ]
        self._spec["panels"].append(panel_spec)

        # Write to bundle
        with ZipBundle(self.path, mode="a") as zb:
            zb.write_json("spec.json", self._spec)
            zb.write_bytes(pltz_filename, pltz_bytes)
        self._modified = False

    def remove_panel(self, panel_id: str) -> None:
        """Remove a panel from the figure."""
        if self._spec is None:
            return
        self._spec["panels"] = [
            p for p in self._spec.get("panels", []) if p["id"] != panel_id
        ]
        self._modified = True

    def get_panel(self, panel_id: str) -> Optional[Dict[str, Any]]:
        """Get panel specification by id."""
        for panel in self.panels:
            if panel["id"] == panel_id:
                return panel
        return None

    def get_panel_pltz(self, panel_id: str) -> Optional[bytes]:
        """Get panel .pltz content as bytes."""
        panel = self.get_panel(panel_id)
        if not panel:
            return None
        try:
            with ZipBundle(self.path, mode="r") as zb:
                return zb.read_bytes(panel["plot"])
        except FileNotFoundError:
            return None

    def get_panel_data(self, panel_id: str) -> Optional[pd.DataFrame]:
        """Get panel data as DataFrame.

        Args:
            panel_id: Panel identifier (e.g., 'A', 'B')

        Returns:
            DataFrame with panel data, or None if not found
        """
        import tempfile

        from scitex.plt import Pltz

        pltz_bytes = self.get_panel_pltz(panel_id)
        if not pltz_bytes:
            return None

        # Extract to temp file and read data
        with tempfile.NamedTemporaryFile(suffix=".pltz", delete=False) as f:
            f.write(pltz_bytes)
            temp_path = f.name

        try:
            pltz = Pltz(temp_path)
            return pltz.data
        finally:
            import os

            os.unlink(temp_path)

    def update_panel_position(self, panel_id: str, x_mm: float, y_mm: float) -> None:
        """Update panel position."""
        for panel in self.panels:
            if panel["id"] == panel_id:
                panel["position"] = {"x_mm": x_mm, "y_mm": y_mm}
                self._modified = True
                return

    def update_panel_size(
        self, panel_id: str, width_mm: float, height_mm: float
    ) -> None:
        """Update panel size."""
        for panel in self.panels:
            if panel["id"] == panel_id:
                panel["size"] = {"width_mm": width_mm, "height_mm": height_mm}
                self._modified = True
                return

    def render_preview(self, format: str = "png", dpi: int = 150) -> bytes:
        """Render composed figure preview."""
        import tempfile

        import matplotlib.pyplot as plt
        from PIL import Image

        from scitex.plt import Pltz

        size = self.size_mm
        fig, ax = plt.subplots(
            figsize=(size.get("width", 170) / 25.4, size.get("height", 120) / 25.4)
        )
        ax.set_xlim(0, size.get("width", 170))
        ax.set_ylim(size.get("height", 120), 0)
        ax.axis("off")

        for panel in self.panels:
            pltz_bytes = self.get_panel_pltz(panel["id"])
            if not pltz_bytes:
                continue
            with tempfile.NamedTemporaryFile(suffix=".pltz", delete=False) as f:
                f.write(pltz_bytes)
                temp_path = f.name
            try:
                pltz = Pltz(temp_path)
                preview = pltz.get_preview() or pltz.render_preview(dpi=dpi)
                pos, sz = panel.get("position", {}), panel.get("size", {})
                img = Image.open(io.BytesIO(preview))
                ax.imshow(
                    img,
                    extent=[
                        pos.get("x_mm", 0),
                        pos.get("x_mm", 0) + sz.get("width_mm", 80),
                        pos.get("y_mm", 0) + sz.get("height_mm", 60),
                        pos.get("y_mm", 0),
                    ],
                )
            finally:
                Path(temp_path).unlink(missing_ok=True)

        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer.getvalue()

    def list_panel_ids(self) -> List[str]:
        """List all panel IDs."""
        return [p["id"] for p in self.panels]

    def __repr__(self) -> str:
        return f"Figz({self.path.name!r}, panels={self.list_panel_ids()})"


# EOF
