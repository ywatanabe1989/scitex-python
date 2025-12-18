#!/usr/bin/env python3
# Timestamp: "2025-12-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_bundle.py

"""
Figz - Unified Element API for .stx bundles.

Everything is an Element. No special "panel" concept.
Self-recursive: bundles can contain child bundles at any level.

Supports both formats:
    - ZIP archive: figure.stx (for storage, transfer)
    - Directory: figure.stx.d/ (for editing, development)

Coordinate System:
- Origin (0,0) at TOP-LEFT
- All positions relative to parent's origin
- Child element positions are LOCAL to parent, not global

Usage:
    from scitex.fig import Figz

    # Create bundle (ZIP format)
    figz = Figz.create("figure.stx", "Figure1")

    # Create bundle (directory format)
    figz = Figz.create("figure.stx.d", "Figure1")

    # Add elements (all types use same API)
    figz.add_element("A", "plot", pltz_bytes, position={"x_mm": 10, "y_mm": 10})
    figz.add_element("title", "text", content="My Figure", position={"x_mm": 5, "y_mm": 2})
    figz.add_element("inset", "figure", child_figz_bytes, position={"x_mm": 100, "y_mm": 60})

    # Elements within child bundles use LOCAL coordinates
    # If "inset" has an annotation at (5,3), its absolute position is (105, 63)

    figz.save()
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scitex.io.bundle import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    TYPE_DEFAULTS,
    ZipBundle,
    generate_bundle_id,
    normalize_spec,
)

from .layout import normalize_position, normalize_size


def _is_directory_bundle(path: Path) -> bool:
    """Check if path is a directory bundle (.stx.d, .figz.d)."""
    return path.suffix == ".d" and path.is_dir()


def _is_stx_path(path: Path) -> bool:
    """Check if path is .stx or .stx.d format."""
    if path.suffix == ".stx":
        return True
    if path.suffix == ".d" and path.stem.endswith(".stx"):
        return True
    return False


__all__ = ["Figz"]


class Figz:
    """Unified Element API for .stx bundles.

    All content types (plot, text, shape, figure, etc.) are "elements".
    No special "panel" terminology - consistent API for everything.
    """

    SCHEMA = {"name": SCHEMA_NAME, "version": SCHEMA_VERSION}
    DEFAULT_SIZE_MM = {"width": 170, "height": 120}

    # Element types that can contain children (have their own coordinate space)
    CONTAINER_TYPES = {"figure", "plot"}
    # Element types that are leaf nodes (no children)
    LEAF_TYPES = {"text", "shape", "image", "stats", "symbol", "equation", "comment"}

    def __init__(
        self,
        path: Union[str, Path],
        name: Optional[str] = None,
        size_mm: Optional[Union[Dict[str, float], tuple]] = None,
        bundle_type: str = "figure",
    ):
        """Load existing bundle or create new one.

        Behavior:
            - Path exists + no params → Load
            - Path exists + matching params → Load (no conflict)
            - Path exists + different params → Error (conflict)
            - Path not exists + name → Create
            - Path not exists + no name → Error (not found)

        Args:
            path: Bundle path (.stx, .stx.d, .figz, .figz.d)
            name: Bundle name/title (required for creation, checked for conflict if loading)
            size_mm: Canvas size as dict {"width": mm, "height": mm} or tuple (width, height)
            bundle_type: Type of bundle (figure, plot, etc.)

        Raises:
            ValueError: If path exists but params don't match (conflict)
            FileNotFoundError: If path doesn't exist and name not provided

        Examples:
            # Load existing
            figz = Figz("existing.stx")

            # Create new (file doesn't exist)
            figz = Figz("new.stx", name="Figure1", size_mm=(200, 150))

            # Load existing with validation (error if mismatch)
            figz = Figz("existing.stx", name="Figure1", size_mm=(200, 150))
        """
        self.path = Path(path)
        self._spec: Optional[Dict[str, Any]] = None
        self._style: Optional[Dict[str, Any]] = None
        self._modified = False

        if self.path.exists():
            # File exists - load and check for conflicts
            self._is_dir = _is_directory_bundle(self.path)
            self._is_stx = _is_stx_path(self.path)
            self._load()

            # Check for conflicts if creation params provided
            if name is not None or size_mm is not None:
                self._check_conflicts(name, size_mm, bundle_type)
        else:
            # File doesn't exist - create mode
            if name is None:
                raise FileNotFoundError(
                    f"Bundle not found: {self.path}. "
                    "To create new, provide 'name' parameter."
                )
            self._create_new(name, size_mm, bundle_type)

    def _load(self) -> None:
        """Load bundle, normalizing to v2.0.0 format."""
        if self._is_dir:
            self._load_from_directory()
        else:
            self._load_from_zip()

    def _load_from_zip(self) -> None:
        """Load from ZIP archive."""
        with ZipBundle(self.path, mode="r") as zb:
            try:
                spec = zb.read_json("spec.json")
                self._spec = normalize_spec(spec, "figure")
                # Migrate legacy "panels" to "elements" if needed
                self._migrate_panels_to_elements()
            except FileNotFoundError:
                self._spec = self._create_default_spec("Untitled")
            try:
                self._style = zb.read_json("style.json")
            except FileNotFoundError:
                self._style = {}

    def _load_from_directory(self) -> None:
        """Load from directory bundle."""
        spec_path = self.path / "spec.json"
        style_path = self.path / "style.json"

        if spec_path.exists():
            with open(spec_path, encoding="utf-8") as f:
                spec = json.load(f)
            self._spec = normalize_spec(spec, "figure")
            self._migrate_panels_to_elements()
        else:
            self._spec = self._create_default_spec("Untitled")

        if style_path.exists():
            with open(style_path, encoding="utf-8") as f:
                self._style = json.load(f)
        else:
            self._style = {}

    def _migrate_panels_to_elements(self) -> None:
        """Migrate legacy 'panels' to unified 'elements' format."""
        if self._spec is None:
            return

        panels = self._spec.get("panels", [])
        if not panels:
            return

        elements = self._spec.get("elements", [])
        element_ids = {e["id"] for e in elements}

        for panel in panels:
            if panel["id"] not in element_ids:
                # Convert panel to element format
                element = {
                    "id": panel["id"],
                    "type": "plot",
                    "mode": "embed",
                    "ref": panel.get("plot", f"{panel['id']}.pltz"),
                    "position": normalize_position(panel.get("position")),
                    "size": normalize_size(panel.get("size")),
                }
                if "label" in panel:
                    element["label"] = panel["label"]
                elements.append(element)

        self._spec["elements"] = elements

    def _check_conflicts(
        self,
        name: Optional[str],
        size_mm: Optional[Union[Dict[str, float], tuple]],
        bundle_type: str,
    ) -> None:
        """Check if provided params conflict with loaded spec."""
        conflicts = []

        # Normalize size_mm from tuple if needed
        if isinstance(size_mm, (list, tuple)):
            size_mm = {"width": size_mm[0], "height": size_mm[1]}

        # Check name/title
        if name is not None:
            existing_name = self._spec.get("title")
            if existing_name and existing_name != name:
                conflicts.append(
                    f"title: existing='{existing_name}' vs provided='{name}'"
                )

        # Check size
        if size_mm is not None:
            existing_size = self._spec.get("size_mm", {})
            if existing_size:
                existing_w = existing_size.get("width")
                existing_h = existing_size.get("height")
                new_w = size_mm.get("width")
                new_h = size_mm.get("height")
                if (existing_w, existing_h) != (new_w, new_h):
                    conflicts.append(
                        f"size_mm: existing=({existing_w}, {existing_h}) vs "
                        f"provided=({new_w}, {new_h})"
                    )

        # Check bundle type
        existing_type = self._spec.get("type")
        if existing_type and existing_type != bundle_type:
            conflicts.append(
                f"type: existing='{existing_type}' vs provided='{bundle_type}'"
            )

        if conflicts:
            raise ValueError(
                f"Conflict loading {self.path}:\n  " + "\n  ".join(conflicts)
            )

    def _create_default_spec(
        self, name: str, size_mm: Optional[Dict] = None, bundle_type: str = "figure"
    ) -> Dict[str, Any]:
        """Create default v2.0.0 spec."""
        size = size_mm or self.DEFAULT_SIZE_MM
        constraints = TYPE_DEFAULTS.get(
            bundle_type, {"allow_children": True, "max_depth": 3}
        )
        return {
            "schema": self.SCHEMA,
            "type": bundle_type,
            "bundle_id": generate_bundle_id(),
            "constraints": constraints,
            "title": name,
            "size_mm": size,
            "elements": [],
        }

    def _create_new(
        self,
        name: str,
        size_mm: Optional[Union[Dict[str, float], tuple]] = None,
        bundle_type: str = "figure",
    ) -> None:
        """Create a new bundle at self.path."""
        # Normalize size_mm from tuple if needed
        if isinstance(size_mm, (list, tuple)):
            size_mm = {"width": size_mm[0], "height": size_mm[1]}

        # Detect if directory format requested
        is_dir_format = self.path.suffix == ".d" and self.path.stem.endswith(
            (".stx", ".figz")
        )

        # Warn about legacy extensions
        if self.path.suffix == ".figz" or (
            self.path.suffix == ".d" and self.path.stem.endswith(".figz")
        ):
            warnings.warn(
                ".figz extension is deprecated. Use .stx instead.",
                DeprecationWarning,
                stacklevel=3,
            )

        # Ensure proper extension
        valid_suffixes = (".stx", ".figz")
        valid_dir_suffixes = (".stx.d", ".figz.d")
        path_str = str(self.path)
        if not (
            self.path.suffix in valid_suffixes
            or any(path_str.endswith(s) for s in valid_dir_suffixes)
        ):
            self.path = self.path.with_suffix(".stx")

        self._is_dir = is_dir_format
        self._is_stx = _is_stx_path(self.path)
        self._spec = self._create_default_spec(name, size_mm, bundle_type)
        self._style = {}
        self._modified = True

        # Write to disk
        if is_dir_format:
            self.path.mkdir(parents=True, exist_ok=True)
            with open(self.path / "spec.json", "w", encoding="utf-8") as f:
                json.dump(self._spec, f, indent=2)
            with open(self.path / "style.json", "w", encoding="utf-8") as f:
                json.dump(self._style, f, indent=2)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with ZipBundle(self.path, mode="w") as zb:
                zb.write_json("spec.json", self._spec)
                zb.write_json("style.json", self._style)

    @classmethod
    def load_or_create(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        size_mm: Optional[Union[Dict[str, float], tuple]] = None,
        bundle_type: str = "figure",
    ) -> Figz:
        """Load existing bundle or create new one (no conflict).

        This is equivalent to calling Figz() directly.

        Behavior:
            - Path exists + no params → Load
            - Path exists + params → Error (conflict)
            - Path not exists + name → Create
            - Path not exists + no name → Error (not found)

        Args:
            path: Bundle path (.stx, .stx.d, .figz, .figz.d)
            name: Bundle name/title (required for creation)
            size_mm: Canvas size as dict or tuple (width_mm, height_mm)
            bundle_type: Type of bundle (figure, plot, etc.)

        Returns:
            Figz instance

        Raises:
            ValueError: If path exists but creation params provided (conflict)
            FileNotFoundError: If path doesn't exist and name not provided

        Examples:
            # Load existing
            figz = Figz.load_or_create("existing.stx")

            # Create new (path doesn't exist)
            figz = Figz.load_or_create("new.stx", name="Figure1", size_mm=(200, 150))
        """
        return cls(path, name=name, size_mm=size_mm, bundle_type=bundle_type)

    @classmethod
    def create(
        cls,
        path: Union[str, Path],
        name: str,
        size_mm: Optional[Dict[str, float]] = None,
        bundle_type: str = "figure",
    ) -> Figz:
        """Create a new bundle.

        Args:
            path: Output path (.stx or .stx.d extension)
            name: Bundle name/title
            size_mm: Canvas size {"width": mm, "height": mm}
            bundle_type: Type of bundle (figure, plot, etc.)

        Returns:
            New Figz instance

        Examples:
            # ZIP format (recommended for storage/transfer)
            figz = Figz.create("figure.stx", "My Figure")

            # Directory format (for editing/development)
            figz = Figz.create("figure.stx.d", "My Figure")
        """
        path = Path(path)

        # Detect if directory format requested
        is_dir_format = path.suffix == ".d" and path.stem.endswith((".stx", ".figz"))

        # Warn about legacy extensions
        if path.suffix == ".figz" or (
            path.suffix == ".d" and path.stem.endswith(".figz")
        ):
            warnings.warn(
                ".figz extension is deprecated. Use .stx instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Ensure proper extension
        valid_suffixes = (".stx", ".figz")
        valid_dir_suffixes = (".stx.d", ".figz.d")
        path_str = str(path)
        if not (
            path.suffix in valid_suffixes
            or any(path_str.endswith(s) for s in valid_dir_suffixes)
        ):
            path = path.with_suffix(".stx")

        size = size_mm or cls.DEFAULT_SIZE_MM
        constraints = TYPE_DEFAULTS.get(
            bundle_type, {"allow_children": True, "max_depth": 3}
        )

        spec = {
            "schema": cls.SCHEMA,
            "type": bundle_type,
            "bundle_id": generate_bundle_id(),
            "constraints": constraints,
            "title": name,
            "size_mm": size,
            "elements": [],
        }

        if is_dir_format:
            # Create directory bundle
            path.mkdir(parents=True, exist_ok=True)
            with open(path / "spec.json", "w", encoding="utf-8") as f:
                json.dump(spec, f, indent=2)
            with open(path / "style.json", "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2)
        else:
            # Create ZIP bundle
            path.parent.mkdir(parents=True, exist_ok=True)
            with ZipBundle(path, mode="w") as zb:
                zb.write_json("spec.json", spec)
                zb.write_json("style.json", {})

        return cls(path)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def spec(self) -> Dict[str, Any]:
        """Bundle specification."""
        return self._spec or {}

    @spec.setter
    def spec(self, value: Dict[str, Any]) -> None:
        self._spec = value
        self._modified = True

    @property
    def style(self) -> Dict[str, Any]:
        """Bundle style."""
        return self._style or {}

    @style.setter
    def style(self, value: Dict[str, Any]) -> None:
        self._style = value
        self._modified = True

    @property
    def bundle_id(self) -> Optional[str]:
        """Unique bundle identifier."""
        return self.spec.get("bundle_id")

    @property
    def bundle_type(self) -> str:
        """Bundle type (figure, plot, etc.)."""
        return self.spec.get("type", "figure")

    @property
    def elements(self) -> List[Dict[str, Any]]:
        """List of all elements."""
        return self.spec.get("elements", [])

    @property
    def size_mm(self) -> Dict[str, float]:
        """Canvas size in mm."""
        return self.spec.get("size_mm", self.DEFAULT_SIZE_MM)

    @property
    def constraints(self) -> Dict[str, Any]:
        """Bundle constraints (allow_children, max_depth)."""
        return self.spec.get("constraints", TYPE_DEFAULTS.get(self.bundle_type, {}))

    # =========================================================================
    # Unified Element API
    # =========================================================================

    def add_element(
        self,
        element_id: str,
        element_type: str,
        content: Optional[Union[bytes, str, Dict]] = None,
        position: Optional[Dict[str, float]] = None,
        size: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> None:
        """Add an element to this bundle.

        This is the unified API - use for ALL element types.
        Positions are relative to this bundle's origin (0,0 = top-left).

        Args:
            element_id: Unique identifier (e.g., "A", "title", "arrow_1")
            element_type: Type of element:
                - "plot": Matplotlib figure or .pltz bytes - content=Figure or bytes
                - "figure": Child figure (.stx) - content=bytes
                - "text": Text annotation - content=str or {"value": str, ...}
                - "shape": Shape (arrow, bracket, line) - content=shape_spec
                - "image": Raster image - content=bytes
                - "stats": Stats bundle - content=bytes
            content: Element content (type-dependent)
            position: Position {"x_mm": float, "y_mm": float} relative to parent
            size: Size {"width_mm": float, "height_mm": float}
            **kwargs: Additional element-specific properties

        Example:
            # Add a matplotlib figure directly
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 4, 9])
            figz.add_element("A", "plot", fig, {"x_mm": 10, "y_mm": 20},
                             {"width_mm": 60, "height_mm": 45})

            # Add text annotation
            figz.add_element("title", "text", "My Figure", {"x_mm": 85, "y_mm": 5})

            # Add arrow shape
            figz.add_element("arrow1", "shape", {
                "shape_type": "arrow",
                "start": {"x_mm": 20, "y_mm": 30},
                "end": {"x_mm": 40, "y_mm": 50}
            })
        """
        if self._spec is None:
            self._spec = self._create_default_spec("Untitled")

        pos = normalize_position(position)
        sz = normalize_size(size) if size else None

        # Build element spec
        element: Dict[str, Any] = {
            "id": element_id,
            "type": element_type,
            "position": pos,
        }

        if sz:
            element["size"] = sz

        # Add extra properties
        element.update(kwargs)

        # Handle content based on type
        ref_path = None
        content_bytes = None

        if element_type in ("plot", "figure", "stats", "image"):
            # Check if content is a matplotlib figure (for "plot" type)
            if element_type == "plot":
                import matplotlib.figure

                if isinstance(content, matplotlib.figure.Figure) or hasattr(
                    content, "figure"
                ):
                    # Convert matplotlib figure to .stx bundle bytes
                    content_bytes = self._figure_to_stx_bytes(content, element_id)
                elif isinstance(content, bytes):
                    content_bytes = content

            elif element_type == "image":
                # Handle image content: path string, Path object, or bytes
                img_ext = ".png"  # Default extension
                if isinstance(content, (str, Path)):
                    # Read from file path
                    img_path = Path(content)
                    if img_path.exists():
                        with open(img_path, "rb") as f:
                            content_bytes = f.read()
                        img_ext = img_path.suffix.lower() or ".png"
                    else:
                        raise FileNotFoundError(f"Image not found: {content}")
                elif isinstance(content, bytes):
                    content_bytes = content
                    # Detect format from bytes
                    if content_bytes[:4] == b"\x89PNG":
                        img_ext = ".png"
                    elif content_bytes[:4] == b"<svg" or b"<svg" in content_bytes[:100]:
                        img_ext = ".svg"
                    elif content_bytes[:2] == b"\xff\xd8":
                        img_ext = ".jpg"
                    elif content_bytes[:4] == b"%PDF":
                        img_ext = ".pdf"
                element["image_format"] = img_ext.lstrip(".")

            elif isinstance(content, bytes):
                content_bytes = content

            if content_bytes:
                # Store as child file
                if element_type == "image":
                    ext = element.get("image_format", "png")
                    ext = f".{ext}" if not ext.startswith(".") else ext
                else:
                    ext = {
                        "plot": ".stx",
                        "figure": ".stx",
                        "stats": ".stx",
                    }.get(element_type, ".bin")
                ref_path = f"children/{element_id}{ext}"
                element["mode"] = "embed"
                element["ref"] = ref_path

                # Validate if adding child bundle
                if element_type in ("plot", "figure") and content_bytes:
                    self._validate_can_add_child(content_bytes, element_id)

        elif element_type == "text":
            if isinstance(content, str):
                element["content"] = content
            elif isinstance(content, dict):
                element.update(content)

        elif element_type == "shape":
            if isinstance(content, dict):
                element.update(content)

        # Remove existing element with same id
        self._spec["elements"] = [
            e for e in self._spec.get("elements", []) if e["id"] != element_id
        ]
        self._spec["elements"].append(element)

        # Write to bundle
        if self._is_dir:
            with open(self.path / "spec.json", "w", encoding="utf-8") as f:
                json.dump(self._spec, f, indent=2)
            if ref_path and content_bytes:
                with open(self.path / ref_path, "wb") as f:
                    f.write(content_bytes)
        else:
            with ZipBundle(self.path, mode="a") as zb:
                zb.write_json("spec.json", self._spec)
                if ref_path and content_bytes:
                    zb.write_bytes(ref_path, content_bytes)

        self._modified = False

    def get_element(self, element_id: str) -> Optional[Dict[str, Any]]:
        """Get element specification by id."""
        for elem in self.elements:
            if elem["id"] == element_id:
                return elem
        return None

    def get_element_content(self, element_id: str) -> Optional[bytes]:
        """Get element content bytes (for embedded bundles/images)."""
        elem = self.get_element(element_id)
        if not elem or "ref" not in elem:
            return None
        try:
            if self._is_dir:
                with open(self.path / elem["ref"], "rb") as f:
                    return f.read()
            else:
                with ZipBundle(self.path, mode="r") as zb:
                    return zb.read_bytes(elem["ref"])
        except FileNotFoundError:
            return None

    def remove_element(self, element_id: str) -> None:
        """Remove an element."""
        if self._spec is None:
            return
        self._spec["elements"] = [
            e for e in self._spec.get("elements", []) if e["id"] != element_id
        ]
        self._modified = True

    def update_element_position(
        self, element_id: str, x_mm: float, y_mm: float
    ) -> None:
        """Update element position."""
        elem = self.get_element(element_id)
        if elem:
            elem["position"] = {"x_mm": x_mm, "y_mm": y_mm}
            self._modified = True

    def update_element_size(
        self, element_id: str, width_mm: float, height_mm: float
    ) -> None:
        """Update element size."""
        elem = self.get_element(element_id)
        if elem:
            elem["size"] = {"width_mm": width_mm, "height_mm": height_mm}
            self._modified = True

    def list_element_ids(self, element_type: Optional[str] = None) -> List[str]:
        """List element IDs, optionally filtered by type."""
        if element_type:
            return [e["id"] for e in self.elements if e.get("type") == element_type]
        return [e["id"] for e in self.elements]

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _figure_to_stx_bytes(self, fig, basename: str = "plot") -> bytes:
        """Convert a matplotlib figure to .stx bundle bytes.

        Args:
            fig: matplotlib Figure or object with .figure attribute
            basename: Name for the bundle and its exports (e.g., "plot_A")

        Returns:
            bytes: ZIP archive bytes of the .stx bundle
        """
        import tempfile
        from pathlib import Path

        import matplotlib.figure

        # Extract the actual Figure object
        if hasattr(fig, "figure"):
            fig = fig.figure

        if not isinstance(fig, matplotlib.figure.Figure):
            raise TypeError(f"Expected matplotlib Figure, got {type(fig).__name__}")

        # Save to a temporary .stx bundle and read bytes
        with tempfile.TemporaryDirectory() as tmpdir:
            stx_path = Path(tmpdir) / f"{basename}.stx"

            # Use scitex.io.save to create the stx bundle
            from scitex.io import save as io_save

            io_save(fig, stx_path, verbose=False, basename=basename)

            # Read the bundle bytes
            with open(stx_path, "rb") as f:
                return f.read()

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_can_add_child(self, child_bytes: bytes, child_id: str) -> None:
        """Validate child bundle can be added."""
        import tempfile
        import zipfile

        from scitex.io.bundle import (
            CircularReferenceError,
            ConstraintError,
            DepthLimitError,
            validate_stx_bundle,
        )

        # Check if this bundle allows children
        if not self.constraints.get("allow_children", True):
            raise ConstraintError(
                f"Bundle type '{self.bundle_type}' cannot have children"
            )

        # Get depth info
        parent_depth = self._spec.get("_depth", 0) if self._spec else 0
        max_depth = self.constraints.get("max_depth", 3)

        child_depth = parent_depth + 1
        if child_depth > max_depth:
            raise DepthLimitError(
                f"Adding child exceeds max_depth={max_depth} (current={parent_depth})"
            )

        # Check if content is a valid ZIP before validating circular references
        # This allows adding raw bytes that aren't bundles (e.g., raw data)
        if not child_bytes or len(child_bytes) < 4:
            return  # Too small to be a valid bundle

        # ZIP files start with PK signature (0x504B)
        if child_bytes[:2] != b"PK":
            return  # Not a ZIP file, skip bundle validation

        # Check circular reference for valid ZIP bundles
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".stx", delete=False) as f:
                f.write(child_bytes)
                temp_path = f.name

            with zipfile.ZipFile(temp_path, "r") as zf:
                if "spec.json" not in zf.namelist():
                    return  # Not a valid bundle (no spec.json)
                child_spec = json.loads(zf.read("spec.json"))

            parent_id = self.bundle_id
            child_bundle_id = child_spec.get("bundle_id")

            if parent_id and child_bundle_id == parent_id:
                raise CircularReferenceError(
                    f"Cannot add bundle to itself: {parent_id}"
                )

            visited = {parent_id} if parent_id else set()
            validate_stx_bundle(child_spec, visited=visited, depth=child_depth)

        except zipfile.BadZipFile:
            return  # Not a valid ZIP, skip validation
        finally:
            if temp_path:
                Path(temp_path).unlink(missing_ok=True)

    # =========================================================================
    # Save/Load
    # =========================================================================

    def save(
        self, path: Optional[Union[str, Path]] = None, verbose: bool = True
    ) -> Path:
        """Save bundle to disk.

        Args:
            path: Optional new path. If None, saves to current path.
            verbose: If True, log success message.

        Returns:
            Path to saved bundle.
        """
        from scitex.logging import getLogger
        from scitex.path._getsize import getsize
        from scitex.str._readable_bytes import readable_bytes

        logger = getLogger(__name__)

        original_path = self.path
        original_is_dir = self._is_dir

        if path is not None:
            # Save to new location
            self.path = Path(path)
            self._is_dir = self.path.suffix == ".d"

        if self._is_dir:
            self._save_to_directory(original_path, original_is_dir)
        else:
            self._save_to_zip(original_path, original_is_dir)
        self._modified = False

        if verbose:
            try:
                size = readable_bytes(getsize(self.path))
                logger.success(f"Saved to: {self.path} ({size})")
            except Exception:
                logger.success(f"Saved to: {self.path}")

        return self.path

    def _save_to_zip(
        self, original_path: Optional[Path] = None, original_is_dir: bool = False
    ) -> None:
        """Save to ZIP archive."""
        import zipfile

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with ZipBundle(self.path, mode="a") as zb:
            if self._spec is not None:
                zb.write_json("spec.json", self._spec)
            if self._style is not None:
                zb.write_json("style.json", self._style)

            # Generate and save exports with simple names (bundle dir provides context)
            if self.elements:
                try:
                    # Export all three formats with simple names
                    for fmt in ("png", "svg", "pdf"):
                        export_bytes = self.render_preview_format(fmt=fmt, dpi=150)
                        zb.write_bytes(f"exports/figure.{fmt}", export_bytes)
                except Exception:
                    pass  # Skip if rendering fails

            # Copy children from original bundle if saving to new location
            if original_path and original_path != self.path and original_path.exists():
                if original_is_dir:
                    # Copy from directory bundle
                    children_dir = original_path / "children"
                    if children_dir.exists():
                        for child_file in children_dir.iterdir():
                            with open(child_file, "rb") as f:
                                zb.write_bytes(f"children/{child_file.name}", f.read())
                else:
                    # Copy from ZIP bundle
                    with zipfile.ZipFile(original_path, "r") as src_zip:
                        for name in src_zip.namelist():
                            if name.startswith("children/"):
                                zb.write_bytes(name, src_zip.read(name))

    def _save_to_directory(
        self, original_path: Optional[Path] = None, original_is_dir: bool = False
    ) -> None:
        """Save to directory bundle."""
        import shutil
        import zipfile

        # Ensure directory exists
        self.path.mkdir(parents=True, exist_ok=True)

        if self._spec is not None:
            with open(self.path / "spec.json", "w", encoding="utf-8") as f:
                json.dump(self._spec, f, indent=2)
        if self._style is not None:
            with open(self.path / "style.json", "w", encoding="utf-8") as f:
                json.dump(self._style, f, indent=2)

        # Create directory structure
        exports_dir = self.path / "exports"
        cache_dir = self.path / "cache"
        exports_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate and save exports with simple names (bundle dir provides context)
        if self.elements:
            try:
                # Export all three formats with simple names
                for fmt in ("png", "svg", "pdf"):
                    export_bytes = self.render_preview_format(fmt=fmt, dpi=150)
                    with open(exports_dir / f"figure.{fmt}", "wb") as f:
                        f.write(export_bytes)

                # Save geometry for GUI editing
                geometry = self._extract_geometry()
                if geometry:
                    with open(cache_dir / "geometry_px.json", "w") as f:
                        json.dump(geometry, f, indent=2)
            except Exception:
                pass  # Skip if rendering fails

        # Copy children and exports from original bundle if saving to new location
        if original_path and original_path != self.path and original_path.exists():
            # Copy children directory
            children_dest = self.path / "children"
            # Copy exports directory
            exports_dest = self.path / "exports"

            if original_is_dir:
                # Copy from directory bundle
                children_src = original_path / "children"
                if children_src.exists():
                    if children_dest.exists():
                        shutil.rmtree(children_dest)
                    shutil.copytree(children_src, children_dest)

                exports_src = original_path / "exports"
                if exports_src.exists():
                    if exports_dest.exists():
                        shutil.rmtree(exports_dest)
                    shutil.copytree(exports_src, exports_dest)
            else:
                # Extract from ZIP bundle
                with zipfile.ZipFile(original_path, "r") as src_zip:
                    for name in src_zip.namelist():
                        if name.startswith("children/"):
                            # Extract to children directory
                            children_dest.mkdir(parents=True, exist_ok=True)
                            child_name = name[len("children/") :]
                            if child_name:  # Skip the directory entry itself
                                with open(children_dest / child_name, "wb") as f:
                                    f.write(src_zip.read(name))
                        elif name.startswith("exports/"):
                            # Extract to exports directory
                            exports_dest.mkdir(parents=True, exist_ok=True)
                            export_name = name[len("exports/") :]
                            if export_name:  # Skip the directory entry itself
                                with open(exports_dest / export_name, "wb") as f:
                                    f.write(src_zip.read(name))

    @property
    def is_directory(self) -> bool:
        """Return True if this is a directory bundle (.stx.d)."""
        return self._is_dir

    def pack(self, output_path: Optional[Union[str, Path]] = None) -> Figz:
        """Convert directory bundle to ZIP archive.

        Args:
            output_path: Output ZIP path. If None, uses same name without .d

        Returns:
            New Figz instance pointing to ZIP archive.

        Example:
            figz = Figz("figure.stx.d")
            packed = figz.pack()  # Creates figure.stx
        """
        if not self._is_dir:
            raise ValueError("Bundle is already a ZIP archive")

        if output_path is None:
            # Remove .d suffix: figure.stx.d -> figure.stx
            output_path = self.path.parent / self.path.stem
        output_path = Path(output_path)

        # Save any pending changes first
        self.save()

        # Create ZIP from directory
        from scitex.io.bundle import pack as bundle_pack

        bundle_pack(self.path, output_path)

        return Figz(output_path)

    def unpack(self, output_path: Optional[Union[str, Path]] = None) -> Figz:
        """Convert ZIP archive to directory bundle.

        Args:
            output_path: Output directory path. If None, adds .d suffix.

        Returns:
            New Figz instance pointing to directory bundle.

        Example:
            figz = Figz("figure.stx")
            unpacked = figz.unpack()  # Creates figure.stx.d/
        """
        if self._is_dir:
            raise ValueError("Bundle is already a directory")

        if output_path is None:
            # Add .d suffix: figure.stx -> figure.stx.d
            output_path = self.path.parent / f"{self.path.name}.d"
        output_path = Path(output_path)

        # Save any pending changes first
        self.save()

        # Create output directory and extract ZIP contents into it
        import zipfile

        output_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.path, "r") as zf:
            zf.extractall(output_path)

        return Figz(output_path)

    # =========================================================================
    # Auto-Crop
    # =========================================================================

    def auto_crop(self, margin_mm: float = 5.0) -> Dict[str, Any]:
        """Auto-crop the figure by adjusting coordinates and canvas size.

        This method:
        1. Calculates the bounding box of all elements
        2. Shifts all element positions so content starts at (margin, margin)
        3. Resizes the canvas to fit content + margin

        Unlike pixel-based cropping, this operates on coordinates directly,
        preserving vector quality and precision.

        Args:
            margin_mm: Margin to add around content (default: 5mm)

        Returns:
            Dict with crop information:
            - original_size: {"width", "height"} before cropping
            - new_size: {"width", "height"} after cropping
            - offset: {"x_mm", "y_mm"} shift applied to elements
            - bounds: Original content bounds before shift

        Example:
            >>> figz = Figz.create("test.stx", "Test")
            >>> figz.add_element("A", "text", "Hello", {"x_mm": 50, "y_mm": 40})
            >>> crop_info = figz.auto_crop(margin_mm=5)
            >>> figz.size_mm  # Canvas resized to fit content
            >>> figz.get_element("A")["position"]  # Position shifted
        """
        from .layout import auto_crop_layout, content_bounds

        if not self.elements:
            return {
                "original_size": self.size_mm.copy(),
                "new_size": self.size_mm.copy(),
                "offset": {"x_mm": 0, "y_mm": 0},
                "bounds": None,
            }

        # Get original info
        original_size = self.size_mm.copy()
        bounds = content_bounds(self.elements)

        # Calculate cropped layout
        shifted_elements, new_size = auto_crop_layout(self.elements, margin_mm)

        # Calculate offset applied
        if bounds:
            offset = {
                "x_mm": bounds["x_mm"] - margin_mm,
                "y_mm": bounds["y_mm"] - margin_mm,
            }
        else:
            offset = {"x_mm": 0, "y_mm": 0}

        # Apply changes
        self._spec["elements"] = shifted_elements
        self._spec["size_mm"] = {
            "width": new_size["width_mm"],
            "height": new_size["height_mm"],
        }
        self._modified = True

        return {
            "original_size": original_size,
            "new_size": {
                "width": new_size["width_mm"],
                "height": new_size["height_mm"],
            },
            "offset": offset,
            "bounds": bounds,
        }

    def get_content_bounds(self) -> Optional[Dict[str, float]]:
        """Get the bounding box of all elements.

        Returns:
            Bounding box {"x_mm", "y_mm", "width_mm", "height_mm"}
            or None if no elements.
        """
        from .layout import content_bounds

        return content_bounds(self.elements)

    # =========================================================================
    # Geometry Extraction (for GUI editing)
    # =========================================================================

    def _extract_geometry(self) -> dict:
        """Extract geometry data for all elements (hit areas for GUI editing).

        Returns a dictionary with bounding boxes and positions for each element,
        enabling interactive selection and editing in GUI tools.

        Returns
        -------
        dict
            Geometry data with element positions, sizes, and hit areas:
            {
                "canvas": {"width_mm": float, "height_mm": float},
                "elements": [
                    {
                        "id": str,
                        "type": str,
                        "position_mm": {"x": float, "y": float},
                        "size_mm": {"width": float, "height": float},
                        "bbox_mm": {"x0": float, "y0": float, "x1": float, "y1": float}
                    },
                    ...
                ]
            }
        """
        size = self.size_mm
        geometry = {
            "canvas": {
                "width_mm": size.get("width", 170),
                "height_mm": size.get("height", 120),
            },
            "elements": [],
        }

        for elem in self.elements:
            elem_geom = {
                "id": elem.get("id"),
                "type": elem.get("type"),
            }

            # Position
            pos = elem.get("position", {})
            x_mm = pos.get("x_mm", 0)
            y_mm = pos.get("y_mm", 0)
            elem_geom["position_mm"] = {"x": x_mm, "y": y_mm}

            # Size (if available)
            sz = elem.get("size", {})
            width_mm = sz.get("width_mm", 0)
            height_mm = sz.get("height_mm", 0)

            if width_mm > 0 and height_mm > 0:
                elem_geom["size_mm"] = {"width": width_mm, "height": height_mm}
                elem_geom["bbox_mm"] = {
                    "x0": x_mm,
                    "y0": y_mm,
                    "x1": x_mm + width_mm,
                    "y1": y_mm + height_mm,
                }

            # Shape-specific geometry
            if elem.get("type") == "shape":
                start = elem.get("start", {})
                end = elem.get("end", {})
                if start and end:
                    elem_geom["start_mm"] = start
                    elem_geom["end_mm"] = end
                    # Calculate bounding box for shapes
                    x0 = min(start.get("x_mm", 0), end.get("x_mm", 0))
                    y0 = min(start.get("y_mm", 0), end.get("y_mm", 0))
                    x1 = max(start.get("x_mm", 0), end.get("x_mm", 0))
                    y1 = max(start.get("y_mm", 0), end.get("y_mm", 0))
                    elem_geom["bbox_mm"] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

            geometry["elements"].append(elem_geom)

        return geometry

    # =========================================================================
    # Rendering
    # =========================================================================

    def render_preview(self, dpi: int = 150) -> bytes:
        """Render composed preview."""
        import tempfile

        import matplotlib.pyplot as plt
        from PIL import Image

        from scitex.plt import Pltz

        size = self.size_mm
        width_in = size.get("width", 170) / 25.4
        height_in = size.get("height", 120) / 25.4

        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax.set_xlim(0, size.get("width", 170))
        ax.set_ylim(size.get("height", 120), 0)  # Flip Y for top-left origin
        ax.axis("off")

        for elem in self.elements:
            elem_type = elem.get("type")
            pos = elem.get("position", {})
            sz = elem.get("size", {})

            if elem_type == "plot":
                content = self.get_element_content(elem["id"])
                if not content:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".pltz", delete=False) as f:
                    f.write(content)
                    temp_path = f.name
                try:
                    pltz = Pltz(temp_path)
                    preview = pltz.get_preview() or pltz.render_preview(dpi=dpi)
                    img = Image.open(io.BytesIO(preview))

                    # Get target size from element spec
                    target_width = sz.get("width_mm", 80)
                    target_height = sz.get("height_mm", 60)

                    # Get image original aspect ratio
                    img_width, img_height = img.size
                    img_aspect = img_width / img_height
                    target_aspect = target_width / target_height

                    # Calculate actual rendering size preserving aspect ratio
                    if img_aspect > target_aspect:
                        # Image is wider - fit to width
                        render_width = target_width
                        render_height = target_width / img_aspect
                    else:
                        # Image is taller - fit to height
                        render_height = target_height
                        render_width = target_height * img_aspect

                    # Center the image within the target area
                    x_offset = (target_width - render_width) / 2
                    y_offset = (target_height - render_height) / 2

                    x_start = pos.get("x_mm", 0) + x_offset
                    y_start = pos.get("y_mm", 0) + y_offset

                    ax.imshow(
                        img,
                        extent=[
                            x_start,
                            x_start + render_width,
                            y_start + render_height,
                            y_start,
                        ],
                        aspect="auto",
                    )
                finally:
                    Path(temp_path).unlink(missing_ok=True)

            elif elem_type == "text":
                content = elem.get("content", "")
                ax.text(
                    pos.get("x_mm", 0),
                    pos.get("y_mm", 0),
                    content,
                    fontsize=elem.get("fontsize", 10),
                    ha=elem.get("ha", "left"),
                    va=elem.get("va", "top"),
                )

            elif elem_type == "shape":
                shape_type = elem.get("shape_type", "")
                start = elem.get("start", {})
                end = elem.get("end", {})
                start_x = start.get("x_mm", 0)
                start_y = start.get("y_mm", 0)
                end_x = end.get("x_mm", 0)
                end_y = end.get("y_mm", 0)

                if shape_type == "arrow":
                    ax.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="black",
                            lw=1.5,
                        ),
                    )

                elif shape_type == "bracket":
                    # Draw a horizontal bracket with vertical ends
                    bracket_height = 3  # mm
                    mid_x = (start_x + end_x) / 2
                    # Draw left vertical
                    ax.plot(
                        [start_x, start_x],
                        [start_y, start_y - bracket_height],
                        "k-",
                        lw=1.5,
                    )
                    # Draw right vertical
                    ax.plot(
                        [end_x, end_x],
                        [end_y, end_y - bracket_height],
                        "k-",
                        lw=1.5,
                    )
                    # Draw horizontal
                    ax.plot(
                        [start_x, end_x],
                        [start_y - bracket_height, end_y - bracket_height],
                        "k-",
                        lw=1.5,
                    )

                elif shape_type == "line":
                    ax.plot(
                        [start_x, end_x],
                        [start_y, end_y],
                        "k-",
                        lw=1.5,
                    )

            elif elem_type == "image":
                content = self.get_element_content(elem["id"])
                if not content:
                    continue
                try:
                    img = Image.open(io.BytesIO(content))

                    # Get target size from element spec
                    target_width = sz.get("width_mm", 50)
                    target_height = sz.get("height_mm", 50)

                    # Get image original aspect ratio
                    img_width, img_height = img.size
                    img_aspect = img_width / img_height
                    target_aspect = target_width / target_height

                    # Calculate actual rendering size preserving aspect ratio
                    if img_aspect > target_aspect:
                        render_width = target_width
                        render_height = target_width / img_aspect
                    else:
                        render_height = target_height
                        render_width = target_height * img_aspect

                    # Center the image within the target area
                    x_offset = (target_width - render_width) / 2
                    y_offset = (target_height - render_height) / 2

                    x_start = pos.get("x_mm", 0) + x_offset
                    y_start = pos.get("y_mm", 0) + y_offset

                    ax.imshow(
                        img,
                        extent=[
                            x_start,
                            x_start + render_width,
                            y_start + render_height,
                            y_start,
                        ],
                        aspect="auto",
                    )
                except Exception:
                    pass  # Skip if image loading fails

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer.getvalue()

    def render_preview_format(self, fmt: str = "png", dpi: int = 150) -> bytes:
        """Render composed preview in specified format.

        Args:
            fmt: Output format ("png", "svg", "pdf")
            dpi: Resolution for raster formats

        Returns:
            bytes: Rendered image data
        """
        import tempfile

        import matplotlib.pyplot as plt
        from PIL import Image

        from scitex.plt import Pltz

        size = self.size_mm
        width_in = size.get("width", 170) / 25.4
        height_in = size.get("height", 120) / 25.4

        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax.set_xlim(0, size.get("width", 170))
        ax.set_ylim(size.get("height", 120), 0)  # Flip Y for top-left origin
        ax.axis("off")

        for elem in self.elements:
            elem_type = elem.get("type")
            pos = elem.get("position", {})
            sz = elem.get("size", {})

            if elem_type == "plot":
                content = self.get_element_content(elem["id"])
                if not content:
                    continue
                with tempfile.NamedTemporaryFile(suffix=".stx", delete=False) as f:
                    f.write(content)
                    temp_path = f.name
                try:
                    pltz = Pltz(temp_path)
                    preview = pltz.get_preview() or pltz.render_preview(dpi=dpi)
                    img = Image.open(io.BytesIO(preview))

                    # Get target size from element spec
                    target_width = sz.get("width_mm", 80)
                    target_height = sz.get("height_mm", 60)

                    # Get image original aspect ratio
                    img_width, img_height = img.size
                    img_aspect = img_width / img_height
                    target_aspect = target_width / target_height

                    # Calculate actual rendering size preserving aspect ratio
                    if img_aspect > target_aspect:
                        render_width = target_width
                        render_height = target_width / img_aspect
                    else:
                        render_height = target_height
                        render_width = target_height * img_aspect

                    # Center the image within the target area
                    x_offset = (target_width - render_width) / 2
                    y_offset = (target_height - render_height) / 2

                    x_start = pos.get("x_mm", 0) + x_offset
                    y_start = pos.get("y_mm", 0) + y_offset

                    ax.imshow(
                        img,
                        extent=[
                            x_start,
                            x_start + render_width,
                            y_start + render_height,
                            y_start,
                        ],
                        aspect="auto",
                    )
                finally:
                    Path(temp_path).unlink(missing_ok=True)

            elif elem_type == "text":
                content = elem.get("content", "")
                ax.text(
                    pos.get("x_mm", 0),
                    pos.get("y_mm", 0),
                    content,
                    fontsize=elem.get("fontsize", 10),
                    ha=elem.get("ha", "left"),
                    va=elem.get("va", "top"),
                )

            elif elem_type == "shape":
                shape_type = elem.get("shape_type", "")
                start = elem.get("start", {})
                end = elem.get("end", {})
                start_x = start.get("x_mm", 0)
                start_y = start.get("y_mm", 0)
                end_x = end.get("x_mm", 0)
                end_y = end.get("y_mm", 0)

                if shape_type == "arrow":
                    ax.annotate(
                        "",
                        xy=(end_x, end_y),
                        xytext=(start_x, start_y),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                    )
                elif shape_type == "bracket":
                    bracket_height = 3
                    ax.plot(
                        [start_x, start_x],
                        [start_y, start_y - bracket_height],
                        "k-",
                        lw=1.5,
                    )
                    ax.plot(
                        [end_x, end_x], [end_y, end_y - bracket_height], "k-", lw=1.5
                    )
                    ax.plot(
                        [start_x, end_x],
                        [start_y - bracket_height, end_y - bracket_height],
                        "k-",
                        lw=1.5,
                    )
                elif shape_type == "line":
                    ax.plot([start_x, end_x], [start_y, end_y], "k-", lw=1.5)

            elif elem_type == "image":
                content = self.get_element_content(elem["id"])
                if not content:
                    continue
                try:
                    img = Image.open(io.BytesIO(content))

                    # Get target size from element spec
                    target_width = sz.get("width_mm", 50)
                    target_height = sz.get("height_mm", 50)

                    # Get image original aspect ratio
                    img_width, img_height = img.size
                    img_aspect = img_width / img_height
                    target_aspect = target_width / target_height

                    # Calculate actual rendering size preserving aspect ratio
                    if img_aspect > target_aspect:
                        render_width = target_width
                        render_height = target_width / img_aspect
                    else:
                        render_height = target_height
                        render_width = target_height * img_aspect

                    # Center the image within the target area
                    x_offset = (target_width - render_width) / 2
                    y_offset = (target_height - render_height) / 2

                    x_start = pos.get("x_mm", 0) + x_offset
                    y_start = pos.get("y_mm", 0) + y_offset

                    ax.imshow(
                        img,
                        extent=[
                            x_start,
                            x_start + render_width,
                            y_start + render_height,
                            y_start,
                        ],
                        aspect="auto",
                    )
                except Exception:
                    pass  # Skip if image loading fails

        buffer = io.BytesIO()
        fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer.getvalue()

    # =========================================================================
    # Legacy Compatibility (deprecated)
    # =========================================================================

    @property
    def panels(self) -> List[Dict[str, Any]]:
        """Legacy: Get elements of type 'plot'."""
        warnings.warn(
            "'panels' is deprecated. Use 'elements' with type='plot' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [e for e in self.elements if e.get("type") == "plot"]

    def add_panel(
        self,
        panel_id: str,
        pltz_bytes: bytes,
        position: Optional[Dict[str, float]] = None,
        size: Optional[Dict[str, float]] = None,
    ) -> None:
        """Legacy: Add a plot element. Use add_element() instead."""
        warnings.warn(
            "'add_panel' is deprecated. Use add_element(id, 'plot', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.add_element(panel_id, "plot", pltz_bytes, position, size)

    def get_panel(self, panel_id: str) -> Optional[Dict[str, Any]]:
        """Legacy: Get panel. Use get_element() instead."""
        warnings.warn(
            "'get_panel' is deprecated. Use get_element() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_element(panel_id)

    def get_panel_pltz(self, panel_id: str) -> Optional[bytes]:
        """Legacy: Get panel content. Use get_element_content() instead."""
        warnings.warn(
            "'get_panel_pltz' is deprecated. Use get_element_content() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_element_content(panel_id)

    def list_panel_ids(self) -> List[str]:
        """Legacy: List panel IDs. Use list_element_ids('plot') instead."""
        warnings.warn(
            "'list_panel_ids' is deprecated. Use list_element_ids('plot') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_element_ids("plot")

    def __repr__(self) -> str:
        elements = self.list_element_ids()
        return (
            f"Figz({self.path.name!r}, type={self.bundle_type!r}, elements={elements})"
        )


# EOF
