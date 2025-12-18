#!/usr/bin/env python3
# Timestamp: "2025-12-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_stx.py

"""
Unified .stx Bundle Handling.

Provides spec normalization, migration, and type detection for the unified
.stx bundle format. Supports backward compatibility with legacy formats.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

from ._types import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    STX_EXTENSION,
    TYPE_DEFAULTS,
    CircularReferenceError,
    ConstraintError,
    DepthLimitError,
    StxType,
)

__all__ = [
    "normalize_spec",
    "migrate_v1_to_v2",
    "get_stx_type",
    "create_stx_spec",
    "validate_stx_bundle",
    "generate_bundle_id",
    "is_stx_format",
    "get_default_constraints",
]


def generate_bundle_id() -> str:
    """Generate a unique bundle ID (UUID4)."""
    return str(uuid.uuid4())


def get_default_constraints(bundle_type: str) -> Dict[str, Any]:
    """Get default constraints for a bundle type.

    Args:
        bundle_type: Type string (figure, plot, stats, etc.)

    Returns:
        Dict with allow_children and max_depth.
    """
    return TYPE_DEFAULTS.get(bundle_type, {"allow_children": False, "max_depth": 1})


def is_stx_format(path: Union[str, Path]) -> bool:
    """Check if path is a .stx bundle."""
    p = Path(path)
    return p.suffix == STX_EXTENSION or (p.suffix == ".d" and p.stem.endswith(".stx"))


def get_stx_type(spec: Dict[str, Any]) -> Optional[str]:
    """Get bundle type from spec.json.

    For .stx format (v2.0.0), reads from spec["type"].
    For legacy formats, infers from schema name.

    Args:
        spec: Parsed spec.json dictionary.

    Returns:
        Type string (figure, plot, stats) or None if unknown.
    """
    # v2.0.0: type field in spec
    if "type" in spec:
        return spec["type"]

    # v1.0.0: infer from schema name
    schema = spec.get("schema", {})
    schema_name = schema.get("name", "")

    if "fig" in schema_name or "figure" in schema_name:
        return StxType.FIGURE
    elif "plt" in schema_name or "plot" in schema_name:
        return StxType.PLOT
    elif "stats" in schema_name:
        return StxType.STATS

    return None


def normalize_spec(
    spec: Dict[str, Any], bundle_type: Optional[str] = None
) -> Dict[str, Any]:
    """Normalize a spec to v2.0.0 format.

    Adds missing fields with defaults:
    - schema.name = "scitex.bundle"
    - schema.version = "2.0.0"
    - type (inferred if missing)
    - bundle_id (generated if missing)
    - constraints (from TYPE_DEFAULTS)

    Args:
        spec: Original spec dictionary.
        bundle_type: Override type (optional).

    Returns:
        Normalized spec dictionary.
    """
    normalized = spec.copy()

    # Ensure schema
    if "schema" not in normalized:
        normalized["schema"] = {}
    normalized["schema"]["name"] = SCHEMA_NAME
    normalized["schema"]["version"] = SCHEMA_VERSION

    # Ensure type
    if "type" not in normalized:
        normalized["type"] = bundle_type or get_stx_type(spec) or StxType.FIGURE

    # Ensure bundle_id
    if "bundle_id" not in normalized:
        normalized["bundle_id"] = generate_bundle_id()

    # Ensure constraints
    if "constraints" not in normalized:
        normalized["constraints"] = get_default_constraints(normalized["type"])

    return normalized


def create_stx_spec(
    bundle_type: str,
    title: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Create a new .stx spec.json.

    Args:
        bundle_type: Type (figure, plot, stats, etc.)
        title: Optional title for the bundle.
        **kwargs: Additional type-specific fields.

    Returns:
        Complete spec dictionary.
    """
    now = datetime.utcnow().isoformat() + "Z"

    spec = {
        "schema": {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
        },
        "type": bundle_type,
        "bundle_id": generate_bundle_id(),
        "constraints": get_default_constraints(bundle_type),
        "created": now,
        "modified": now,
    }

    if title:
        spec["title"] = title

    # Add type-specific fields
    if bundle_type == StxType.FIGURE:
        spec["elements"] = kwargs.get("elements", [])
        spec["size_mm"] = kwargs.get("size_mm", {"width": 170, "height": 120})
    elif bundle_type == StxType.PLOT:
        spec["plot_type"] = kwargs.get("plot_type", "line")
        spec["axes"] = kwargs.get("axes", {})
        spec["traces"] = kwargs.get("traces", [])
    elif bundle_type == StxType.STATS:
        spec["comparisons"] = kwargs.get("comparisons", [])
        spec["provenance"] = kwargs.get("provenance", {})

    spec.update({k: v for k, v in kwargs.items() if k not in spec})

    return spec


def migrate_v1_to_v2(spec: Dict[str, Any], legacy_type: str) -> Dict[str, Any]:
    """Migrate a v1.0.0 spec to v2.0.0 format.

    Args:
        spec: Original v1.0.0 spec.
        legacy_type: Legacy type string (figz, pltz, statsz).

    Returns:
        Migrated v2.0.0 spec.
    """
    # Get new type
    new_type = StxType.FROM_LEGACY.get(legacy_type, legacy_type)

    # Start with normalized base
    migrated = normalize_spec(spec, new_type)

    # Preserve original schema in metadata for reference
    original_schema = spec.get("schema", {})
    migrated["_migrated_from"] = {
        "schema": original_schema,
        "legacy_type": legacy_type,
    }

    # Type-specific migrations
    if new_type == StxType.FIGURE:
        # panels -> elements
        if "panels" in spec:
            migrated["elements"] = _migrate_panels_to_elements(spec["panels"])
        # figure -> title
        if "figure" in spec and isinstance(spec["figure"], dict):
            if "title" in spec["figure"]:
                migrated["title"] = spec["figure"]["title"]
            if "id" in spec["figure"]:
                migrated["figure_id"] = spec["figure"]["id"]

    elif new_type == StxType.STATS:
        # metadata -> provenance
        if "metadata" in spec and "provenance" not in migrated:
            migrated["provenance"] = spec["metadata"]

    return migrated


def _migrate_panels_to_elements(panels: list) -> list:
    """Migrate figz panels to stx elements."""
    elements = []
    for panel in panels:
        element = {
            "id": panel.get("id", panel.get("label", "unknown")),
            "type": "plot",  # Panels are typically plots
            "mode": "embed",
        }

        # Position
        if "position" in panel:
            element["position"] = panel["position"]
        elif "x_mm" in panel and "y_mm" in panel:
            element["position"] = {"x_mm": panel["x_mm"], "y_mm": panel["y_mm"]}

        # Size
        if "size" in panel:
            element["size"] = panel["size"]
        elif "width_mm" in panel and "height_mm" in panel:
            element["size"] = {
                "width_mm": panel["width_mm"],
                "height_mm": panel["height_mm"],
            }

        # Reference to child bundle
        if "plot" in panel:
            ref = panel["plot"]
            if not ref.endswith(".stx"):
                # Convert .pltz reference to children path
                ref = f"children/{panel['id']}.stx"
            element["ref"] = ref

        if "label" in panel:
            element["label"] = panel["label"]

        elements.append(element)

    return elements


def validate_stx_bundle(
    spec: Dict[str, Any],
    visited: Optional[Set[str]] = None,
    depth: int = 0,
) -> None:
    """Validate a .stx bundle spec with safety checks.

    CRITICAL: visited is for CYCLE detection only.
    depth is a SEPARATE argument for DEPTH limit (NOT len(visited)!).

    Args:
        spec: Bundle spec dictionary.
        visited: Set of bundle_ids seen in current path (for cycle detection).
        depth: Current nesting depth (increments per level).

    Raises:
        CircularReferenceError: If circular reference detected.
        DepthLimitError: If depth exceeds max_depth.
        ConstraintError: If type constraints violated.
    """
    if visited is None:
        visited = set()

    # Get bundle_id
    bundle_id = spec.get("bundle_id")
    if not bundle_id:
        raise ConstraintError("Missing required field: bundle_id")

    # 1. Circular reference check (using visited set)
    if bundle_id in visited:
        raise CircularReferenceError(f"Circular reference detected: {bundle_id}")
    visited.add(bundle_id)

    # 2. Get constraints
    bundle_type = spec.get("type", StxType.FIGURE)
    constraints = spec.get("constraints", get_default_constraints(bundle_type))
    max_depth = constraints.get("max_depth", 3)
    allow_children = constraints.get("allow_children", False)

    # 3. Depth check (using depth ARG, NOT len(visited))
    if depth > max_depth:
        raise DepthLimitError(
            f"Depth {depth} exceeds max_depth {max_depth} for type '{bundle_type}'"
        )

    # 4. Children constraint check
    elements = spec.get("elements", [])
    has_children = len(elements) > 0 and any(
        e.get("ref", "").endswith(".stx") for e in elements
    )

    if has_children and not allow_children:
        raise ConstraintError(f"Type '{bundle_type}' cannot have children")

    # Note: Recursive validation of children happens in load()
    # when children are actually loaded


# EOF
