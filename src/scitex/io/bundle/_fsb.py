#!/usr/bin/env python3
# Timestamp: "2025-12-19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_fsb.py

"""
FSB (Figure-Statistics Bundle) Adapter.

Bridges the FSB package with scitex-code's bundle system, providing:
- Direct access to FSB's Bundle class and models
- Conversion utilities between FSB and scitex formats
- Schema validation via FSB's JSON schemas

FSB is the single source of truth for bundle specification.
This adapter ensures scitex-code uses FSB consistently.

Usage:
    from scitex.io.bundle import fsb

    # Create new bundle using FSB
    bundle = fsb.Bundle("my_plot", create=True, node_type="plot")
    bundle.encoding = {"traces": [...]}
    bundle.save()

    # Convert existing scitex spec to FSB format
    fsb_data = fsb.from_scitex_spec(spec, style)

    # Convert FSB back to scitex format
    spec, style = fsb.to_scitex_spec(bundle)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Re-export FSB public API
try:
    import fsb as _fsb
    from fsb import Bundle, load_schema, validate
    from fsb import DataInfo as FSBDataInfo
    from fsb import Encoding as FSBEncoding
    from fsb import Node as FSBNode
    from fsb import Stats as FSBStats
    from fsb import Theme as FSBTheme
    from fsb import __version__ as FSB_VERSION
    from fsb.models import Axes, BBox, ChannelEncoding, NodeRefs, SizeMM, TraceEncoding

    FSB_AVAILABLE = True
except ImportError:
    FSB_AVAILABLE = False
    FSB_VERSION = None
    Bundle = None

__all__ = [
    # Availability check
    "FSB_AVAILABLE",
    "FSB_VERSION",
    # FSB classes (re-exported)
    "Bundle",
    "FSBNode",
    "FSBEncoding",
    "FSBTheme",
    "FSBStats",
    "FSBDataInfo",
    # FSB models
    "BBox",
    "SizeMM",
    "Axes",
    "NodeRefs",
    "ChannelEncoding",
    "TraceEncoding",
    # FSB functions
    "validate",
    "load_schema",
    # Conversion utilities
    "from_scitex_spec",
    "to_scitex_spec",
    "create_bundle",
    "load_bundle",
]


def _require_fsb():
    """Raise ImportError if FSB is not available."""
    if not FSB_AVAILABLE:
        raise ImportError(
            "FSB package not installed. Install with: pip install fsb>=0.1.1"
        )


def create_bundle(
    path: Union[str, Path],
    node_type: str = "plot",
    name: Optional[str] = None,
    size_mm: Optional[Dict[str, float]] = None,
) -> "Bundle":
    """Create a new FSB bundle.

    Args:
        path: Path for the bundle (directory or .zip).
        node_type: Type of node ('figure' or 'plot').
        name: Human-readable name.
        size_mm: Physical size in mm {"width": ..., "height": ...}.

    Returns:
        FSB Bundle instance.

    Example:
        >>> bundle = create_bundle("my_plot", node_type="plot")
        >>> bundle.encoding = {"traces": [...]}
        >>> bundle.save()
    """
    _require_fsb()
    return Bundle(path, create=True, node_type=node_type, name=name, size_mm=size_mm)


def load_bundle(path: Union[str, Path]) -> "Bundle":
    """Load an existing FSB bundle.

    Args:
        path: Path to bundle (directory or .zip).

    Returns:
        FSB Bundle instance.
    """
    _require_fsb()
    return Bundle(path)


def from_scitex_spec(
    spec: Dict[str, Any],
    style: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convert scitex spec.json + style.json to FSB format.

    Transforms scitex's merged spec/style format into FSB's separated
    node.json, encoding.json, theme.json structure.

    Args:
        spec: scitex spec.json content.
        style: scitex style.json content (optional).

    Returns:
        Dict with 'node', 'encoding', 'theme' keys in FSB format.
    """
    _require_fsb()

    # Extract node info from spec
    node_data = {
        "id": spec.get("bundle_id", "unknown"),
        "type": spec.get("type", "plot"),
        "bbox_norm": spec.get("bbox_norm", {"x0": 0, "y0": 0, "x1": 1, "y1": 1}),
    }

    if "title" in spec:
        node_data["name"] = spec["title"]

    if "size_mm" in spec:
        node_data["size_mm"] = spec["size_mm"]

    # Extract encoding from spec elements/traces
    encoding_data = {"traces": []}

    elements = spec.get("elements", [])
    for elem in elements:
        if elem.get("type") == "trace":
            trace = {
                "trace_id": elem.get("id", f"trace_{len(encoding_data['traces'])}"),
            }
            if "data_ref" in elem:
                trace["data_ref"] = elem["data_ref"]
            if "x" in elem:
                trace["x"] = elem["x"]
            if "y" in elem:
                trace["y"] = elem["y"]
            if "color" in elem:
                trace["color"] = elem["color"]
            encoding_data["traces"].append(trace)

    # Extract theme from style or spec
    theme_data = {}
    if style:
        # style.json may have encoding mixed in - extract theme parts
        for key in ["colors", "typography", "lines", "markers", "grid", "preset"]:
            if key in style:
                theme_data[key] = style[key]

        # Per-trace theming
        if "traces" in style:
            theme_data["traces"] = style["traces"]

    return {
        "node": node_data,
        "encoding": encoding_data,
        "theme": theme_data,
    }


def to_scitex_spec(
    bundle: "Bundle",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Convert FSB bundle to scitex spec.json + style.json format.

    Merges FSB's separated structure back into scitex's format for
    backward compatibility.

    Args:
        bundle: FSB Bundle instance.

    Returns:
        Tuple of (spec, style) dictionaries.
    """
    _require_fsb()

    from ._types import SCHEMA_NAME, SCHEMA_VERSION

    # Build spec.json
    spec = {
        "schema": {"name": SCHEMA_NAME, "version": SCHEMA_VERSION},
        "bundle_id": bundle.node.id if bundle.node else "unknown",
        "type": bundle.bundle_type,
    }

    if bundle.node:
        spec["bbox_norm"] = bundle.node.bbox_norm.to_dict()
        if bundle.node.name:
            spec["title"] = bundle.node.name
        if bundle.node.size_mm:
            spec["size_mm"] = bundle.node.size_mm.to_dict()

    # Convert encoding traces to elements
    elements = []
    if bundle.encoding and "traces" in bundle.encoding:
        for trace in bundle.encoding["traces"]:
            elem = {
                "id": trace.get("trace_id", f"trace_{len(elements)}"),
                "type": "trace",
            }
            for key in ["data_ref", "x", "y", "color", "size", "group"]:
                if key in trace:
                    elem[key] = trace[key]
            elements.append(elem)

    if elements:
        spec["elements"] = elements

    # Build style.json from theme
    style = {}
    if bundle.theme:
        style.update(bundle.theme)

    return spec, style


def bundle_to_dict(bundle: "Bundle") -> Dict[str, Any]:
    """Convert FSB Bundle to a flat dictionary for scitex compatibility.

    Args:
        bundle: FSB Bundle instance.

    Returns:
        Dictionary with all bundle data.
    """
    _require_fsb()

    spec, style = to_scitex_spec(bundle)
    return {
        "type": "stx",
        "content_type": bundle.bundle_type,
        "spec": spec,
        "style": style,
        "path": bundle.path,
        "is_zip": bundle.path.suffix == ".zip",
    }


def dict_to_bundle(
    data: Dict[str, Any],
    path: Union[str, Path],
) -> "Bundle":
    """Create FSB Bundle from scitex dictionary data.

    Args:
        data: Dictionary with 'spec' and optionally 'style'.
        path: Path for the new bundle.

    Returns:
        FSB Bundle instance.
    """
    _require_fsb()

    spec = data.get("spec", {})
    style = data.get("style", {})

    node_type = spec.get("type", "plot")
    name = spec.get("title")
    size_mm = spec.get("size_mm")

    bundle = Bundle(path, create=True, node_type=node_type, name=name, size_mm=size_mm)

    # Set encoding and theme from converted data
    fsb_data = from_scitex_spec(spec, style)
    bundle.encoding = fsb_data["encoding"]
    bundle.theme = fsb_data["theme"]

    return bundle


# EOF
