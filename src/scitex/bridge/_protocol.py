#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/bridge/_protocol.py
# Time-stamp: "2024-12-09 09:30:00 (ywatanabe)"
"""
Bridge Protocol - Versioning and compatibility for cross-module communication.

This module defines the bridge protocol version and provides utilities
for ensuring compatibility between different versions of scitex modules.

Protocol Versioning
-------------------
The bridge protocol version follows semantic versioning:
- MAJOR: Breaking changes in bridge interfaces
- MINOR: New bridge functions added (backward compatible)
- PATCH: Bug fixes (backward compatible)

Usage:
    from scitex.bridge import BRIDGE_PROTOCOL_VERSION, check_protocol_compatibility
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# Protocol Version
# =============================================================================

BRIDGE_PROTOCOL_VERSION = "1.0.0"
"""
Current bridge protocol version.

Changes:
- 1.0.0: Initial protocol
    - Stats → Plt: add_stat_to_axes, extract_stats_from_axes
    - Stats → Vis: stat_result_to_annotation, add_stats_to_figure_model
    - Plt → Vis: figure_to_vis_model, axes_to_vis_axes
    - Coordinate conventions: axes coords (0-1) for plt, data coords for vis
"""


# =============================================================================
# Protocol Metadata
# =============================================================================


@dataclass
class ProtocolInfo:
    """
    Bridge protocol information for serialization and compatibility.

    Parameters
    ----------
    version : str
        Protocol version string (semver)
    source_module : str
        Module that created the data
    target_module : str
        Target module for the data
    coordinate_system : str
        Coordinate system used ("axes", "data", "figure", "mm", "px")
    """

    version: str = BRIDGE_PROTOCOL_VERSION
    source_module: str = ""
    target_module: str = ""
    coordinate_system: str = "data"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bridge_protocol_version": self.version,
            "source_module": self.source_module,
            "target_module": self.target_module,
            "coordinate_system": self.coordinate_system,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProtocolInfo":
        """Create from dictionary."""
        return cls(
            version=data.get("bridge_protocol_version", BRIDGE_PROTOCOL_VERSION),
            source_module=data.get("source_module", ""),
            target_module=data.get("target_module", ""),
            coordinate_system=data.get("coordinate_system", "data"),
        )


# =============================================================================
# Compatibility Utilities
# =============================================================================


def parse_version(version: str) -> Tuple[int, int, int]:
    """
    Parse a version string into (major, minor, patch) tuple.

    Parameters
    ----------
    version : str
        Version string like "1.2.3"

    Returns
    -------
    tuple
        (major, minor, patch) integers
    """
    parts = version.split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (major, minor, patch)


def check_protocol_compatibility(
    data_version: str,
    current_version: str = BRIDGE_PROTOCOL_VERSION,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a data version is compatible with the current protocol.

    Parameters
    ----------
    data_version : str
        Version of the data being loaded
    current_version : str
        Current protocol version (default: BRIDGE_PROTOCOL_VERSION)

    Returns
    -------
    tuple
        (is_compatible, warning_message)
        - is_compatible: True if data can be safely used
        - warning_message: None if compatible, else a warning string

    Examples
    --------
    >>> is_compat, msg = check_protocol_compatibility("1.0.0")
    >>> is_compat
    True

    >>> is_compat, msg = check_protocol_compatibility("2.0.0")
    >>> is_compat
    False
    >>> msg
    'Major version mismatch: data v2.0.0, current v1.0.0'
    """
    data_major, data_minor, _ = parse_version(data_version)
    curr_major, curr_minor, _ = parse_version(current_version)

    # Major version mismatch = incompatible
    if data_major != curr_major:
        return (
            False,
            f"Major version mismatch: data v{data_version}, current v{current_version}",
        )

    # Minor version newer than current = warning (may have unknown fields)
    if data_minor > curr_minor:
        return (
            True,
            f"Data version newer than current: data v{data_version}, "
            f"current v{current_version}. Some features may be ignored.",
        )

    return (True, None)


def add_protocol_metadata(
    data: Dict[str, Any],
    source_module: str,
    target_module: str,
    coordinate_system: str = "data",
) -> Dict[str, Any]:
    """
    Add bridge protocol metadata to a dictionary.

    Parameters
    ----------
    data : dict
        Data dictionary to annotate
    source_module : str
        Source module name (e.g., "stats", "plt")
    target_module : str
        Target module name (e.g., "vis", "plt")
    coordinate_system : str
        Coordinate system used (default: "data")

    Returns
    -------
    dict
        Data with protocol metadata added

    Examples
    --------
    >>> data = {"x": 10, "y": 20}
    >>> annotated = add_protocol_metadata(data, "stats", "vis")
    >>> annotated["_bridge_protocol"]["bridge_protocol_version"]
    '1.0.0'
    """
    protocol = ProtocolInfo(
        source_module=source_module,
        target_module=target_module,
        coordinate_system=coordinate_system,
    )
    data["_bridge_protocol"] = protocol.to_dict()
    return data


def extract_protocol_metadata(data: Dict[str, Any]) -> Optional[ProtocolInfo]:
    """
    Extract bridge protocol metadata from a dictionary.

    Parameters
    ----------
    data : dict
        Data dictionary that may contain protocol metadata

    Returns
    -------
    ProtocolInfo or None
        Protocol info if present, None otherwise
    """
    if "_bridge_protocol" in data:
        return ProtocolInfo.from_dict(data["_bridge_protocol"])
    return None


# =============================================================================
# Coordinate System Definitions
# =============================================================================

COORDINATE_SYSTEMS = {
    "axes": {
        "description": "Normalized axes coordinates (0-1)",
        "x_range": (0.0, 1.0),
        "y_range": (0.0, 1.0),
        "used_by": ["plt", "matplotlib"],
    },
    "data": {
        "description": "Data coordinates (actual x/y values)",
        "x_range": None,  # Depends on data
        "y_range": None,
        "used_by": ["vis", "FigureModel"],
    },
    "figure": {
        "description": "Figure coordinates (0-1 over entire figure)",
        "x_range": (0.0, 1.0),
        "y_range": (0.0, 1.0),
        "used_by": ["matplotlib", "suptitle"],
    },
    "mm": {
        "description": "Physical millimeters",
        "x_range": None,  # Depends on figure size
        "y_range": None,
        "used_by": ["vis", "publication"],
    },
    "px": {
        "description": "Pixels",
        "x_range": None,  # Depends on DPI and size
        "y_range": None,
        "used_by": ["canvas", "gui"],
    },
}


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "BRIDGE_PROTOCOL_VERSION",
    "ProtocolInfo",
    "parse_version",
    "check_protocol_compatibility",
    "add_protocol_metadata",
    "extract_protocol_metadata",
    "COORDINATE_SYSTEMS",
]


# EOF
