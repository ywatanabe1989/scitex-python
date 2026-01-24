#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_state.py

"""
Editor state management for DearPyGui editor.

Provides EditorState dataclass to hold all editor state.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EditorState:
    """Holds all state for the DearPyGui editor."""

    # Core data
    json_path: Path
    metadata: Dict[str, Any]
    csv_data: Optional[Any] = None
    png_path: Optional[Path] = None
    manual_overrides: Dict[str, Any] = field(default_factory=dict)

    # Defaults
    scitex_defaults: Dict[str, Any] = field(default_factory=dict)
    metadata_defaults: Dict[str, Any] = field(default_factory=dict)
    current_overrides: Dict[str, Any] = field(default_factory=dict)

    # Modification tracking
    initial_overrides: Dict[str, Any] = field(default_factory=dict)
    user_modified: bool = False
    texture_id: Optional[int] = None

    # Selection state
    selected_element: Optional[Dict[str, Any]] = None
    selected_trace_index: Optional[int] = None

    # Preview bounds
    preview_bounds: Optional[Tuple[int, int, int, int]] = None
    axes_transform: Optional[Tuple] = None
    element_bboxes: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    element_bboxes_raw: Dict[str, Tuple] = field(default_factory=dict)

    # Hover state
    hovered_element: Optional[Dict[str, Any]] = None
    last_hover_check: float = 0
    backend_name: str = "dearpygui"

    # Cached rendering
    cached_base_image: Optional[Any] = None
    cached_base_data: Optional[List[float]] = None
    cache_dirty: bool = True

    @classmethod
    def create(
        cls,
        json_path: Path,
        metadata: Dict[str, Any],
        csv_data: Optional[Any] = None,
        png_path: Optional[Path] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
    ) -> "EditorState":
        """Create an EditorState with properly initialized defaults."""
        from .._defaults import extract_defaults_from_metadata, get_scitex_defaults

        state = cls(
            json_path=Path(json_path),
            metadata=metadata,
            csv_data=csv_data,
            png_path=Path(png_path) if png_path else None,
            manual_overrides=manual_overrides or {},
        )

        # Get SciTeX defaults and merge with metadata
        state.scitex_defaults = get_scitex_defaults()
        state.metadata_defaults = extract_defaults_from_metadata(metadata)

        # Start with defaults, then overlay manual overrides
        state.current_overrides = copy.deepcopy(state.scitex_defaults)
        state.current_overrides.update(state.metadata_defaults)
        state.current_overrides.update(state.manual_overrides)

        # Track modifications
        state.initial_overrides = copy.deepcopy(state.current_overrides)

        return state


# EOF
