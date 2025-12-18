#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_validate.py

"""Validation functions for Figz bundles."""

import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


def validate_can_add_child(
    child_bytes: bytes,
    child_id: str,
    parent_bundle_id: Optional[str],
    constraints: Dict[str, Any],
    bundle_type: str,
    spec: Optional[Dict[str, Any]],
) -> None:
    """Validate child bundle can be added.

    Args:
        child_bytes: Bytes of the child bundle
        child_id: ID being assigned to the child
        parent_bundle_id: Bundle ID of the parent
        constraints: Parent bundle constraints
        bundle_type: Type of parent bundle
        spec: Parent bundle spec

    Raises:
        ConstraintError: If bundle type doesn't allow children
        DepthLimitError: If adding child exceeds max depth
        CircularReferenceError: If child would create circular reference
    """
    from scitex.io.bundle import (
        CircularReferenceError,
        ConstraintError,
        DepthLimitError,
        validate_stx_bundle,
    )

    # Check if this bundle allows children
    if not constraints.get("allow_children", True):
        raise ConstraintError(f"Bundle type '{bundle_type}' cannot have children")

    # Get depth info
    parent_depth = spec.get("_depth", 0) if spec else 0
    max_depth = constraints.get("max_depth", 3)

    child_depth = parent_depth + 1
    if child_depth > max_depth:
        raise DepthLimitError(
            f"Adding child exceeds max_depth={max_depth} (current={parent_depth})"
        )

    # Check if content is a valid ZIP before validating circular references
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

        child_bundle_id = child_spec.get("bundle_id")

        if parent_bundle_id and child_bundle_id == parent_bundle_id:
            raise CircularReferenceError(
                f"Cannot add bundle to itself: {parent_bundle_id}"
            )

        visited = {parent_bundle_id} if parent_bundle_id else set()
        validate_stx_bundle(child_spec, visited=visited, depth=child_depth)

    except zipfile.BadZipFile:
        return  # Not a valid ZIP, skip validation
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


# EOF
