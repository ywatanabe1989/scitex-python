#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/io/bundle/_manifest.py

"""Bundle manifest handling.

Manifests are JSON files inside bundles that identify the bundle type
and version. This provides a more reliable way to identify bundle types
than relying solely on file extensions.

Example manifest.json:
    {
        "scitex": {
            "type": "plot",
            "version": "1.0.0"
        }
    }
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "MANIFEST_FILENAME",
    "create_manifest",
    "write_manifest",
    "read_manifest",
    "get_type_from_manifest",
]

MANIFEST_FILENAME = "manifest.json"


def create_manifest(bundle_type: str, version: str = "1.0.0") -> Dict[str, Any]:
    """Create a manifest dictionary.

    Args:
        bundle_type: Type of bundle ('plot', 'figure', 'stats')
        version: Bundle format version

    Returns:
        Manifest dictionary
    """
    return {"scitex": {"type": bundle_type, "version": version}}


def write_manifest(bundle_path: Path, bundle_type: str, version: str = "1.0.0") -> None:
    """Write manifest.json to bundle directory.

    Args:
        bundle_path: Path to bundle directory
        bundle_type: Type of bundle
        version: Bundle format version
    """
    manifest = create_manifest(bundle_type, version)
    manifest_path = Path(bundle_path) / MANIFEST_FILENAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(bundle_path: Path) -> Optional[Dict[str, Any]]:
    """Read manifest.json from bundle.

    Args:
        bundle_path: Path to bundle (directory or ZIP)

    Returns:
        Manifest dictionary or None if not found
    """
    bundle_path = Path(bundle_path)
    manifest_path = bundle_path / MANIFEST_FILENAME

    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def get_type_from_manifest(bundle_path: Path) -> Optional[str]:
    """Get bundle type from manifest.

    Args:
        bundle_path: Path to bundle

    Returns:
        Bundle type string or None if manifest not found/invalid
    """
    manifest = read_manifest(bundle_path)
    if manifest and "scitex" in manifest:
        return manifest["scitex"].get("type")
    return None


# EOF
