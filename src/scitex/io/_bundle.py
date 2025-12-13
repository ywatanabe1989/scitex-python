#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_bundle.py

"""
SciTeX Bundle I/O - Shared utilities for .figz, .pltz, and .statsz formats.

This module provides common bundle operations. Domain-specific logic is in:
    - scitex.plt.io._bundle  (pltz: hitmap, CSV, overview)
    - scitex.fig.io._bundle  (figz: panel composition, nested pltz)
    - scitex.stats.io._bundle (statsz: comparison metadata)

Bundle formats:
    .figz  - Publication Figure Bundle (panels + layout)
    .pltz  - Reproducible Plot Bundle (data + spec + exports)
    .statsz - Statistical Results Bundle (stats + metadata)

Each bundle can be:
    - Directory form: Figure1.figz.d/
    - ZIP archive form: Figure1.figz
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "is_bundle",
    "load_bundle",
    "save_bundle",
    "pack_bundle",
    "unpack_bundle",
    "validate_bundle",
    "validate_spec",
    "BundleType",
    "BundleValidationError",
    "BUNDLE_EXTENSIONS",
    "get_bundle_type",
    "dir_to_zip_path",
    "zip_to_dir_path",
]

# Bundle extensions
BUNDLE_EXTENSIONS = (".figz", ".pltz", ".statsz")


class BundleValidationError(ValueError):
    """Error raised when bundle validation fails."""
    pass


class BundleType:
    """Bundle type constants."""
    FIGZ = "figz"
    PLTZ = "pltz"
    STATSZ = "statsz"


def get_bundle_type(path: Union[str, Path]) -> Optional[str]:
    """Get bundle type from path.

    Args:
        path: Path to bundle (directory or ZIP).

    Returns:
        Bundle type string or None if not a bundle.
    """
    p = Path(path)

    # Directory bundle: ends with .figz.d, .pltz.d, .statsz.d
    if p.is_dir() and p.suffix == ".d":
        stem = p.stem  # e.g., "Figure1.figz"
        for ext in BUNDLE_EXTENSIONS:
            if stem.endswith(ext):
                return ext[1:]  # Remove leading dot
        return None

    # ZIP bundle: ends with .figz, .pltz, .statsz
    if p.suffix in BUNDLE_EXTENSIONS:
        return p.suffix[1:]  # Remove leading dot

    return None


def is_bundle(path: Union[str, Path]) -> bool:
    """Check if path is a SciTeX bundle (directory or ZIP).

    Args:
        path: Path to check.

    Returns:
        True if path is a bundle.
    """
    return get_bundle_type(path) is not None


def dir_to_zip_path(dir_path: Path) -> Path:
    """Convert directory path to ZIP path.

    Example: Figure1.figz.d/ -> Figure1.figz
    """
    if dir_path.suffix == ".d":
        return dir_path.with_suffix("")
    return dir_path


def zip_to_dir_path(zip_path: Path) -> Path:
    """Convert ZIP path to directory path.

    Example: Figure1.figz -> Figure1.figz.d/
    """
    return Path(str(zip_path) + ".d")


def pack_bundle(
    dir_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Pack a bundle directory into a ZIP archive.

    The ZIP archive includes the .d directory as a top-level folder so that
    standard unzip extracts to: fname.pltz.d/fname.csv, fname.pltz.d/fname.json, etc.

    Args:
        dir_path: Path to bundle directory (e.g., Figure1.figz.d/).
        output_path: Output ZIP path. Auto-generated if None.

    Returns:
        Path to created ZIP archive.
    """
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    if output_path is None:
        output_path = dir_to_zip_path(dir_path)
    else:
        output_path = Path(output_path)

    # Get the directory name to use as top-level folder in ZIP
    # e.g., "stx_line.pltz.d" for path "/path/to/stx_line.pltz.d"
    dir_name = dir_path.name

    # Create ZIP archive with directory structure preserved
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                # Include directory name in archive path
                # e.g., "stx_line.pltz.d/stx_line.csv"
                rel_path = file_path.relative_to(dir_path)
                arcname = Path(dir_name) / rel_path
                zf.write(file_path, arcname)

    return output_path


def unpack_bundle(
    zip_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Unpack a bundle ZIP archive into a directory.

    The ZIP archive contains a top-level .d directory, so extraction goes to
    the parent directory. E.g., stx_line.pltz extracts to create stx_line.pltz.d/

    Args:
        zip_path: Path to bundle ZIP (e.g., Figure1.figz).
        output_path: Output directory path. Auto-generated if None.

    Returns:
        Path to created directory.
    """
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise ValueError(f"Not a file: {zip_path}")

    # Determine extraction target
    if output_path is None:
        # Extract to same directory as ZIP file (ZIP contains .d folder structure)
        extract_to = zip_path.parent
        expected_dir = zip_to_dir_path(zip_path)
    else:
        output_path = Path(output_path)
        extract_to = output_path.parent
        expected_dir = output_path

    # Extract ZIP archive (contains .d directory structure)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    return expected_dir


def validate_spec(
    spec: Dict[str, Any], bundle_type: str, strict: bool = False
) -> List[str]:
    """Validate a bundle specification against its schema.

    Args:
        spec: The specification dictionary to validate.
        bundle_type: Bundle type ('figz', 'pltz', 'statsz').
        strict: If True, raise BundleValidationError on failure.

    Returns:
        List of validation warning/error messages (empty if valid).

    Raises:
        BundleValidationError: If strict=True and validation fails.
    """
    errors = []

    # Basic schema validation
    if "schema" not in spec:
        errors.append("Missing required field: schema")
    elif not isinstance(spec["schema"], dict):
        errors.append("'schema' field must be a dictionary")
    else:
        schema = spec["schema"]
        if "name" not in schema:
            errors.append("schema.name is required")
        if "version" not in schema:
            errors.append("schema.version is required")

    # Delegate to domain-specific validators
    if bundle_type == BundleType.FIGZ:
        from scitex.fig.io._bundle import validate_figz_spec
        errors.extend(validate_figz_spec(spec))
    elif bundle_type == BundleType.PLTZ:
        from scitex.plt.io._bundle import validate_pltz_spec
        errors.extend(validate_pltz_spec(spec))
    elif bundle_type == BundleType.STATSZ:
        from scitex.stats.io._bundle import validate_statsz_spec
        errors.extend(validate_statsz_spec(spec))
    else:
        errors.append(f"Unknown bundle type: {bundle_type}")

    if strict and errors:
        raise BundleValidationError("; ".join(errors))

    return errors


def validate_bundle(
    path: Union[str, Path], strict: bool = False
) -> Dict[str, Any]:
    """Validate a bundle and return validation results.

    Args:
        path: Path to bundle (directory or ZIP).
        strict: If True, raise BundleValidationError on failure.

    Returns:
        Dictionary with:
        - 'valid': bool
        - 'errors': list of error messages
        - 'bundle_type': detected bundle type
        - 'spec': parsed spec (if available)

    Raises:
        BundleValidationError: If strict=True and validation fails.
    """
    result = {
        "valid": True,
        "errors": [],
        "bundle_type": None,
        "spec": None,
    }

    p = Path(path)

    # Check bundle type
    bundle_type = get_bundle_type(p)
    if bundle_type is None:
        result["valid"] = False
        result["errors"].append(f"Not a valid bundle path: {path}")
        if strict:
            raise BundleValidationError(result["errors"][0])
        return result

    result["bundle_type"] = bundle_type

    # Try to load and validate
    try:
        bundle = load_bundle(path)
        spec = bundle.get("spec")
        result["spec"] = spec

        if spec is not None:
            errors = validate_spec(spec, bundle_type, strict=False)
            if errors:
                result["valid"] = False
                result["errors"].extend(errors)
        else:
            result["valid"] = False
            result["errors"].append("Bundle has no spec")

    except Exception as e:
        result["valid"] = False
        result["errors"].append(f"Failed to load bundle: {e}")

    if strict and not result["valid"]:
        raise BundleValidationError("; ".join(result["errors"]))

    return result


def load_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    """Load bundle from directory or ZIP transparently.

    Args:
        path: Path to bundle (directory or ZIP).

    Returns:
        Bundle data as dictionary with:
        - 'type': Bundle type ('figz', 'pltz', 'statsz')
        - 'spec': Parsed JSON specification
        - 'path': Original path
        - 'is_zip': Whether loaded from ZIP
        - Additional fields based on bundle type
    """
    p = Path(path)
    bundle_type = get_bundle_type(p)

    if bundle_type is None:
        raise ValueError(f"Not a valid bundle: {path}")

    result = {
        "type": bundle_type,
        "path": p,
        "is_zip": False,
    }

    # Handle ZIP vs directory
    if p.is_file() and p.suffix in BUNDLE_EXTENSIONS:
        # ZIP archive - extract to temp and load
        result["is_zip"] = True
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(temp_dir)
        bundle_dir = temp_dir
    else:
        bundle_dir = p

    # Delegate to domain-specific loaders
    if bundle_type == BundleType.FIGZ:
        from scitex.fig.io._bundle import load_figz_bundle
        result.update(load_figz_bundle(bundle_dir))
    elif bundle_type == BundleType.PLTZ:
        from scitex.plt.io import load_layered_pltz_bundle
        result.update(load_layered_pltz_bundle(bundle_dir))
    elif bundle_type == BundleType.STATSZ:
        from scitex.stats.io._bundle import load_statsz_bundle
        result.update(load_statsz_bundle(bundle_dir))

    return result


def save_bundle(
    data: Dict[str, Any],
    path: Union[str, Path],
    bundle_type: Optional[str] = None,
    as_zip: bool = False,
) -> Path:
    """Save data as a bundle.

    Args:
        data: Bundle data dictionary.
        path: Output path (with or without .d suffix).
        bundle_type: Bundle type ('figz', 'pltz', 'statsz'). Auto-detected if None.
        as_zip: If True, save as ZIP archive.

    Returns:
        Path to saved bundle.
    """
    p = Path(path)

    # Determine bundle type
    if bundle_type is None:
        bundle_type = get_bundle_type(p)
        if bundle_type is None:
            raise ValueError(f"Cannot determine bundle type from path: {path}")

    # Determine if saving as directory or ZIP
    if as_zip or (p.suffix in BUNDLE_EXTENSIONS and not str(p).endswith(".d")):
        save_as_zip = True
        if p.suffix == ".d":
            zip_path = dir_to_zip_path(p)
        else:
            zip_path = p
        dir_path = zip_to_dir_path(zip_path)
    else:
        save_as_zip = False
        if not str(p).endswith(".d"):
            dir_path = Path(str(p) + ".d")
        else:
            dir_path = p

    # Create directory
    dir_path.mkdir(parents=True, exist_ok=True)

    # Delegate to domain-specific savers
    if bundle_type == BundleType.FIGZ:
        from scitex.fig.io._bundle import save_figz_bundle
        save_figz_bundle(data, dir_path)
    elif bundle_type == BundleType.PLTZ:
        # Note: This path is only reached when calling save_bundle() directly
        # The main stx.io.save() flow uses _save_pltz_bundle() which handles layered format
        from scitex.plt.io._bundle import save_pltz_bundle
        save_pltz_bundle(data, dir_path)
    elif bundle_type == BundleType.STATSZ:
        from scitex.stats.io._bundle import save_statsz_bundle
        save_statsz_bundle(data, dir_path)
    else:
        raise ValueError(f"Unknown bundle type: {bundle_type}")

    # Pack to ZIP if requested
    if save_as_zip:
        pack_bundle(dir_path, zip_path)
        shutil.rmtree(dir_path)  # Remove temp directory
        return zip_path

    return dir_path


# Backward compatibility aliases
_get_bundle_type = get_bundle_type
_dir_to_zip_path = dir_to_zip_path
_zip_to_dir_path = zip_to_dir_path

# EOF
