#!/usr/bin/env python3
# Timestamp: 2026-01-07
# File: src/scitex/io/bundle/_core.py

"""
SciTeX Bundle Core Operations.

Provides load, save, copy, pack, unpack, and validate operations for
.figz, .pltz, and .statsz bundle formats (and new .figure.zip, .plot.zip, .stats.zip).

Each bundle can exist in two forms:
    - Directory: Figure1.figz.d/ or Figure1.figure/
    - ZIP archive: Figure1.figz or Figure1.figure.zip
"""

import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._types import (
    DIR_EXTENSIONS_NEW,
    EXTENSIONS,
    EXTENSIONS_LEGACY,
    EXTENSIONS_NEW,
    BundleNotFoundError,
    BundleType,
    BundleValidationError,
)

__all__ = [
    "load",
    "save",
    "copy",
    "pack",
    "unpack",
    "validate",
    "validate_spec",
    "is_bundle",
    "get_type",
    "dir_to_zip_path",
    "zip_to_dir_path",
]


def get_type(path: Union[str, Path]) -> Optional[str]:
    """Get bundle type from path.

    Args:
        path: Path to bundle (directory or ZIP).

    Returns:
        Bundle type string ('figz', 'pltz', 'statsz', 'figure', 'plot', 'stats')
        or None if not a bundle.

    Example:
        >>> get_type("Figure1.figz")
        'figz'
        >>> get_type("plot.pltz.d")
        'pltz'
        >>> get_type("Figure1.figure.zip")
        'figure'
        >>> get_type("plot.plot")
        'plot'
    """
    p = Path(path)
    name = p.name.lower()

    # Check new double extensions first (.figure.zip, .plot.zip, .stats.zip)
    for ext in EXTENSIONS_NEW:
        if name.endswith(ext):
            # Return type without dots: 'figure', 'plot', 'stats'
            return ext.split(".")[1]

    # Check new directory extensions (.figure, .plot, .stats)
    for ext in DIR_EXTENSIONS_NEW:
        if name.endswith(ext) and p.is_dir():
            return ext[1:]  # Remove leading dot

    # Directory bundle: ends with .figz.d, .pltz.d, .statsz.d
    if p.is_dir() and p.suffix == ".d":
        stem = p.stem  # e.g., "Figure1.figz"
        for ext in EXTENSIONS_LEGACY:
            if stem.endswith(ext):
                return ext[1:]  # Remove leading dot
        return None

    # ZIP bundle: ends with .figz, .pltz, .statsz
    if p.suffix in EXTENSIONS_LEGACY:
        return p.suffix[1:]  # Remove leading dot

    return None


def is_bundle(path: Union[str, Path]) -> bool:
    """Check if path is a SciTeX bundle (directory or ZIP).

    Args:
        path: Path to check.

    Returns:
        True if path is a bundle.

    Example:
        >>> is_bundle("Figure1.figz")
        True
        >>> is_bundle("data.csv")
        False
    """
    return get_type(path) is not None


def dir_to_zip_path(dir_path: Path) -> Path:
    """Convert directory path to ZIP path.

    Example:
        >>> dir_to_zip_path(Path("Figure1.figz.d"))
        Path('Figure1.figz')
    """
    if dir_path.suffix == ".d":
        return dir_path.with_suffix("")
    return dir_path


def zip_to_dir_path(zip_path: Path) -> Path:
    """Convert ZIP path to directory path.

    Example:
        >>> zip_to_dir_path(Path("Figure1.figz"))
        Path('Figure1.figz.d')
    """
    return Path(str(zip_path) + ".d")


def pack(
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

    Example:
        >>> pack("plot.pltz.d")
        Path('plot.pltz')
    """
    dir_path = Path(dir_path)

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    if output_path is None:
        output_path = dir_to_zip_path(dir_path)
    else:
        output_path = Path(output_path)

    # Get the directory name to use as top-level folder in ZIP
    dir_name = dir_path.name

    # Create ZIP archive with directory structure preserved
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in dir_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(dir_path)
                arcname = Path(dir_name) / rel_path
                zf.write(file_path, arcname)

    return output_path


def unpack(
    zip_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
) -> Path:
    """Unpack a bundle ZIP archive into a directory.

    Args:
        zip_path: Path to bundle ZIP (e.g., Figure1.figz).
        output_path: Output directory path. Auto-generated if None.

    Returns:
        Path to created directory.

    Example:
        >>> unpack("plot.pltz")
        Path('plot.pltz.d')
    """
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise ValueError(f"Not a file: {zip_path}")

    if output_path is None:
        extract_to = zip_path.parent
        expected_dir = zip_to_dir_path(zip_path)
    else:
        output_path = Path(output_path)
        extract_to = output_path.parent
        expected_dir = output_path

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
        from scitex.canvas.io._bundle import validate_figz_spec

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


def validate(path: Union[str, Path], strict: bool = False) -> Dict[str, Any]:
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

    bundle_type = get_type(p)
    if bundle_type is None:
        result["valid"] = False
        result["errors"].append(f"Not a valid bundle path: {path}")
        if strict:
            raise BundleValidationError(result["errors"][0])
        return result

    result["bundle_type"] = bundle_type

    try:
        bundle = load(path)
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


def load(path: Union[str, Path], in_memory: bool = True) -> Dict[str, Any]:
    """Load bundle from directory or ZIP transparently.

    Args:
        path: Path to bundle (directory or ZIP).
        in_memory: If True, load ZIP contents in-memory without extracting.
                   If False, extract to temp directory (legacy behavior).

    Returns:
        Bundle data as dictionary with:
        - 'type': Bundle type ('figz', 'pltz', 'statsz')
        - 'spec': Parsed JSON specification
        - 'path': Original path
        - 'is_zip': Whether loaded from ZIP
        - Additional fields based on bundle type

    Example:
        >>> bundle = load("Figure1.figz")
        >>> bundle['type']
        'figz'
        >>> bundle['spec']['schema']['name']
        'scitex.canvas'
    """
    p = Path(path)
    bundle_type = get_type(p)

    if bundle_type is None:
        raise BundleNotFoundError(f"Not a valid bundle: {path}")

    result = {
        "type": bundle_type,
        "path": p,
        "is_zip": False,
    }

    # Handle ZIP vs directory
    if p.is_file() and p.suffix in EXTENSIONS:
        result["is_zip"] = True

        if in_memory:
            from ._zip import ZipBundle

            with ZipBundle(p, mode="r") as zb:
                result["_zip_bundle"] = zb
                try:
                    result["spec"] = zb.read_json("spec.json")
                except FileNotFoundError:
                    result["spec"] = None
                try:
                    result["style"] = zb.read_json("style.json")
                except FileNotFoundError:
                    result["style"] = None
                try:
                    result["data"] = zb.read_csv("data.csv")
                except FileNotFoundError:
                    result["data"] = None

                result["files"] = zb.namelist()

            return result
        else:
            import tempfile

            temp_dir = Path(tempfile.mkdtemp())
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(temp_dir)
            bundle_dir = temp_dir
    else:
        bundle_dir = p

    # Delegate to domain-specific loaders
    if bundle_type == BundleType.FIGZ:
        from scitex.canvas.io._bundle import load_figz_bundle

        result.update(load_figz_bundle(bundle_dir))
    elif bundle_type == BundleType.PLTZ:
        from scitex.plt.io import load_layered_pltz_bundle

        result.update(load_layered_pltz_bundle(bundle_dir))
    elif bundle_type == BundleType.STATSZ:
        from scitex.stats.io._bundle import load_statsz_bundle

        result.update(load_statsz_bundle(bundle_dir))

    return result


def save(
    data: Dict[str, Any],
    path: Union[str, Path],
    bundle_type: Optional[str] = None,
    as_zip: bool = False,
    atomic: bool = True,
) -> Path:
    """Save data as a bundle.

    Args:
        data: Bundle data dictionary.
        path: Output path (with or without .d suffix).
        bundle_type: Bundle type ('figz', 'pltz', 'statsz'). Auto-detected if None.
        as_zip: If True, save as ZIP archive.
        atomic: If True, use atomic write (temp file + rename) for ZIP.

    Returns:
        Path to saved bundle.

    Example:
        >>> save({"spec": {...}, "data": df}, "plot.pltz", as_zip=True)
        Path('plot.pltz')
    """
    p = Path(path)

    if bundle_type is None:
        bundle_type = get_type(p)
        if bundle_type is None:
            raise ValueError(f"Cannot determine bundle type from path: {path}")

    # Determine if saving as directory or ZIP
    if as_zip or (p.suffix in EXTENSIONS and not str(p).endswith(".d")):
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

    # For direct ZIP saving with atomic writes
    if save_as_zip and atomic and bundle_type != BundleType.FIGZ:
        from ._zip import ZipBundle

        with ZipBundle(zip_path, mode="w") as zb:
            if "spec" in data:
                zb.write_json("spec.json", data["spec"])

            if "style" in data:
                zb.write_json("style.json", data["style"])

            if "data" in data and data["data"] is not None:
                import pandas as pd

                if isinstance(data["data"], pd.DataFrame):
                    zb.write_csv("data.csv", data["data"])

            for key in ["png", "svg", "pdf"]:
                if key in data and data[key] is not None:
                    export_data = data[key]
                    if isinstance(export_data, bytes):
                        zb.write_bytes(f"exports/figure.{key}", export_data)

        return zip_path

    dir_path.mkdir(parents=True, exist_ok=True)

    # Delegate to domain-specific savers
    if bundle_type == BundleType.FIGZ:
        from scitex.canvas.io._bundle import save_figz_bundle

        save_figz_bundle(data, dir_path)
    elif bundle_type == BundleType.PLTZ:
        from scitex.plt.io._bundle import save_pltz_bundle

        save_pltz_bundle(data, dir_path)
    elif bundle_type == BundleType.STATSZ:
        from scitex.stats.io._bundle import save_statsz_bundle

        save_statsz_bundle(data, dir_path)
    else:
        raise ValueError(f"Unknown bundle type: {bundle_type}")

    if save_as_zip:
        pack(dir_path, zip_path)
        shutil.rmtree(dir_path)
        return zip_path

    return dir_path


def copy(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Copy a bundle from source to destination.

    Handles both directory (.d) and ZIP formats transparently.
    If source is ZIP, extracts to destination directory.
    If source is directory, copies to destination directory.

    Args:
        src: Source bundle path (directory or ZIP).
        dst: Destination path (will be created as directory).
        overwrite: If True, overwrite existing destination.

    Returns:
        Path to copied bundle (directory form).

    Raises:
        BundleNotFoundError: If source bundle doesn't exist.
        FileExistsError: If destination exists and overwrite=False.

    Example:
        >>> copy("gallery/line/plot.pltz.d", "my_project/A.pltz.d")
        >>> copy("template.pltz", "output/panel.pltz.d")
    """
    src_path = Path(src)
    dst_path = Path(dst)

    # Validate source exists
    if not src_path.exists():
        if src_path.suffix in EXTENSIONS:
            alt_path = Path(str(src_path) + ".d")
            if alt_path.exists():
                src_path = alt_path
            else:
                raise BundleNotFoundError(
                    f"Bundle not found: {src} (also tried {alt_path})"
                )
        elif str(src_path).endswith(".d"):
            alt_path = Path(str(src_path)[:-2])
            if alt_path.exists():
                src_path = alt_path
            else:
                raise BundleNotFoundError(
                    f"Bundle not found: {src} (also tried {alt_path})"
                )
        else:
            raise BundleNotFoundError(f"Bundle not found: {src}")

    # Handle destination
    if dst_path.exists():
        if overwrite:
            if dst_path.is_dir():
                shutil.rmtree(dst_path)
            else:
                dst_path.unlink()
        else:
            raise FileExistsError(f"Destination exists: {dst}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy based on source type
    if src_path.is_dir():
        shutil.copytree(src_path, dst_path)
    elif src_path.suffix in EXTENSIONS or zipfile.is_zipfile(src_path):
        unpack(src_path, dst_path)
    else:
        raise ValueError(f"Unknown bundle format: {src_path}")

    return dst_path


# EOF
