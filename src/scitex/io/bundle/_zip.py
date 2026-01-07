#!/usr/bin/env python3
# Timestamp: "2025-12-16 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/bundle/_zip.py

"""
SciTeX ZipBundle - In-memory zip archive handler with atomic writes.

Provides efficient access to .figure.zip, .plot.zip, .stats.zip bundles without
extracting to disk. Supports:
    - In-memory file reading
    - Atomic writes using temp files
    - Context manager protocol
    - Dict-like access to files
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import pandas as pd

__all__ = [
    "ZipBundle",
    "open",
    "create",
    "zip_directory",
]


class ZipBundle:
    """In-memory zip bundle handler.

    Provides efficient read/write access to zip archives without extracting
    to disk. Files are read into memory on demand and writes are atomic.

    Usage:
        # Reading
        with ZipBundle("figure.figure.zip") as bundle:
            spec = bundle.read_json("spec.json")
            csv_data = bundle.read_csv("data.csv")
            png_bytes = bundle.read_bytes("exports/figure.png")

        # Writing
        with ZipBundle("output.plot.zip", mode="w") as bundle:
            bundle.write_json("spec.json", spec_dict)
            bundle.write_csv("data.csv", dataframe)
            bundle.write_bytes("exports/plot.png", png_bytes)

        # Modifying (read then write atomically)
        with ZipBundle("figure.figure.zip", mode="a") as bundle:
            spec = bundle.read_json("spec.json")
            spec["title"] = "Updated"
            bundle.write_json("spec.json", spec)
    """

    def __init__(
        self,
        path: Union[str, Path],
        mode: str = "r",
        compression: int = zipfile.ZIP_DEFLATED,
    ):
        """Initialize ZipBundle.

        Args:
            path: Path to zip file (.figure.zip, .plot.zip, .stats.zip)
            mode: 'r' for read, 'w' for write, 'a' for append/modify
            compression: ZIP compression method (default: ZIP_DEFLATED)
        """
        self.path = Path(path)
        self.mode = mode
        self.compression = compression
        self._zipfile: Optional[zipfile.ZipFile] = None
        self._cache: Dict[str, bytes] = {}
        self._pending_writes: Dict[str, bytes] = {}
        self._temp_path: Optional[Path] = None
        self._closed = False

    def __enter__(self) -> "ZipBundle":
        """Enter context manager."""
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager with atomic commit on success."""
        if exc_type is None and self.mode in ("w", "a"):
            self._atomic_commit()
        self.close()

    def _open(self) -> None:
        """Open the zip bundle."""
        if self.mode == "r":
            if not self.path.exists():
                raise FileNotFoundError(f"Bundle not found: {self.path}")
            self._zipfile = zipfile.ZipFile(self.path, "r")
        elif self.mode == "w":
            self._temp_path = Path(tempfile.mktemp(suffix=self.path.suffix))
            self._zipfile = zipfile.ZipFile(self._temp_path, "w", self.compression)
        elif self.mode == "a":
            if self.path.exists():
                with zipfile.ZipFile(self.path, "r") as zf:
                    for name in zf.namelist():
                        self._cache[name] = zf.read(name)
            self._temp_path = Path(tempfile.mktemp(suffix=self.path.suffix))
            self._zipfile = zipfile.ZipFile(self._temp_path, "w", self.compression)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def close(self) -> None:
        """Close the zip bundle."""
        if self._zipfile and not self._closed:
            self._zipfile.close()
            self._closed = True
            if self._temp_path and self._temp_path.exists():
                if self.mode in ("w", "a") and not self.path.exists():
                    self._temp_path.unlink()

    def _atomic_commit(self) -> None:
        """Atomically commit writes by renaming temp file."""
        if self._temp_path is None:
            return

        self._zipfile.close()
        self._closed = True

        if os.name == "nt" and self.path.exists():
            backup_path = self.path.with_suffix(self.path.suffix + ".bak")
            self.path.rename(backup_path)
            try:
                self._temp_path.rename(self.path)
                backup_path.unlink()
            except Exception:
                backup_path.rename(self.path)
                raise
        else:
            self._temp_path.rename(self.path)

    # =========================================================================
    # File listing
    # =========================================================================

    def namelist(self) -> List[str]:
        """List all files in the bundle."""
        if self.mode == "r":
            return self._zipfile.namelist()
        else:
            names = set(self._cache.keys())
            names.update(self._pending_writes.keys())
            return sorted(names)

    def __contains__(self, name: str) -> bool:
        """Check if file exists in bundle."""
        return name in self.namelist()

    def __iter__(self) -> Iterator[str]:
        """Iterate over file names in bundle."""
        return iter(self.namelist())

    # =========================================================================
    # Reading
    # =========================================================================

    def read_bytes(self, name: str) -> bytes:
        """Read file as bytes.

        Args:
            name: File path within the zip (e.g., "spec.json", "exports/plot.png")

        Returns:
            File contents as bytes.
        """
        if name in self._pending_writes:
            return self._pending_writes[name]
        if name in self._cache:
            return self._cache[name]

        if self._zipfile is None:
            raise RuntimeError("Bundle not opened")

        try:
            data = self._zipfile.read(name)
            self._cache[name] = data
            return data
        except KeyError:
            dir_name = self.path.name + ".d"
            full_name = f"{dir_name}/{name}"
            try:
                data = self._zipfile.read(full_name)
                self._cache[name] = data
                return data
            except KeyError:
                for arc_name in self._zipfile.namelist():
                    if arc_name.endswith(".d/" + name) or arc_name.endswith(
                        ".d/" + name.replace("/", "/")
                    ):
                        data = self._zipfile.read(arc_name)
                        self._cache[name] = data
                        return data
                raise FileNotFoundError(f"File not found in bundle: {name}")

    def read_text(self, name: str, encoding: str = "utf-8") -> str:
        """Read file as text.

        Args:
            name: File path within the zip.
            encoding: Text encoding (default: utf-8).

        Returns:
            File contents as string.
        """
        return self.read_bytes(name).decode(encoding)

    def read_json(self, name: str = "spec.json") -> Dict[str, Any]:
        """Read and parse JSON file.

        Args:
            name: JSON file path within the zip.

        Returns:
            Parsed JSON as dictionary.
        """
        return json.loads(self.read_text(name))

    def read_csv(self, name: str = "data.csv", **kwargs) -> pd.DataFrame:
        """Read CSV file as pandas DataFrame.

        Args:
            name: CSV file path within the zip.
            **kwargs: Additional arguments passed to pd.read_csv.

        Returns:
            DataFrame with CSV contents.
        """
        data = self.read_bytes(name)
        return pd.read_csv(io.BytesIO(data), **kwargs)

    def read_image(self, name: str) -> bytes:
        """Read image file as bytes.

        Args:
            name: Image file path (e.g., "exports/figure.png").

        Returns:
            Image bytes.
        """
        return self.read_bytes(name)

    # =========================================================================
    # Writing
    # =========================================================================

    def write_bytes(self, name: str, data: bytes) -> None:
        """Write bytes to file in bundle.

        Args:
            name: File path within the zip.
            data: Bytes to write.
        """
        if self.mode == "r":
            raise RuntimeError("Cannot write in read mode")

        self._pending_writes[name] = data
        self._cache[name] = data
        self._zipfile.writestr(name, data)

    def write_text(self, name: str, text: str, encoding: str = "utf-8") -> None:
        """Write text to file in bundle.

        Args:
            name: File path within the zip.
            text: Text to write.
            encoding: Text encoding (default: utf-8).
        """
        self.write_bytes(name, text.encode(encoding))

    def write_json(self, name: str, data: Dict[str, Any], indent: int = 2) -> None:
        """Write dictionary as JSON file.

        Args:
            name: JSON file path within the zip.
            data: Dictionary to serialize.
            indent: JSON indentation (default: 2).
        """
        text = json.dumps(data, indent=indent, ensure_ascii=False)
        self.write_text(name, text)

    def write_csv(
        self, name: str, df: pd.DataFrame, index: bool = False, **kwargs
    ) -> None:
        """Write pandas DataFrame as CSV.

        Args:
            name: CSV file path within the zip.
            df: DataFrame to write.
            index: Include index column (default: False).
            **kwargs: Additional arguments passed to df.to_csv.
        """
        buffer = io.BytesIO()
        df.to_csv(buffer, index=index, **kwargs)
        self.write_bytes(name, buffer.getvalue())

    def write_image(self, name: str, data: bytes) -> None:
        """Write image bytes to bundle.

        Args:
            name: Image file path (e.g., "exports/figure.png").
            data: Image bytes.
        """
        self.write_bytes(name, data)

    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def spec(self) -> Optional[Dict[str, Any]]:
        """Get bundle specification (spec.json)."""
        try:
            return self.read_json("spec.json")
        except FileNotFoundError:
            return None

    @property
    def style(self) -> Optional[Dict[str, Any]]:
        """Get bundle style (style.json)."""
        try:
            return self.read_json("style.json")
        except FileNotFoundError:
            return None

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get bundle data (data.csv)."""
        try:
            return self.read_csv("data.csv")
        except FileNotFoundError:
            return None


def open(path: Union[str, Path], mode: str = "r") -> ZipBundle:
    """Open a bundle for reading or writing.

    Args:
        path: Path to bundle file.
        mode: 'r' for read, 'w' for write, 'a' for append.

    Returns:
        ZipBundle instance (use as context manager).

    Example:
        with open("figure.figure.zip") as bundle:
            spec = bundle.spec
            data = bundle.data
    """
    return ZipBundle(path, mode=mode)


def create(
    path: Union[str, Path],
    spec: Dict[str, Any],
    data: Optional[pd.DataFrame] = None,
    style: Optional[Dict[str, Any]] = None,
    exports: Optional[Dict[str, bytes]] = None,
) -> Path:
    """Create a new bundle with atomic write.

    Args:
        path: Output path (.figure.zip, .plot.zip, or .stats.zip).
        spec: Bundle specification dictionary.
        data: Optional CSV data as DataFrame.
        style: Optional style dictionary.
        exports: Optional dict mapping export paths to bytes.

    Returns:
        Path to created bundle.

    Example:
        create(
            "plot.plot.zip",
            spec={"schema": {"name": "scitex.plt", "version": "1.0"}},
            data=df,
            exports={"exports/plot.png": png_bytes},
        )
    """
    path = Path(path)

    with ZipBundle(path, mode="w") as bundle:
        bundle.write_json("spec.json", spec)

        if data is not None:
            bundle.write_csv("data.csv", data)

        if style is not None:
            bundle.write_json("style.json", style)

        if exports:
            for export_path, export_data in exports.items():
                bundle.write_bytes(export_path, export_data)

    return path


def zip_directory(
    source_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Convert a directory bundle (.plot, .figure) to a ZIP bundle (.plot.zip, .figure.zip).

    Creates a ZIP archive from an existing directory bundle, preserving
    the internal structure. The ZIP file is created atomically using a temp file.

    Args:
        source_dir: Path to directory bundle (e.g., "panel_A.plot", "figure.figure")
        output_path: Optional output ZIP path. If not provided, derives from source_dir
            by appending .zip (e.g., .plot -> .plot.zip)

    Returns:
        Path to created ZIP bundle.

    Raises:
        FileNotFoundError: If source directory doesn't exist.
        ValueError: If source is not a valid bundle directory.

    Example:
        # Convert panel_A.plot to panel_A.plot.zip
        zip_path = zip_directory("panel_A.plot")

        # Convert with custom output path
        zip_path = zip_directory("A.plot", "output/A.plot.zip")
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    if not source_dir.is_dir():
        raise ValueError(f"Source is not a directory: {source_dir}")

    if output_path is None:
        dir_name = source_dir.name
        # New format: .plot -> .plot.zip, .figure -> .figure.zip, .stats -> .stats.zip
        if dir_name.endswith((".plot", ".figure", ".stats")):
            zip_name = dir_name + ".zip"
        else:
            zip_name = dir_name + ".zip"
        output_path = source_dir.parent / zip_name
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = Path(tempfile.mktemp(suffix=output_path.suffix))

    try:
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zf.write(file_path, arcname)

        if output_path.exists():
            output_path.unlink()
        shutil.move(str(temp_path), str(output_path))

    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return output_path


# EOF
