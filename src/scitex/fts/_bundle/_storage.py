#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_storage.py

"""
Storage Abstraction for FTS Bundles.

Provides a unified interface for reading/writing bundle contents
regardless of whether the bundle is a ZIP file or directory.

Usage:
    storage = get_storage(Path("bundle.zip"))  # or Path("bundle/")
    data = storage.read("node.json")
    storage.write("encoding.json", json_bytes)
"""

import json
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union


class Storage(ABC):
    """Abstract storage interface for FTS bundles."""

    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        """Storage path."""
        return self._path

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if file exists in storage."""
        pass

    @abstractmethod
    def read(self, name: str) -> bytes:
        """Read file contents as bytes."""
        pass

    @abstractmethod
    def write(self, name: str, data: bytes) -> None:
        """Write data to file."""
        pass

    @abstractmethod
    def list(self) -> List[str]:
        """List all files in storage."""
        pass

    def read_json(self, name: str) -> Optional[dict]:
        """Read and parse JSON file."""
        if not self.exists(name):
            return None
        data = self.read(name)
        return json.loads(data.decode("utf-8"))

    def write_json(self, name: str, obj: dict, indent: int = 2) -> None:
        """Write object as JSON file."""
        data = json.dumps(obj, indent=indent).encode("utf-8")
        self.write(name, data)


class ZipStorage(Storage):
    """Storage implementation for ZIP files."""

    def exists(self, name: str) -> bool:
        if not self._path.exists():
            return False
        with zipfile.ZipFile(self._path, "r") as zf:
            return name in zf.namelist()

    def read(self, name: str) -> bytes:
        with zipfile.ZipFile(self._path, "r") as zf:
            return zf.read(name)

    def write(self, name: str, data: bytes) -> None:
        # ZIP files need special handling - read existing, add/update, rewrite
        existing = {}
        if self._path.exists():
            with zipfile.ZipFile(self._path, "r") as zf:
                for item in zf.namelist():
                    if item != name:
                        existing[item] = zf.read(item)

        with zipfile.ZipFile(self._path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item_name, item_data in existing.items():
                zf.writestr(item_name, item_data)
            zf.writestr(name, data)

    def list(self) -> List[str]:
        if not self._path.exists():
            return []
        with zipfile.ZipFile(self._path, "r") as zf:
            return zf.namelist()

    def write_all(self, files: dict) -> None:
        """Write multiple files at once (more efficient for ZIP)."""
        with zipfile.ZipFile(self._path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in files.items():
                if isinstance(data, str):
                    data = data.encode("utf-8")
                zf.writestr(name, data)


class DirStorage(Storage):
    """Storage implementation for directories."""

    def exists(self, name: str) -> bool:
        return (self._path / name).exists()

    def read(self, name: str) -> bytes:
        with open(self._path / name, "rb") as f:
            return f.read()

    def write(self, name: str, data: bytes) -> None:
        file_path = self._path / name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(data)

    def list(self) -> List[str]:
        if not self._path.exists():
            return []
        result = []
        for item in self._path.rglob("*"):
            if item.is_file():
                result.append(str(item.relative_to(self._path)))
        return result

    def write_all(self, files: dict) -> None:
        """Write multiple files."""
        self._path.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            if isinstance(data, str):
                data = data.encode("utf-8")
            self.write(name, data)


def get_storage(path: Union[str, Path]) -> Storage:
    """Get appropriate storage for path.

    Args:
        path: Bundle path (directory or .zip file)

    Returns:
        Storage instance (ZipStorage or DirStorage)
    """
    path = Path(path)
    if path.suffix == ".zip":
        return ZipStorage(path)
    return DirStorage(path)


__all__ = [
    "Storage",
    "ZipStorage",
    "DirStorage",
    "get_storage",
]

# EOF
