# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_storage.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_storage.py
# 
# """
# Storage Abstraction for FTS Bundles.
# 
# Provides a unified interface for reading/writing bundle contents
# regardless of whether the bundle is a ZIP file or directory.
# 
# Usage:
#     storage = get_storage(Path("bundle.zip"))  # or Path("bundle/")
#     data = storage.read("node.json")
#     storage.write("encoding.json", json_bytes)
# """
# 
# import json
# import zipfile
# from abc import ABC, abstractmethod
# from pathlib import Path
# from typing import List, Optional, Union
# 
# 
# class Storage(ABC):
#     """Abstract storage interface for FTS bundles."""
# 
#     def __init__(self, path: Path):
#         self._path = path
# 
#     @property
#     def path(self) -> Path:
#         """Storage path."""
#         return self._path
# 
#     @abstractmethod
#     def exists(self, name: str) -> bool:
#         """Check if file exists in storage."""
#         pass
# 
#     @abstractmethod
#     def read(self, name: str) -> bytes:
#         """Read file contents as bytes."""
#         pass
# 
#     @abstractmethod
#     def write(self, name: str, data: bytes) -> None:
#         """Write data to file."""
#         pass
# 
#     @abstractmethod
#     def list(self) -> List[str]:
#         """List all files in storage."""
#         pass
# 
#     def read_json(self, name: str) -> Optional[dict]:
#         """Read and parse JSON file."""
#         if not self.exists(name):
#             return None
#         data = self.read(name)
#         return json.loads(data.decode("utf-8"))
# 
#     def write_json(self, name: str, obj: dict, indent: int = 2) -> None:
#         """Write object as JSON file."""
#         data = json.dumps(obj, indent=indent).encode("utf-8")
#         self.write(name, data)
# 
# 
# class ZipStorage(Storage):
#     """Storage implementation for ZIP files.
# 
#     ZIP files are structured with a top-level directory named after the bundle.
#     For example, my_figure.zip contains:
#         my_figure/
#             node.json
#             encoding.json
#             theme.json
#     This ensures `unzip my_figure.zip` creates a my_figure/ directory.
#     """
# 
#     @property
#     def _prefix(self) -> str:
#         """Top-level directory name inside the ZIP (bundle stem)."""
#         return self._path.stem + "/"
# 
#     def _prefixed(self, name: str) -> str:
#         """Add prefix to file path."""
#         return self._prefix + name
# 
#     def _unprefixed(self, name: str) -> str:
#         """Remove prefix from file path."""
#         if name.startswith(self._prefix):
#             return name[len(self._prefix) :]
#         return name
# 
#     def exists(self, name: str) -> bool:
#         if not self._path.exists():
#             return False
#         with zipfile.ZipFile(self._path, "r") as zf:
#             # Check both prefixed and unprefixed for backwards compatibility
#             return self._prefixed(name) in zf.namelist() or name in zf.namelist()
# 
#     def read(self, name: str) -> bytes:
#         with zipfile.ZipFile(self._path, "r") as zf:
#             # Try prefixed first, fall back to unprefixed for backwards compatibility
#             prefixed = self._prefixed(name)
#             if prefixed in zf.namelist():
#                 return zf.read(prefixed)
#             return zf.read(name)
# 
#     def write(self, name: str, data: bytes) -> None:
#         # ZIP files need special handling - read existing, add/update, rewrite
#         existing = {}
#         prefixed_name = self._prefixed(name)
#         if self._path.exists():
#             with zipfile.ZipFile(self._path, "r") as zf:
#                 for item in zf.namelist():
#                     if item != prefixed_name:
#                         existing[item] = zf.read(item)
# 
#         with zipfile.ZipFile(self._path, "w", zipfile.ZIP_DEFLATED) as zf:
#             for item_name, item_data in existing.items():
#                 zf.writestr(item_name, item_data)
#             zf.writestr(prefixed_name, data)
# 
#     def list(self) -> List[str]:
#         if not self._path.exists():
#             return []
#         with zipfile.ZipFile(self._path, "r") as zf:
#             # Return unprefixed names for API consistency
#             return [self._unprefixed(n) for n in zf.namelist() if not n.endswith("/")]
# 
#     def write_all(self, files: dict) -> None:
#         """Write multiple files at once (more efficient for ZIP).
# 
#         Files are stored under a top-level directory matching the bundle name.
#         """
#         with zipfile.ZipFile(self._path, "w", zipfile.ZIP_DEFLATED) as zf:
#             for name, data in files.items():
#                 if isinstance(data, str):
#                     data = data.encode("utf-8")
#                 zf.writestr(self._prefixed(name), data)
# 
# 
# class DirStorage(Storage):
#     """Storage implementation for directories."""
# 
#     def exists(self, name: str) -> bool:
#         return (self._path / name).exists()
# 
#     def read(self, name: str) -> bytes:
#         with open(self._path / name, "rb") as f:
#             return f.read()
# 
#     def write(self, name: str, data: bytes) -> None:
#         file_path = self._path / name
#         file_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(file_path, "wb") as f:
#             f.write(data)
# 
#     def list(self) -> List[str]:
#         if not self._path.exists():
#             return []
#         result = []
#         for item in self._path.rglob("*"):
#             if item.is_file():
#                 result.append(str(item.relative_to(self._path)))
#         return result
# 
#     def write_all(self, files: dict) -> None:
#         """Write multiple files."""
#         self._path.mkdir(parents=True, exist_ok=True)
#         for name, data in files.items():
#             if isinstance(data, str):
#                 data = data.encode("utf-8")
#             self.write(name, data)
# 
# 
# def get_storage(path: Union[str, Path]) -> Storage:
#     """Get appropriate storage for path.
# 
#     Args:
#         path: Bundle path (directory or .zip file)
# 
#     Returns:
#         Storage instance (ZipStorage or DirStorage)
#     """
#     path = Path(path)
#     if path.suffix == ".zip":
#         return ZipStorage(path)
#     return DirStorage(path)
# 
# 
# __all__ = [
#     "Storage",
#     "ZipStorage",
#     "DirStorage",
#     "get_storage",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_storage.py
# --------------------------------------------------------------------------------
