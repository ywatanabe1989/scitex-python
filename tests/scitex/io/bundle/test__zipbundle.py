# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_zipbundle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_zipbundle.py
# 
# """ZipBundle - Simple ZIP file wrapper for FTS bundles."""
# 
# import json
# import zipfile
# from pathlib import Path
# from typing import Any, Dict, Optional, Union
# 
# __all__ = ["ZipBundle"]
# 
# 
# class ZipBundle:
#     """Simple ZIP file wrapper for reading/writing bundle data.
# 
#     Provides a context manager interface for working with ZIP files.
# 
#     Usage:
#         # Read mode
#         with ZipBundle("bundle.zip", mode="r") as bundle:
#             spec = bundle.read_json("spec.json")
#             data = bundle.read_bytes("data.csv")
# 
#         # Write/append mode
#         with ZipBundle("bundle.zip", mode="a") as bundle:
#             bundle.write_json("spec.json", {"type": "plot"})
#             bundle.write_bytes("data.csv", csv_bytes)
#     """
# 
#     def __init__(self, path: Union[str, Path], mode: str = "r"):
#         """Initialize ZipBundle.
# 
#         Args:
#             path: Path to ZIP file.
#             mode: File mode ('r' for read, 'w' for write, 'a' for append).
#         """
#         self.path = Path(path)
#         self.mode = mode
#         self._zf: Optional[zipfile.ZipFile] = None
# 
#     def __enter__(self) -> "ZipBundle":
#         """Enter context manager."""
#         self._zf = zipfile.ZipFile(self.path, self.mode)
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         """Exit context manager."""
#         if self._zf:
#             self._zf.close()
#             self._zf = None
# 
#     def read_json(self, name: str) -> Dict[str, Any]:
#         """Read JSON file from ZIP.
# 
#         Args:
#             name: File name within ZIP.
# 
#         Returns:
#             Parsed JSON data.
#         """
#         if not self._zf:
#             raise RuntimeError("ZipBundle not open")
# 
#         # Try direct path first
#         try:
#             data = self._zf.read(name)
#             return json.loads(data.decode("utf-8"))
#         except KeyError:
#             pass
# 
#         # Try with .d/ directory prefix
#         for item in self._zf.namelist():
#             if item.endswith(f"/{name}") or item == name:
#                 data = self._zf.read(item)
#                 return json.loads(data.decode("utf-8"))
# 
#         raise FileNotFoundError(f"File not found in ZIP: {name}")
# 
#     def read_bytes(self, name: str) -> bytes:
#         """Read raw bytes from ZIP.
# 
#         Args:
#             name: File name within ZIP.
# 
#         Returns:
#             Raw file bytes.
#         """
#         if not self._zf:
#             raise RuntimeError("ZipBundle not open")
# 
#         # Try direct path first
#         try:
#             return self._zf.read(name)
#         except KeyError:
#             pass
# 
#         # Try with .d/ directory prefix
#         for item in self._zf.namelist():
#             if item.endswith(f"/{name}") or item == name:
#                 return self._zf.read(item)
# 
#         raise FileNotFoundError(f"File not found in ZIP: {name}")
# 
#     def write_json(self, name: str, data: Dict[str, Any], indent: int = 2) -> None:
#         """Write JSON file to ZIP.
# 
#         Args:
#             name: File name within ZIP.
#             data: Data to write.
#             indent: JSON indentation level.
#         """
#         if not self._zf:
#             raise RuntimeError("ZipBundle not open")
# 
#         json_str = json.dumps(data, indent=indent)
#         self._zf.writestr(name, json_str.encode("utf-8"))
# 
#     def write_bytes(self, name: str, data: bytes) -> None:
#         """Write raw bytes to ZIP.
# 
#         Args:
#             name: File name within ZIP.
#             data: Data to write.
#         """
#         if not self._zf:
#             raise RuntimeError("ZipBundle not open")
# 
#         self._zf.writestr(name, data)
# 
#     def namelist(self) -> list:
#         """List files in ZIP.
# 
#         Returns:
#             List of file names.
#         """
#         if not self._zf:
#             raise RuntimeError("ZipBundle not open")
#         return self._zf.namelist()
# 
#     def exists(self, name: str) -> bool:
#         """Check if file exists in ZIP.
# 
#         Args:
#             name: File name to check.
# 
#         Returns:
#             True if file exists.
#         """
#         if not self._zf:
#             raise RuntimeError("ZipBundle not open")
# 
#         if name in self._zf.namelist():
#             return True
# 
#         # Check with .d/ directory prefix
#         for item in self._zf.namelist():
#             if item.endswith(f"/{name}"):
#                 return True
# 
#         return False
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_zipbundle.py
# --------------------------------------------------------------------------------
