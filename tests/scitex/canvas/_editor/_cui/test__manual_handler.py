# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_manual_handler.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-12-14 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/edit/manual_handler.py
# 
# """Manual override handling for figure editor."""
# 
# import hashlib
# from pathlib import Path
# 
# __all__ = ["compute_file_hash", "save_manual_overrides"]
# 
# 
# def compute_file_hash(path: Path) -> str:
#     """Compute SHA256 hash of file contents."""
#     with open(path, "rb") as f:
#         return hashlib.sha256(f.read()).hexdigest()
# 
# 
# def save_manual_overrides(json_path: Path, overrides: dict) -> Path:
#     """
#     Save manual overrides to .manual.json file.
# 
#     Parameters
#     ----------
#     json_path : Path
#         Path to base JSON file
#     overrides : dict
#         Override settings (styles, annotations, etc.)
# 
#     Returns
#     -------
#     Path
#         Path to saved manual.json file
#     """
#     import scitex as stx
# 
#     manual_path = json_path.with_suffix(".manual.json")
# 
#     # Compute hash of base JSON for staleness detection
#     base_hash = compute_file_hash(json_path)
# 
#     manual_data = {
#         "base_file": json_path.name,
#         "base_hash": base_hash,
#         "overrides": overrides,
#     }
# 
#     stx.io.save(manual_data, manual_path)
#     return manual_path
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_editor/_cui/_manual_handler.py
# --------------------------------------------------------------------------------
