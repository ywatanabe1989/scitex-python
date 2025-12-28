# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_conversion/_bundle2dict.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_conversion/bundle2dict.py
# 
# """Convert FTS bundle to dictionary."""
# 
# from typing import TYPE_CHECKING, Any, Dict
# 
# if TYPE_CHECKING:
#     from .._FSB import FTS
# 
# 
# def bundle_to_dict(bundle: "FTS") -> Dict[str, Any]:
#     """Convert Figure-Statistics Bundle to a flat dictionary.
# 
#     Args:
#         bundle: Figure-Statistics Bundle instance.
# 
#     Returns:
#         Dictionary with all bundle data.
#     """
#     result = {
#         "path": str(bundle.path),
#         "is_zip": bundle.path.suffix == ".zip",
#         "type": bundle.bundle_type,
#     }
# 
#     if bundle.node:
#         result["node"] = bundle.node.to_dict()
#     if bundle._encoding:
#         result["encoding"] = bundle._encoding.to_dict()
#     if bundle._theme:
#         result["theme"] = bundle._theme.to_dict()
#     if bundle._stats:
#         result["stats"] = bundle._stats.to_dict()
#     if bundle._data_info:
#         result["data_info"] = bundle._data_info.to_dict()
# 
#     return result
# 
# 
# __all__ = ["bundle_to_dict"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_conversion/_bundle2dict.py
# --------------------------------------------------------------------------------
