# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_conversion/_dict2bundle.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_conversion/dict2bundle.py
# 
# """Convert dictionary to FTS bundle."""
# 
# from pathlib import Path
# from typing import TYPE_CHECKING, Any, Dict, Union
# 
# if TYPE_CHECKING:
#     from .._FSB import FTS
# 
# 
# def dict_to_bundle(
#     data: Dict[str, Any],
#     path: Union[str, Path],
# ) -> "FTS":
#     """Create Figure-Statistics Bundle from dictionary data.
# 
#     Args:
#         data: Dictionary with 'node', 'encoding', 'theme', etc.
#         path: Path for the new bundle.
# 
#     Returns:
#         Figure-Statistics Bundle instance.
#     """
#     from .._FSB import FTS
# 
#     node_data = data.get("node", {})
#     node_type = node_data.get("type", "plot")
#     name = node_data.get("name")
#     size_mm = node_data.get("size_mm")
# 
#     bundle = FTS(path, create=True, node_type=node_type, name=name, size_mm=size_mm)
# 
#     if "encoding" in data:
#         bundle.encoding = data["encoding"]
#     if "theme" in data:
#         bundle.theme = data["theme"]
#     if "stats" in data:
#         bundle.stats = data["stats"]
#     if "data_info" in data:
#         bundle.data_info = data["data_info"]
# 
#     return bundle
# 
# 
# __all__ = ["dict_to_bundle"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_conversion/_dict2bundle.py
# --------------------------------------------------------------------------------
