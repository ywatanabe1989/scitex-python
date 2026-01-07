# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_utils/_types.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_utils/_types.py
# 
# """FTS Node type constants and constraints."""
# 
# from typing import Any, Dict
# 
# 
# class NodeType:
#     """Node type constants for FTS bundles.
# 
#     The type is stored in node.json["type"].
# 
#     Usage:
#         from scitex.io.bundle import NodeType
# 
#         if bundle.node.type == NodeType.FIGURE:
#             ...
# 
#     Note:
#         Tables are treated as structured figures in academic contexts.
#         FTS handles figures, tables, and statistics in a unified way.
#     """
# 
#     FIGURE = "figure"
#     PLOT = "plot"
#     TABLE = "table"  # Structured data presentation (demographics, results, etc.)
#     STATS = "stats"
#     IMAGE = "image"
#     TEXT = "text"
#     SHAPE = "shape"
#     SYMBOL = "symbol"
#     COMMENT = "comment"
#     EQUATION = "equation"
# 
# 
# # Legacy alias (deprecated)
# BundleType = NodeType
# 
# 
# # Type-specific default constraints
# TYPE_DEFAULTS: Dict[str, Dict[str, Any]] = {
#     "figure": {"allow_children": True, "max_depth": 3},
#     "plot": {"allow_children": False, "max_depth": 1},
#     "table": {"allow_children": False, "max_depth": 1},  # Tables are leaf nodes
#     "stats": {"allow_children": False, "max_depth": 1},
#     "image": {"allow_children": False, "max_depth": 1},
#     "text": {"allow_children": False, "max_depth": 1},
#     "shape": {"allow_children": False, "max_depth": 1},
#     "symbol": {"allow_children": False, "max_depth": 1},
#     "comment": {"allow_children": False, "max_depth": 1},
#     "equation": {"allow_children": False, "max_depth": 1},
# }
# 
# 
# def get_default_constraints(node_type: str) -> Dict[str, Any]:
#     """Get default constraints for a node type.
# 
#     Args:
#         node_type: Type string (figure, plot, stats, etc.)
# 
#     Returns:
#         Dict with allow_children and max_depth.
#     """
#     return TYPE_DEFAULTS.get(node_type, {"allow_children": False, "max_depth": 1})
# 
# 
# __all__ = [
#     "NodeType",
#     "BundleType",
#     "TYPE_DEFAULTS",
#     "get_default_constraints",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_utils/_types.py
# --------------------------------------------------------------------------------
