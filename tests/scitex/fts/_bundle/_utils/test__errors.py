# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_utils/_errors.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_bundle/_utils/_errors.py
# 
# """FTS Bundle error classes."""
# 
# 
# class BundleError(Exception):
#     """Base exception for bundle operations."""
# 
#     pass
# 
# 
# class BundleValidationError(BundleError, ValueError):
#     """Error raised when bundle validation fails."""
# 
#     pass
# 
# 
# class BundleNotFoundError(BundleError, FileNotFoundError):
#     """Error raised when a bundle is not found."""
# 
#     pass
# 
# 
# class NestedBundleNotFoundError(BundleNotFoundError):
#     """Error raised when a nested bundle or file within it is not found."""
# 
#     pass
# 
# 
# class CircularReferenceError(BundleValidationError):
#     """Error raised when circular reference is detected in bundle hierarchy.
# 
#     This occurs when a bundle references itself directly or indirectly
#     through its children, detected via bundle_id tracking.
#     """
# 
#     pass
# 
# 
# class DepthLimitError(BundleValidationError):
#     """Error raised when bundle nesting exceeds max_depth constraint.
# 
#     The max_depth is defined per bundle type in TYPE_DEFAULTS.
#     Default is 3 for figures, 1 for leaf types.
#     """
# 
#     pass
# 
# 
# class ConstraintError(BundleValidationError):
#     """Error raised when bundle violates its type constraints.
# 
#     Examples:
#     - Leaf type (plot, stats) has children
#     - Unknown type specified
#     """
# 
#     pass
# 
# 
# __all__ = [
#     "BundleError",
#     "BundleValidationError",
#     "BundleNotFoundError",
#     "NestedBundleNotFoundError",
#     "CircularReferenceError",
#     "DepthLimitError",
#     "ConstraintError",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_utils/_errors.py
# --------------------------------------------------------------------------------
