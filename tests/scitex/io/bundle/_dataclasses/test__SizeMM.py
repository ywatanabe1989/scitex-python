# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_SizeMM.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_dataclasses/SizeMM.py
# 
# """SizeMM - Physical size in millimeters."""
# 
# from dataclasses import dataclass
# from typing import Any, Dict, Tuple
# 
# 
# @dataclass
# class SizeMM:
#     """Physical size in millimeters.
# 
#     Used for print-ready figure dimensions.
#     """
# 
#     width: float = 170.0  # Single column default
#     height: float = 120.0
# 
#     def to_dict(self) -> Dict[str, float]:
#         return {"width": self.width, "height": self.height}
# 
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "SizeMM":
#         return cls(
#             width=data.get("width", 170.0),
#             height=data.get("height", 120.0),
#         )
# 
#     def to_inches(self) -> Tuple[float, float]:
#         """Convert to inches (for matplotlib)."""
#         return (self.width / 25.4, self.height / 25.4)
# 
# 
# __all__ = ["SizeMM"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_dataclasses/_SizeMM.py
# --------------------------------------------------------------------------------
