# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_bar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_bar.py
# 
# """Bar plot configurations."""
# 
# from dataclasses import dataclass
# from typing import List, Optional, Union
# 
# 
# @dataclass
# class BarPlotConfig:
#     """Bar plot configuration."""
# 
#     x: Union[List[float], List[str]]
#     height: List[float]
#     width: Optional[float] = 0.8
#     color: Optional[str] = None
#     alpha: Optional[float] = None
#     edge_thickness_mm: Optional[float] = None
#     edgecolor: Optional[str] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# @dataclass
# class BarHPlotConfig:
#     """Horizontal bar plot configuration."""
# 
#     y: Union[List[float], List[str]]
#     width: List[float]
#     height: Optional[float] = 0.8
#     color: Optional[str] = None
#     alpha: Optional[float] = None
#     edge_thickness_mm: Optional[float] = None
#     edgecolor: Optional[str] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# __all__ = ["BarPlotConfig", "BarHPlotConfig"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_bar.py
# --------------------------------------------------------------------------------
