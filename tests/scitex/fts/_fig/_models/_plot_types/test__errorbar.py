# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_errorbar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_errorbar.py
# 
# """Error bar and fill configurations."""
# 
# from dataclasses import dataclass
# from typing import List, Optional, Union
# 
# 
# @dataclass
# class ErrorbarPlotConfig:
#     """Error bar plot configuration."""
# 
#     x: List[float]
#     y: List[float]
#     xerr: Optional[Union[float, List[float]]] = None
#     yerr: Optional[Union[float, List[float]]] = None
#     fmt: str = "o-"
#     color: Optional[str] = None
#     capsize: Optional[float] = None
#     capthick: Optional[float] = None
#     thickness_mm: Optional[float] = None
#     cap_width_mm: Optional[float] = None
#     alpha: Optional[float] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# @dataclass
# class FillBetweenConfig:
#     """Fill between configuration."""
# 
#     x: List[float]
#     y1: List[float]
#     y2: List[float]
#     color: Optional[str] = None
#     alpha: Optional[float] = 0.3
#     linewidth: Optional[float] = None
#     edgecolor: Optional[str] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# @dataclass
# class MeanStdConfig:
#     """Mean±Std configuration (scitex.plt.ax.stx_mean_std)."""
# 
#     y_mean: List[float]
#     xx: Optional[List[float]] = None
#     sd: Union[float, List[float]] = 1.0
#     color: Optional[str] = None
#     alpha: Optional[float] = 0.3
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# __all__ = ["ErrorbarPlotConfig", "FillBetweenConfig", "MeanStdConfig"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_errorbar.py
# --------------------------------------------------------------------------------
