# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_box.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_box.py
# 
# """Box plot configurations."""
# 
# from dataclasses import dataclass
# from typing import List, Optional, Union
# 
# 
# @dataclass
# class BoxPlotConfig:
#     """Box plot configuration."""
# 
#     data: Union[List[List[float]], List[float]]
#     labels: Optional[List[str]] = None
#     positions: Optional[List[float]] = None
#     widths: Optional[float] = None
#     linewidth_mm: Optional[float] = None
#     showfliers: bool = True
#     showmeans: bool = False
#     id: Optional[str] = None
# 
# 
# @dataclass
# class BoxConfig:
#     """Box plot configuration (scitex.plt.ax.stx_box)."""
# 
#     data: List[float]
#     color: Optional[str] = None
#     linewidth_mm: Optional[float] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# __all__ = ["BoxPlotConfig", "BoxConfig"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_box.py
# --------------------------------------------------------------------------------
