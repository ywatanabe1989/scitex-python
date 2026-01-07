# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_violin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_violin.py
# 
# """Violin plot configurations."""
# 
# from dataclasses import dataclass
# from typing import List, Optional
# 
# 
# @dataclass
# class ViolinPlotConfig:
#     """Violin plot configuration."""
# 
#     data: List[List[float]]
#     positions: Optional[List[float]] = None
#     widths: Optional[float] = 0.5
#     showmeans: bool = False
#     showmedians: bool = False
#     showextrema: bool = True
#     id: Optional[str] = None
# 
# 
# @dataclass
# class ViolinConfig:
#     """Violin plot configuration (scitex.plt.ax.stx_violin)."""
# 
#     data: List[List[float]]
#     labels: Optional[List[str]] = None
#     colors: Optional[List[str]] = None
#     id: Optional[str] = None
# 
# 
# __all__ = ["ViolinPlotConfig", "ViolinConfig"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_violin.py
# --------------------------------------------------------------------------------
