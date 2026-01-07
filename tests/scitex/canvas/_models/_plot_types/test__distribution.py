# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_distribution.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_distribution.py
# 
# """Distribution plot configurations (ECDF, KDE)."""
# 
# from dataclasses import dataclass
# from typing import List, Optional
# 
# 
# @dataclass
# class ECDFConfig:
#     """ECDF configuration (scitex.plt.ax.stx_ecdf)."""
# 
#     data: List[float]
#     color: Optional[str] = None
#     linewidth_mm: Optional[float] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# @dataclass
# class KDEConfig:
#     """KDE configuration (scitex.plt.ax.stx_kde)."""
# 
#     data: List[float]
#     bw_method: Optional[str] = None
#     color: Optional[str] = None
#     linewidth_mm: Optional[float] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# __all__ = ["ECDFConfig", "KDEConfig"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_distribution.py
# --------------------------------------------------------------------------------
