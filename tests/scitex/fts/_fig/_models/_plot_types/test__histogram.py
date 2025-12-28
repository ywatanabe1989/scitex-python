# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_histogram.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2025-12-20
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/_histogram.py
# 
# """Histogram configurations."""
# 
# from dataclasses import dataclass
# from typing import List, Optional, Union
# 
# 
# @dataclass
# class HistPlotConfig:
#     """Histogram configuration."""
# 
#     x: List[float]
#     bins: Union[int, List[float]] = 10
#     density: bool = False
#     cumulative: bool = False
#     histtype: str = "bar"
#     color: Optional[str] = None
#     alpha: Optional[float] = None
#     edgecolor: Optional[str] = None
#     range: Optional[tuple] = None
#     label: Optional[str] = None
#     id: Optional[str] = None
# 
# 
# __all__ = ["HistPlotConfig"]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_fig/_models/_plot_types/_histogram.py
# --------------------------------------------------------------------------------
