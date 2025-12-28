# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_plot_content.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_plot_content.py
# 
# """
# Plot content extraction utilities.
# 
# This module provides functions to extract plot artists, legends, annotations,
# and detect plot types from matplotlib axes.
# 
# This module has been refactored: the implementation is now split across multiple
# specialized modules for better maintainability. This file serves as a backward
# compatibility layer.
# """
# 
# # Import from specialized modules
# from ._artist_extraction import _extract_artists, _extract_traces
# from ._legend_extraction import _extract_legend_info
# from ._plot_type_detection import _detect_plot_type
# 
# __all__ = [
#     "_extract_artists",
#     "_extract_traces",  # Backward compatibility alias
#     "_extract_legend_info",
#     "_detect_plot_type",
# ]

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_plot_content.py
# --------------------------------------------------------------------------------
