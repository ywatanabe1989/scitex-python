# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_figure_metadata.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: scitex/plt/utils/metadata/_figure_metadata.py
# 
# """
# Figure-level metadata collection.
# 
# This module provides functions to collect figure-level metadata including
# dimensions, DPI, and overall figure properties.
# """
# 
# from typing import Dict
# from ._dimensions import _extract_figure_dimensions
# 
# 
# def _collect_figure_metadata(fig) -> dict:
#     """
#     Collect figure-level metadata.
# 
#     Parameters
#     ----------
#     fig : matplotlib.figure.Figure
#         The figure to collect metadata from
# 
#     Returns
#     -------
#     dict
#         Figure metadata with size_mm, size_inch, size_px, and dpi
#     """
#     return _extract_figure_dimensions(fig)
# 
# 
# def _initialize_metadata_structure() -> dict:
#     """
#     Initialize the base metadata structure with schema info.
# 
#     Returns
#     -------
#     dict
#         Base metadata dictionary with schema, version, UUID, and runtime info
#     """
#     import datetime
#     import uuid
#     import matplotlib
#     import scitex
# 
#     metadata = {
#         "scitex_schema": "scitex.plt.figure",
#         "scitex_schema_version": "0.1.0",
#         "figure_uuid": str(uuid.uuid4()),
#         "runtime": {
#             "scitex_version": scitex.__version__,
#             "matplotlib_version": matplotlib.__version__,
#             "created_at": datetime.datetime.now().isoformat(),
#         },
#     }
# 
#     return metadata

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/metadata/_figure_metadata.py
# --------------------------------------------------------------------------------
