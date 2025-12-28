# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_MatplotlibPlotMixin/_base.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: _base.py - Core helper methods for MatplotlibPlotMixin
# 
# """Base mixin with core helper methods for plotting."""
# 
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# 
# class PlotBaseMixin:
#     """Base mixin with core helper methods for plotting."""
# 
#     def _get_ax_module(self):
#         """Lazy import ax module to avoid circular imports."""
#         from .....plt import ax as ax_module
#         return ax_module
# 
#     def _apply_scitex_postprocess(
#         self, method_name, result=None, kwargs=None, args=None
#     ):
#         """Apply scitex post-processing styling after plotting.
# 
#         This ensures all scitex wrapper methods get the same styling
#         as matplotlib methods going through __getattr__ (tick locator, spines, etc.).
#         """
#         from scitex.plt.styles import apply_plot_postprocess
#         apply_plot_postprocess(method_name, result, self._axis_mpl, kwargs or {}, args)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_MatplotlibPlotMixin/_base.py
# --------------------------------------------------------------------------------
