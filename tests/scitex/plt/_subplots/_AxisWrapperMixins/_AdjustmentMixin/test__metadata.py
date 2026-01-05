# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_AdjustmentMixin/_metadata.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: _metadata.py - Axis metadata and labels
# 
# """Mixin for axis labels, titles, and metadata."""
# 
# import os
# from typing import Optional
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# 
# class MetadataMixin:
#     """Mixin for setting axis labels, titles, and metadata."""
# 
#     def _get_ax_module(self):
#         """Lazy import ax module to avoid circular imports."""
#         from .....plt import ax as ax_module
#         return ax_module
# 
#     def set_xyt(
#         self,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         t: Optional[str] = None,
#         format_labels: bool = True,
#     ) -> None:
#         """Set xlabel, ylabel, and title."""
#         self._axis_mpl = self._get_ax_module().set_xyt(
#             self._axis_mpl,
#             x=x,
#             y=y,
#             t=t,
#             format_labels=format_labels,
#         )
# 
#     def set_xytc(
#         self,
#         x: Optional[str] = None,
#         y: Optional[str] = None,
#         t: Optional[str] = None,
#         c: Optional[str] = None,
#         format_labels: bool = True,
#     ) -> None:
#         """Set xlabel, ylabel, title, and caption for automatic saving.
# 
#         Parameters
#         ----------
#         x : str, optional
#             X-axis label
#         y : str, optional
#             Y-axis label
#         t : str, optional
#             Title
#         c : str, optional
#             Caption to be saved automatically with scitex.io.save()
#         format_labels : bool, optional
#             Whether to apply automatic formatting, by default True
#         """
#         self._axis_mpl = self._get_ax_module().set_xytc(
#             self._axis_mpl,
#             x=x,
#             y=y,
#             t=t,
#             c=c,
#             format_labels=format_labels,
#         )
# 
#         if c is not False and c is not None:
#             self._scitex_caption = c
# 
#     def set_supxyt(
#         self,
#         xlabel: Optional[str] = None,
#         ylabel: Optional[str] = None,
#         title: Optional[str] = None,
#         format_labels: bool = True,
#     ) -> None:
#         """Set figure-level xlabel, ylabel, and title (suptitle)."""
#         self._axis_mpl = self._get_ax_module().set_supxyt(
#             self._axis_mpl,
#             xlabel=xlabel,
#             ylabel=ylabel,
#             title=title,
#             format_labels=format_labels,
#         )
# 
#     def set_supxytc(
#         self,
#         xlabel: Optional[str] = None,
#         ylabel: Optional[str] = None,
#         title: Optional[str] = None,
#         caption: Optional[str] = None,
#         format_labels: bool = True,
#     ) -> None:
#         """Set figure-level xlabel, ylabel, title, and caption.
# 
#         Parameters
#         ----------
#         xlabel : str, optional
#             Figure-level X-axis label
#         ylabel : str, optional
#             Figure-level Y-axis label
#         title : str, optional
#             Figure-level title (suptitle)
#         caption : str, optional
#             Figure-level caption for automatic saving
#         format_labels : bool, optional
#             Whether to apply automatic formatting
#         """
#         self._axis_mpl = self._get_ax_module().set_supxytc(
#             self._axis_mpl,
#             xlabel=xlabel,
#             ylabel=ylabel,
#             title=title,
#             caption=caption,
#             format_labels=format_labels,
#         )
# 
#         if caption is not False and caption is not None:
#             fig = self._axis_mpl.get_figure()
#             fig._scitex_main_caption = caption
# 
#     def set_meta(
#         self,
#         caption=None,
#         methods=None,
#         stats=None,
#         keywords=None,
#         experimental_details=None,
#         journal_style=None,
#         significance=None,
#         **kwargs,
#     ) -> None:
#         """Set comprehensive scientific metadata with YAML export capability.
# 
#         Parameters
#         ----------
#         caption : str, optional
#             Figure caption text
#         methods : str, optional
#             Experimental methods description
#         stats : str, optional
#             Statistical analysis details
#         keywords : List[str], optional
#             Keywords for categorization
#         experimental_details : Dict[str, Any], optional
#             Structured experimental parameters
#         journal_style : str, optional
#             Target journal style
#         significance : str, optional
#             Significance statement
#         **kwargs : additional metadata
#         """
#         self._axis_mpl = self._get_ax_module().set_meta(
#             self._axis_mpl,
#             caption=caption,
#             methods=methods,
#             stats=stats,
#             keywords=keywords,
#             experimental_details=experimental_details,
#             journal_style=journal_style,
#             significance=significance,
#             **kwargs,
#         )
# 
#     def set_figure_meta(
#         self,
#         caption=None,
#         methods=None,
#         stats=None,
#         significance=None,
#         funding=None,
#         conflicts=None,
#         data_availability=None,
#         **kwargs,
#     ) -> None:
#         """Set figure-level metadata for multi-panel figures.
# 
#         Parameters
#         ----------
#         caption : str, optional
#             Figure-level caption
#         methods : str, optional
#             Overall experimental methods
#         stats : str, optional
#             Overall statistical approach
#         significance : str, optional
#             Significance and implications
#         funding : str, optional
#             Funding acknowledgments
#         conflicts : str, optional
#             Conflict of interest statement
#         data_availability : str, optional
#             Data availability statement
#         **kwargs : additional metadata
#         """
#         self._axis_mpl = self._get_ax_module().set_figure_meta(
#             self._axis_mpl,
#             caption=caption,
#             methods=methods,
#             stats=stats,
#             significance=significance,
#             funding=funding,
#             conflicts=conflicts,
#             data_availability=data_availability,
#             **kwargs,
#         )
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_AdjustmentMixin/_metadata.py
# --------------------------------------------------------------------------------
