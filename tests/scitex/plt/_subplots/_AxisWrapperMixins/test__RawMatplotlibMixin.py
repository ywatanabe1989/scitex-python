# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_RawMatplotlibMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-13 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_RawMatplotlibMixin.py
# 
# """
# Matplotlib aliases (mpl_xxx) for explicit matplotlib-style API.
# 
# Provides consistent naming convention:
# - stx_xxx: scitex-specific methods (ArrayLike input, tracked)
# - sns_xxx: seaborn wrappers (DataFrame input, tracked)
# - mpl_xxx: matplotlib methods (matplotlib-style input, tracked)
# 
# All three API layers track data for reproducibility.
# 
# Usage:
#     ax.stx_line(y)              # ArrayLike input
#     ax.sns_boxplot(data=df, x="group", y="value")  # DataFrame input
#     ax.mpl_plot(x, y)           # matplotlib-style input
#     ax.plot(x, y)               # Same as mpl_plot
# """
# 
# import os
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# 
# class RawMatplotlibMixin:
#     """Mixin providing mpl_xxx aliases for matplotlib-style API.
# 
#     These methods are identical to calling ax.plot(), ax.scatter(), etc.
#     They go through SciTeX's __getattr__ wrapper and are fully tracked.
# 
#     The mpl_* prefix provides:
#     - Explicit naming convention (mpl_* vs stx_* vs sns_*)
#     - Programmatic access via MPL_METHODS registry
#     - Same tracking and styling as regular matplotlib calls
#     """
# 
#     # =========================================================================
#     # Helper to call through __getattr__ wrapper (enables tracking)
#     # =========================================================================
#     def _mpl_call(self, method_name, *args, **kwargs):
#         """Call matplotlib method through __getattr__ wrapper for tracking."""
#         # Use object.__getattribute__ to get the __getattr__ from AxisWrapper
#         # Then call it with the method name to get the tracked wrapper
#         wrapper_class = type(self)
#         # Walk up MRO to find __getattr__ in AxisWrapper
#         for cls in wrapper_class.__mro__:
#             if "__getattr__" in cls.__dict__:
#                 return cls.__getattr__(self, method_name)(*args, **kwargs)
#         # Fallback to direct call if no __getattr__ found
#         return getattr(self._axes_mpl, method_name)(*args, **kwargs)
# 
#     # =========================================================================
#     # Line plots
#     # =========================================================================
#     def mpl_plot(self, *args, **kwargs):
#         """Matplotlib plot() - tracked, identical to ax.plot()."""
#         return self._mpl_call("plot", *args, **kwargs)
# 
#     def mpl_step(self, *args, **kwargs):
#         """Matplotlib step() - tracked, identical to ax.step()."""
#         return self._mpl_call("step", *args, **kwargs)
# 
#     def mpl_stem(self, *args, **kwargs):
#         """Matplotlib stem() - tracked, identical to ax.stem()."""
#         return self._mpl_call("stem", *args, **kwargs)
# 
#     # =========================================================================
#     # Scatter plots
#     # =========================================================================
#     def mpl_scatter(self, *args, **kwargs):
#         """Matplotlib scatter() - tracked, identical to ax.scatter()."""
#         return self._mpl_call("scatter", *args, **kwargs)
# 
#     # =========================================================================
#     # Bar plots
#     # =========================================================================
#     def mpl_bar(self, *args, **kwargs):
#         """Matplotlib bar() - tracked, identical to ax.bar()."""
#         return self._mpl_call("bar", *args, **kwargs)
# 
#     def mpl_barh(self, *args, **kwargs):
#         """Matplotlib barh() - tracked, identical to ax.barh()."""
#         return self._mpl_call("barh", *args, **kwargs)
# 
#     def mpl_bar3d(self, *args, **kwargs):
#         """Matplotlib bar3d() (3D axes) - tracked."""
#         return self._mpl_call("bar3d", *args, **kwargs)
# 
#     # =========================================================================
#     # Histograms
#     # =========================================================================
#     def mpl_hist(self, *args, **kwargs):
#         """Matplotlib hist() - tracked, identical to ax.hist()."""
#         return self._mpl_call("hist", *args, **kwargs)
# 
#     def mpl_hist2d(self, *args, **kwargs):
#         """Matplotlib hist2d() - tracked, identical to ax.hist2d()."""
#         return self._mpl_call("hist2d", *args, **kwargs)
# 
#     def mpl_hexbin(self, *args, **kwargs):
#         """Matplotlib hexbin() - tracked, identical to ax.hexbin()."""
#         return self._mpl_call("hexbin", *args, **kwargs)
# 
#     # =========================================================================
#     # Statistical plots
#     # =========================================================================
#     def mpl_boxplot(self, *args, **kwargs):
#         """Matplotlib boxplot() - tracked, identical to ax.boxplot()."""
#         return self._mpl_call("boxplot", *args, **kwargs)
# 
#     def mpl_violinplot(self, *args, **kwargs):
#         """Matplotlib violinplot() - tracked, identical to ax.violinplot()."""
#         return self._mpl_call("violinplot", *args, **kwargs)
# 
#     def mpl_errorbar(self, *args, **kwargs):
#         """Matplotlib errorbar() - tracked, identical to ax.errorbar()."""
#         return self._mpl_call("errorbar", *args, **kwargs)
# 
#     def mpl_eventplot(self, *args, **kwargs):
#         """Matplotlib eventplot() - tracked, identical to ax.eventplot()."""
#         return self._mpl_call("eventplot", *args, **kwargs)
# 
#     # =========================================================================
#     # Fill and area plots
#     # =========================================================================
#     def mpl_fill(self, *args, **kwargs):
#         """Matplotlib fill() - tracked, identical to ax.fill()."""
#         return self._mpl_call("fill", *args, **kwargs)
# 
#     def mpl_fill_between(self, *args, **kwargs):
#         """Matplotlib fill_between() - tracked, identical to ax.fill_between()."""
#         return self._mpl_call("fill_between", *args, **kwargs)
# 
#     def mpl_fill_betweenx(self, *args, **kwargs):
#         """Matplotlib fill_betweenx() - tracked, identical to ax.fill_betweenx()."""
#         return self._mpl_call("fill_betweenx", *args, **kwargs)
# 
#     def mpl_stackplot(self, *args, **kwargs):
#         """Matplotlib stackplot() - tracked, identical to ax.stackplot()."""
#         return self._mpl_call("stackplot", *args, **kwargs)
# 
#     # =========================================================================
#     # Contour and heatmap plots
#     # =========================================================================
#     def mpl_contour(self, *args, **kwargs):
#         """Matplotlib contour() - tracked, identical to ax.contour()."""
#         return self._mpl_call("contour", *args, **kwargs)
# 
#     def mpl_contourf(self, *args, **kwargs):
#         """Matplotlib contourf() - tracked, identical to ax.contourf()."""
#         return self._mpl_call("contourf", *args, **kwargs)
# 
#     def mpl_imshow(self, *args, **kwargs):
#         """Matplotlib imshow() - tracked, identical to ax.imshow()."""
#         return self._mpl_call("imshow", *args, **kwargs)
# 
#     def mpl_pcolormesh(self, *args, **kwargs):
#         """Matplotlib pcolormesh() - tracked, identical to ax.pcolormesh()."""
#         return self._mpl_call("pcolormesh", *args, **kwargs)
# 
#     def mpl_pcolor(self, *args, **kwargs):
#         """Matplotlib pcolor() - tracked, identical to ax.pcolor()."""
#         return self._mpl_call("pcolor", *args, **kwargs)
# 
#     def mpl_matshow(self, *args, **kwargs):
#         """Matplotlib matshow() - tracked, identical to ax.matshow()."""
#         return self._mpl_call("matshow", *args, **kwargs)
# 
#     # =========================================================================
#     # Vector field plots
#     # =========================================================================
#     def mpl_quiver(self, *args, **kwargs):
#         """Matplotlib quiver() - tracked, identical to ax.quiver()."""
#         return self._mpl_call("quiver", *args, **kwargs)
# 
#     def mpl_streamplot(self, *args, **kwargs):
#         """Matplotlib streamplot() - tracked, identical to ax.streamplot()."""
#         return self._mpl_call("streamplot", *args, **kwargs)
# 
#     def mpl_barbs(self, *args, **kwargs):
#         """Matplotlib barbs() - tracked, identical to ax.barbs()."""
#         return self._mpl_call("barbs", *args, **kwargs)
# 
#     # =========================================================================
#     # Pie and polar plots
#     # =========================================================================
#     def mpl_pie(self, *args, **kwargs):
#         """Matplotlib pie() - tracked, identical to ax.pie()."""
#         return self._mpl_call("pie", *args, **kwargs)
# 
#     # =========================================================================
#     # Text and annotations
#     # =========================================================================
#     def mpl_text(self, *args, **kwargs):
#         """Matplotlib text() - tracked, identical to ax.text()."""
#         return self._mpl_call("text", *args, **kwargs)
# 
#     def mpl_annotate(self, *args, **kwargs):
#         """Matplotlib annotate() - tracked, identical to ax.annotate()."""
#         return self._mpl_call("annotate", *args, **kwargs)
# 
#     # =========================================================================
#     # Lines and spans
#     # =========================================================================
#     def mpl_axhline(self, *args, **kwargs):
#         """Matplotlib axhline() - tracked, identical to ax.axhline()."""
#         return self._mpl_call("axhline", *args, **kwargs)
# 
#     def mpl_axvline(self, *args, **kwargs):
#         """Matplotlib axvline() - tracked, identical to ax.axvline()."""
#         return self._mpl_call("axvline", *args, **kwargs)
# 
#     def mpl_axhspan(self, *args, **kwargs):
#         """Matplotlib axhspan() - tracked, identical to ax.axhspan()."""
#         return self._mpl_call("axhspan", *args, **kwargs)
# 
#     def mpl_axvspan(self, *args, **kwargs):
#         """Matplotlib axvspan() - tracked, identical to ax.axvspan()."""
#         return self._mpl_call("axvspan", *args, **kwargs)
# 
#     # =========================================================================
#     # Patches and shapes
#     # =========================================================================
#     def mpl_add_patch(self, patch, **kwargs):
#         """Matplotlib add_patch() - tracked, identical to ax.add_patch()."""
#         return self._mpl_call("add_patch", patch, **kwargs)
# 
#     def mpl_add_artist(self, artist, **kwargs):
#         """Matplotlib add_artist() - tracked, identical to ax.add_artist()."""
#         return self._mpl_call("add_artist", artist, **kwargs)
# 
#     def mpl_add_collection(self, collection, **kwargs):
#         """Matplotlib add_collection() - tracked, identical to ax.add_collection()."""
#         return self._mpl_call("add_collection", collection, **kwargs)
# 
#     # =========================================================================
#     # 3D plotting (if available)
#     # =========================================================================
#     def mpl_plot_surface(self, *args, **kwargs):
#         """Matplotlib plot_surface() (3D axes) - tracked."""
#         return self._mpl_call("plot_surface", *args, **kwargs)
# 
#     def mpl_plot_wireframe(self, *args, **kwargs):
#         """Matplotlib plot_wireframe() (3D axes) - tracked."""
#         return self._mpl_call("plot_wireframe", *args, **kwargs)
# 
#     def mpl_contour3D(self, *args, **kwargs):
#         """Matplotlib contour3D() (3D axes) - tracked."""
#         return self._mpl_call("contour3D", *args, **kwargs)
# 
#     def mpl_scatter3D(self, *args, **kwargs):
#         """Matplotlib scatter3D() (3D axes) - tracked."""
#         return self._mpl_call("scatter3D", *args, **kwargs)
# 
#     # =========================================================================
#     # Utility method to get raw axes
#     # =========================================================================
#     @property
#     def mpl_axes(self):
#         """Direct access to underlying matplotlib axes object."""
#         return self._axes_mpl
# 
#     def mpl_raw(self, method_name, *args, **kwargs):
#         """Call any matplotlib method by name without scitex processing.
# 
#         Parameters
#         ----------
#         method_name : str
#             Name of matplotlib axes method to call
#         *args, **kwargs
#             Arguments to pass to the method
# 
#         Returns
#         -------
#         result
#             Result from matplotlib method
# 
#         Example
#         -------
#         >>> ax.mpl_raw("tricontour", x, y, z, levels=10)
#         """
#         method = getattr(self._axes_mpl, method_name)
#         return method(*args, **kwargs)
# 
# 
# # Registry of mpl_xxx methods for programmatic access
# MPL_METHODS = [
#     # Line plots
#     "mpl_plot", "mpl_step", "mpl_stem",
#     # Scatter
#     "mpl_scatter",
#     # Bar
#     "mpl_bar", "mpl_barh", "mpl_bar3d",
#     # Histograms
#     "mpl_hist", "mpl_hist2d", "mpl_hexbin",
#     # Statistical
#     "mpl_boxplot", "mpl_violinplot", "mpl_errorbar", "mpl_eventplot",
#     # Fill/area
#     "mpl_fill", "mpl_fill_between", "mpl_fill_betweenx", "mpl_stackplot",
#     # Contour/heatmap
#     "mpl_contour", "mpl_contourf", "mpl_imshow", "mpl_pcolormesh", "mpl_pcolor", "mpl_matshow",
#     # Vector fields
#     "mpl_quiver", "mpl_streamplot", "mpl_barbs",
#     # Pie
#     "mpl_pie",
#     # Text/annotations
#     "mpl_text", "mpl_annotate",
#     # Lines/spans
#     "mpl_axhline", "mpl_axvline", "mpl_axhspan", "mpl_axvspan",
#     # Patches
#     "mpl_add_patch", "mpl_add_artist", "mpl_add_collection",
#     # 3D
#     "mpl_plot_surface", "mpl_plot_wireframe", "mpl_contour3D", "mpl_scatter3D",
# ]
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_RawMatplotlibMixin.py
# --------------------------------------------------------------------------------
