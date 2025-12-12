#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/_RawMatplotlibMixin.py

"""
Raw matplotlib aliases (mpl_xxx) for direct access without scitex processing.

Provides consistent naming convention:
- stx_xxx: scitex-specific methods with tracking, unit awareness
- sns_xxx: seaborn wrappers
- mpl_xxx: raw matplotlib methods (no tracking, no styling, no defaults)

Usage:
    # With tracking and scitex styling
    ax.plot(x, y)  # or ax.stx_line(x, y)

    # Raw matplotlib, no scitex processing
    ax.mpl_plot(x, y)
"""

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)


class RawMatplotlibMixin:
    """Mixin providing mpl_xxx aliases for raw matplotlib access.

    These methods bypass all scitex processing:
    - No tracking
    - No default styling
    - No unit conversion
    - No post-processing

    Useful when you need exact matplotlib behavior or performance-critical code.
    """

    # =========================================================================
    # Line plots
    # =========================================================================
    def mpl_plot(self, *args, **kwargs):
        """Raw matplotlib plot() - no scitex processing."""
        return self._axes_mpl.plot(*args, **kwargs)

    def mpl_step(self, *args, **kwargs):
        """Raw matplotlib step() - no scitex processing."""
        return self._axes_mpl.step(*args, **kwargs)

    def mpl_stem(self, *args, **kwargs):
        """Raw matplotlib stem() - no scitex processing."""
        return self._axes_mpl.stem(*args, **kwargs)

    # =========================================================================
    # Scatter plots
    # =========================================================================
    def mpl_scatter(self, *args, **kwargs):
        """Raw matplotlib scatter() - no scitex processing."""
        return self._axes_mpl.scatter(*args, **kwargs)

    # =========================================================================
    # Bar plots
    # =========================================================================
    def mpl_bar(self, *args, **kwargs):
        """Raw matplotlib bar() - no scitex processing."""
        return self._axes_mpl.bar(*args, **kwargs)

    def mpl_barh(self, *args, **kwargs):
        """Raw matplotlib barh() - no scitex processing."""
        return self._axes_mpl.barh(*args, **kwargs)

    def mpl_bar3d(self, *args, **kwargs):
        """Raw matplotlib bar3d() (3D axes) - no scitex processing."""
        if hasattr(self._axes_mpl, 'bar3d'):
            return self._axes_mpl.bar3d(*args, **kwargs)
        raise AttributeError("bar3d is only available on 3D axes")

    # =========================================================================
    # Histograms
    # =========================================================================
    def mpl_hist(self, *args, **kwargs):
        """Raw matplotlib hist() - no scitex processing."""
        return self._axes_mpl.hist(*args, **kwargs)

    def mpl_hist2d(self, *args, **kwargs):
        """Raw matplotlib hist2d() - no scitex processing."""
        return self._axes_mpl.hist2d(*args, **kwargs)

    def mpl_hexbin(self, *args, **kwargs):
        """Raw matplotlib hexbin() - no scitex processing."""
        return self._axes_mpl.hexbin(*args, **kwargs)

    # =========================================================================
    # Statistical plots
    # =========================================================================
    def mpl_boxplot(self, *args, **kwargs):
        """Raw matplotlib boxplot() - no scitex processing."""
        return self._axes_mpl.boxplot(*args, **kwargs)

    def mpl_violinplot(self, *args, **kwargs):
        """Raw matplotlib violinplot() - no scitex processing."""
        return self._axes_mpl.violinplot(*args, **kwargs)

    def mpl_errorbar(self, *args, **kwargs):
        """Raw matplotlib errorbar() - no scitex processing."""
        return self._axes_mpl.errorbar(*args, **kwargs)

    def mpl_eventplot(self, *args, **kwargs):
        """Raw matplotlib eventplot() - no scitex processing."""
        return self._axes_mpl.eventplot(*args, **kwargs)

    # =========================================================================
    # Fill and area plots
    # =========================================================================
    def mpl_fill(self, *args, **kwargs):
        """Raw matplotlib fill() - no scitex processing."""
        return self._axes_mpl.fill(*args, **kwargs)

    def mpl_fill_between(self, *args, **kwargs):
        """Raw matplotlib fill_between() - no scitex processing."""
        return self._axes_mpl.fill_between(*args, **kwargs)

    def mpl_fill_betweenx(self, *args, **kwargs):
        """Raw matplotlib fill_betweenx() - no scitex processing."""
        return self._axes_mpl.fill_betweenx(*args, **kwargs)

    def mpl_stackplot(self, *args, **kwargs):
        """Raw matplotlib stackplot() - no scitex processing."""
        return self._axes_mpl.stackplot(*args, **kwargs)

    # =========================================================================
    # Contour and heatmap plots
    # =========================================================================
    def mpl_contour(self, *args, **kwargs):
        """Raw matplotlib contour() - no scitex processing."""
        return self._axes_mpl.contour(*args, **kwargs)

    def mpl_contourf(self, *args, **kwargs):
        """Raw matplotlib contourf() - no scitex processing."""
        return self._axes_mpl.contourf(*args, **kwargs)

    def mpl_imshow(self, *args, **kwargs):
        """Raw matplotlib imshow() - no scitex processing."""
        return self._axes_mpl.imshow(*args, **kwargs)

    def mpl_pcolormesh(self, *args, **kwargs):
        """Raw matplotlib pcolormesh() - no scitex processing."""
        return self._axes_mpl.pcolormesh(*args, **kwargs)

    def mpl_pcolor(self, *args, **kwargs):
        """Raw matplotlib pcolor() - no scitex processing."""
        return self._axes_mpl.pcolor(*args, **kwargs)

    def mpl_matshow(self, *args, **kwargs):
        """Raw matplotlib matshow() - no scitex processing."""
        if hasattr(self._axes_mpl, 'matshow'):
            return self._axes_mpl.matshow(*args, **kwargs)
        return self._fig_mpl.add_subplot().matshow(*args, **kwargs)

    # =========================================================================
    # Vector field plots
    # =========================================================================
    def mpl_quiver(self, *args, **kwargs):
        """Raw matplotlib quiver() - no scitex processing."""
        return self._axes_mpl.quiver(*args, **kwargs)

    def mpl_streamplot(self, *args, **kwargs):
        """Raw matplotlib streamplot() - no scitex processing."""
        return self._axes_mpl.streamplot(*args, **kwargs)

    def mpl_barbs(self, *args, **kwargs):
        """Raw matplotlib barbs() - no scitex processing."""
        return self._axes_mpl.barbs(*args, **kwargs)

    # =========================================================================
    # Pie and polar plots
    # =========================================================================
    def mpl_pie(self, *args, **kwargs):
        """Raw matplotlib pie() - no scitex processing."""
        return self._axes_mpl.pie(*args, **kwargs)

    # =========================================================================
    # Text and annotations
    # =========================================================================
    def mpl_text(self, *args, **kwargs):
        """Raw matplotlib text() - no scitex processing."""
        return self._axes_mpl.text(*args, **kwargs)

    def mpl_annotate(self, *args, **kwargs):
        """Raw matplotlib annotate() - no scitex processing."""
        return self._axes_mpl.annotate(*args, **kwargs)

    # =========================================================================
    # Lines and spans
    # =========================================================================
    def mpl_axhline(self, *args, **kwargs):
        """Raw matplotlib axhline() - no scitex processing."""
        return self._axes_mpl.axhline(*args, **kwargs)

    def mpl_axvline(self, *args, **kwargs):
        """Raw matplotlib axvline() - no scitex processing."""
        return self._axes_mpl.axvline(*args, **kwargs)

    def mpl_axhspan(self, *args, **kwargs):
        """Raw matplotlib axhspan() - no scitex processing."""
        return self._axes_mpl.axhspan(*args, **kwargs)

    def mpl_axvspan(self, *args, **kwargs):
        """Raw matplotlib axvspan() - no scitex processing."""
        return self._axes_mpl.axvspan(*args, **kwargs)

    # =========================================================================
    # Patches and shapes
    # =========================================================================
    def mpl_add_patch(self, patch, **kwargs):
        """Raw matplotlib add_patch() - no scitex processing."""
        return self._axes_mpl.add_patch(patch)

    def mpl_add_artist(self, artist, **kwargs):
        """Raw matplotlib add_artist() - no scitex processing."""
        return self._axes_mpl.add_artist(artist)

    def mpl_add_collection(self, collection, **kwargs):
        """Raw matplotlib add_collection() - no scitex processing."""
        return self._axes_mpl.add_collection(collection)

    # =========================================================================
    # 3D plotting (if available)
    # =========================================================================
    def mpl_plot_surface(self, *args, **kwargs):
        """Raw matplotlib plot_surface() (3D axes) - no scitex processing."""
        if hasattr(self._axes_mpl, 'plot_surface'):
            return self._axes_mpl.plot_surface(*args, **kwargs)
        raise AttributeError("plot_surface is only available on 3D axes")

    def mpl_plot_wireframe(self, *args, **kwargs):
        """Raw matplotlib plot_wireframe() (3D axes) - no scitex processing."""
        if hasattr(self._axes_mpl, 'plot_wireframe'):
            return self._axes_mpl.plot_wireframe(*args, **kwargs)
        raise AttributeError("plot_wireframe is only available on 3D axes")

    def mpl_contour3D(self, *args, **kwargs):
        """Raw matplotlib contour3D() (3D axes) - no scitex processing."""
        if hasattr(self._axes_mpl, 'contour3D'):
            return self._axes_mpl.contour3D(*args, **kwargs)
        raise AttributeError("contour3D is only available on 3D axes")

    def mpl_scatter3D(self, *args, **kwargs):
        """Raw matplotlib scatter3D() (3D axes) - no scitex processing."""
        if hasattr(self._axes_mpl, 'scatter3D'):
            return self._axes_mpl.scatter3D(*args, **kwargs)
        # Fallback to scatter for 3D axes
        if hasattr(self._axes_mpl, 'scatter'):
            return self._axes_mpl.scatter(*args, **kwargs)
        raise AttributeError("scatter3D is only available on 3D axes")

    # =========================================================================
    # Utility method to get raw axes
    # =========================================================================
    @property
    def mpl_axes(self):
        """Direct access to underlying matplotlib axes object."""
        return self._axes_mpl

    def mpl_raw(self, method_name, *args, **kwargs):
        """Call any matplotlib method by name without scitex processing.

        Parameters
        ----------
        method_name : str
            Name of matplotlib axes method to call
        *args, **kwargs
            Arguments to pass to the method

        Returns
        -------
        result
            Result from matplotlib method

        Example
        -------
        >>> ax.mpl_raw("tricontour", x, y, z, levels=10)
        """
        method = getattr(self._axes_mpl, method_name)
        return method(*args, **kwargs)


# Registry of mpl_xxx methods for programmatic access
MPL_METHODS = [
    # Line plots
    "mpl_plot", "mpl_step", "mpl_stem",
    # Scatter
    "mpl_scatter",
    # Bar
    "mpl_bar", "mpl_barh", "mpl_bar3d",
    # Histograms
    "mpl_hist", "mpl_hist2d", "mpl_hexbin",
    # Statistical
    "mpl_boxplot", "mpl_violinplot", "mpl_errorbar", "mpl_eventplot",
    # Fill/area
    "mpl_fill", "mpl_fill_between", "mpl_fill_betweenx", "mpl_stackplot",
    # Contour/heatmap
    "mpl_contour", "mpl_contourf", "mpl_imshow", "mpl_pcolormesh", "mpl_pcolor", "mpl_matshow",
    # Vector fields
    "mpl_quiver", "mpl_streamplot", "mpl_barbs",
    # Pie
    "mpl_pie",
    # Text/annotations
    "mpl_text", "mpl_annotate",
    # Lines/spans
    "mpl_axhline", "mpl_axvline", "mpl_axhspan", "mpl_axvspan",
    # Patches
    "mpl_add_patch", "mpl_add_artist", "mpl_add_collection",
    # 3D
    "mpl_plot_surface", "mpl_plot_wireframe", "mpl_contour3D", "mpl_scatter3D",
]


# EOF
