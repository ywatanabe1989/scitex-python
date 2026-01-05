#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-19 02:53:28 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_FigWrapper.py.new
# ----------------------------------------
import os

__FILE__ = "./src/scitex/plt/_subplots/_FigWrapper.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from functools import wraps
import warnings
import numpy as np
import pandas as pd

from scitex import logging

logger = logging.getLogger(__name__)


class FigWrapper:
    def __init__(self, fig_mpl):
        self._fig_mpl = fig_mpl
        self._axes = []  # Keep track of axes for synchronization
        self._last_saved_info = None
        self._not_saved_yet_flag = True
        self._called_from_mng_io_save = False

    @property
    def figure(
        self,
    ):
        return self._fig_mpl

    def __getattr__(self, attr):
        # print(f"Attribute of FigWrapper: {attr}")
        attr_mpl = getattr(self._fig_mpl, attr)

        if callable(attr_mpl):

            @wraps(attr_mpl)
            def wrapper(*args, track=None, id=None, **kwargs):
                # Suppress constrained_layout warnings for certain operations
                import warnings

                with warnings.catch_warnings():
                    if attr in ["subplots_adjust", "tight_layout"]:
                        warnings.filterwarnings(
                            "ignore",
                            message=".*constrained_layout.*",
                            category=UserWarning,
                        )
                        warnings.filterwarnings(
                            "ignore",
                            message=".*layout engine.*incompatible.*",
                            category=UserWarning,
                        )
                    results = attr_mpl(*args, **kwargs)
                # self._track(track, id, attr, args, kwargs)
                return results

            return wrapper

        else:
            return attr_mpl

    def __dir__(self):
        # Combine attributes from both self and the wrapped matplotlib figure
        attrs = set(dir(self.__class__))
        attrs.update(object.__dir__(self))
        attrs.update(dir(self._fig_mpl))
        return sorted(attrs)

    def savefig(self, fname, *args, embed_metadata=True, metadata=None, **kwargs):
        """
        Save figure with automatic metadata embedding.

        Parameters
        ----------
        fname : str
            Output file path
        embed_metadata : bool, optional
            Automatically embed dimension/style metadata in PNG/JPEG/TIFF/PDF (default: True)
        metadata : dict, optional
            Additional custom metadata to merge with auto-collected metadata
        *args, **kwargs
            Passed to scitex.io.save_image or matplotlib savefig

        Notes
        -----
        For PNG/JPEG/TIFF/PDF formats, metadata is automatically embedded including:
        - Software versions (scitex, matplotlib)
        - Timestamp
        - Figure/axes dimensions (mm, inch, px)
        - DPI settings
        - Styling parameters (if available via _scitex_metadata)
        - Mode (display/publication)

        For other formats (SVG, etc.), delegates to matplotlib's savefig.

        Examples
        --------
        >>> fig, ax = splt.subplots(fig_mm={'width': 35, 'height': 24.5})
        >>> ax.plot(x, y)
        >>> fig.savefig('result.png', dpi=300)  # Metadata embedded automatically!

        >>> # Add custom metadata
        >>> fig.savefig('result.png', dpi=300, metadata={'experiment': 'test_001'})

        >>> # Disable metadata embedding
        >>> fig.savefig('result.png', embed_metadata=False)
        """
        # Check if this is a format that can have metadata (PNG/JPEG/TIFF/PDF)
        # Handle both string paths and file-like objects (e.g., BytesIO)
        if isinstance(fname, str):
            is_image_format = fname.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf")
            )
        else:
            # For file-like objects, check the 'format' kwarg if provided
            # Otherwise default to False (no metadata embedding for BytesIO etc.)
            fmt = kwargs.get('format', '').lower() if kwargs.get('format') else ''
            is_image_format = fmt in ('png', 'jpg', 'jpeg', 'tiff', 'tif', 'pdf')

        if is_image_format and embed_metadata:
            # Collect automatic metadata
            auto_metadata = None

            # Get first axes if available
            # Keep the scitex AxisWrapper (for history tracking) separate from matplotlib axes
            ax = None
            ax_scitex = None  # scitex AxisWrapper with history
            if hasattr(self, "axes"):
                try:
                    # Try to get first axes from various wrapper types
                    if hasattr(self.axes, "_ax"):  # AxisWrapper
                        ax = self.axes._ax
                        ax_scitex = self.axes  # Keep the wrapper for history
                    elif hasattr(self.axes, "_axis_mpl"):  # Alternative
                        ax = self.axes._axis_mpl
                        ax_scitex = self.axes
                    elif hasattr(self.axes, "flatten"):  # AxesWrapper
                        flat = list(self.axes.flatten())
                        if flat and hasattr(flat[0], "_ax"):
                            ax = flat[0]._ax
                            ax_scitex = flat[0]  # Keep the wrapper for history
                        elif flat and hasattr(flat[0], "_axis_mpl"):
                            ax = flat[0]._axis_mpl
                            ax_scitex = flat[0]
                except Exception:
                    pass

            # If still no axes, try from figure
            if (
                ax is None
                and hasattr(self._fig_mpl, "axes")
                and len(self._fig_mpl.axes) > 0
            ):
                ax = self._fig_mpl.axes[0]

            # Collect metadata
            # Pass ax_scitex if available (has history for plot type detection)
            try:
                from scitex.plt.utils import collect_figure_metadata

                auto_metadata = collect_figure_metadata(self._fig_mpl, ax_scitex if ax_scitex else ax)

                # Merge with custom metadata
                if metadata:
                    if "custom" not in auto_metadata:
                        auto_metadata["custom"] = {}
                    auto_metadata["custom"].update(metadata)
            except Exception as e:
                # If metadata collection fails, warn but continue
                logger.warning(f"Could not collect metadata: {e}")
                auto_metadata = metadata

            # Use scitex.io.save_image for metadata embedding
            try:
                from scitex.io._save_modules import save_image

                save_image(
                    self._fig_mpl, fname, metadata=auto_metadata, *args, **kwargs
                )
            except Exception as e:
                # Fallback to regular matplotlib savefig
                logger.warning(f"Metadata embedding failed, using regular savefig: {e}")
                self._fig_mpl.savefig(fname, *args, **kwargs)
        else:
            # For non-image formats or when metadata disabled, use regular savefig
            self._fig_mpl.savefig(fname, *args, **kwargs)

    def export_as_csv(self):
        """Export plotted data from all axes.

        This method collects data from all axes in the figure and combines
        them into a single DataFrame with appropriate axis identifiers in
        the column names.

        Returns:
            pd.DataFrame: Combined DataFrame with data from all axes,
                          with axis ID prefixes for each column.
        """
        dfs = []

        # Use the _traverse_axes helper method to iterate through all axes
        # regardless of their structure (single, array, list, etc.)
        for ii, ax in enumerate(self._traverse_axes()):
            # Try different ways to access the export_as_csv method
            df = None
            try:
                if hasattr(ax, "_axis_mpl") and hasattr(ax._axis_mpl, "export_as_csv"):
                    # If it's a nested structure with _axis_mpl having export_as_csv
                    df = ax._axis_mpl.export_as_csv()
                elif hasattr(ax, "export_as_csv"):
                    # Direct AxisWrapper object
                    df = ax.export_as_csv()
                else:
                    # Skip if no export method available
                    continue
            except Exception:
                continue

            # Process the DataFrame if it's not empty
            if df is not None and not df.empty:
                # Column names already include axis position via get_csv_column_name
                # (single source of truth from _csv_column_naming.py)
                # Only handle duplicates by adding a counter
                new_cols = []
                col_counts = {}
                for col in df.columns:
                    col_str = str(col)

                    # Handle duplicates by adding a counter
                    if col_str in col_counts:
                        col_counts[col_str] += 1
                        col_str = f"{col_str}_{col_counts[col_str]}"
                    else:
                        col_counts[col_str] = 0

                    new_cols.append(col_str)

                df.columns = new_cols
                dfs.append(df)

        # Return concatenated DataFrame or empty DataFrame if no data
        return pd.concat(dfs, axis=1) if dfs else pd.DataFrame()

    def colorbar(self, mappable, ax=None, **kwargs):
        """Add a colorbar to the figure, automatically unwrapping SciTeX axes.

        This method properly handles both regular matplotlib axes and SciTeX
        AxisWrapper objects when creating colorbars.

        Parameters:
        -----------
        mappable : ScalarMappable
            The image, contour set, etc. to which the colorbar applies
        ax : Axes or AxisWrapper, optional
            The axes to attach the colorbar to. If not specified, uses current axes.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib's colorbar

        Returns:
        --------
        Colorbar
            The created colorbar object
        """
        # Unwrap axes if it's a SciTeX AxisWrapper
        if ax is not None:
            ax_mpl = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax
        else:
            ax_mpl = None

        # Call matplotlib's colorbar with the unwrapped axes
        return self._fig_mpl.colorbar(mappable, ax=ax_mpl, **kwargs)

    def _traverse_axes(self):
        """Helper method to traverse all axis wrappers in the figure."""
        if hasattr(self, "axes"):
            # Check if we're dealing with an AxesWrapper instance
            if hasattr(self.axes, "_axes_scitex") and hasattr(
                self.axes._axes_scitex, "flat"
            ):
                # This is an AxesWrapper, get the individual AxisWrapper objects
                for ax in self.axes._axes_scitex.flat:
                    yield ax
            elif not hasattr(self.axes, "__iter__"):
                # Single axis case
                yield self.axes
            else:
                # Multiple axes case
                if hasattr(self.axes, "flat"):
                    # 2D array of axes
                    for ax in self.axes.flat:
                        yield ax
                elif hasattr(self.axes, "ravel"):
                    # Numpy array
                    for ax in self.axes.ravel():
                        yield ax
                elif isinstance(self.axes, (list, tuple)):
                    # List of axes
                    for ax in self.axes:
                        yield ax

    @property
    def history(self):
        """Aggregate tracking history from all axes in the figure.

        Returns a combined OrderedDict of all tracking records from all axes,
        enabling FTS bundle creation to build encoding from plot operations.
        """
        from collections import OrderedDict

        combined = OrderedDict()
        for ax in self._traverse_axes():
            if hasattr(ax, "history") and ax.history:
                combined.update(ax.history)
        return combined

    def legend(self, *args, loc="best", **kwargs):
        """Legend with 'best' automatic placement by default for all axes."""
        for ax in self._traverse_axes():
            try:
                ax.legend(*args, loc=loc, **kwargs)
            except Exception as e:
                pass

    def supxyt(self, x=False, y=False, t=False):
        """Wrapper for supxlabel, supylabel, and suptitle"""
        if x is not False:
            self._fig_mpl.supxlabel(x)
        if y is not False:
            self._fig_mpl.supylabel(y)
        if t is not False:
            self._fig_mpl.suptitle(t)
        return self._fig_mpl

    def tight_layout(self, *, rect=[0, 0.03, 1, 0.95], **kwargs):
        """Wrapper for tight_layout with rect=[0, 0.03, 1, 0.95] by default.

        Handles cases where certain axes (like colorbars) are incompatible
        with tight_layout. If the figure is using constrained_layout, this
        method does nothing as constrained_layout handles spacing automatically.
        """
        import warnings

        # Check if figure is already using constrained_layout
        if (
            hasattr(self._fig_mpl, "get_constrained_layout")
            and self._fig_mpl.get_constrained_layout()
        ):
            # Figure is using constrained_layout, which handles colorbars better
            # No need to call tight_layout
            return

        try:
            with warnings.catch_warnings():
                # Suppress the specific warning about incompatible axes
                warnings.filterwarnings(
                    "ignore",
                    message="This figure includes Axes that are not compatible with tight_layout",
                )
                self._fig_mpl.tight_layout(rect=rect, **kwargs)
        except Exception:
            # If tight_layout fails completely, try constrained_layout as fallback
            try:
                self._fig_mpl.set_constrained_layout(True)
                self._fig_mpl.set_constrained_layout_pads(w_pad=0.04, h_pad=0.04)
            except Exception:
                # If both fail, do nothing - figure will use default layout
                pass

    def adjust_layout(self, **kwargs):
        """Adjust the constrained layout parameters.

        Parameters
        ----------
        w_pad : float, optional
            Width padding around axes (default: 0.05)
        h_pad : float, optional
            Height padding around axes (default: 0.05)
        wspace : float, optional
            Width space between subplots (default: 0.02)
        hspace : float, optional
            Height space between subplots (default: 0.02)
        rect : list of 4 floats, optional
            Rectangle in normalized figure coordinates to fit the whole layout
            [left, bottom, right, top] (default: [0, 0, 1, 1])
        """
        if (
            hasattr(self._fig_mpl, "get_constrained_layout")
            and self._fig_mpl.get_constrained_layout()
        ):
            # Update constrained layout parameters
            self._fig_mpl.set_constrained_layout_pads(**kwargs)
        else:
            # Fall back to tight_layout with rect parameter if provided
            if "rect" in kwargs:
                self.tight_layout(rect=kwargs["rect"])

    def close(self):
        """Close the underlying matplotlib figure"""
        import matplotlib.pyplot as plt

        plt.close(self._fig_mpl)

    @property
    def number(self):
        """Return the figure number for matplotlib.pyplot.close() compatibility"""
        return self._fig_mpl.number

    def __del__(self):
        """Cleanup when FigWrapper is deleted"""
        try:
            import matplotlib.pyplot as plt

            plt.close(self._fig_mpl)
        except:
            pass


# EOF
