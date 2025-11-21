#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-13 22:42:39 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/SciTeX-Code/src/scitex/plt/__init__.py
# ----------------------------------------
import os
__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
"""Scitex plt module."""

# Register Arial fonts eagerly (before lazy imports)
import matplotlib.font_manager as fm
import matplotlib as mpl

_arial_enabled = False
try:
    fm.findfont("Arial", fallback_to_default=False)
    _arial_enabled = True
except Exception:
    # Search for Arial font files and register them
    arial_paths = [
        f for f in fm.findSystemFonts()
        if os.path.basename(f).lower().startswith("arial")
    ]

    if arial_paths:
        for path in arial_paths:
            try:
                fm.fontManager.addfont(path)
            except Exception:
                pass

        # Verify Arial is now available
        try:
            fm.findfont("Arial", fallback_to_default=False)
            _arial_enabled = True
        except Exception:
            pass

# Configure matplotlib to use Arial if available
if _arial_enabled:
    mpl.rcParams["font.family"] = "Arial"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
else:
    # Warn about missing Arial
    try:
        from scitex.logging import getLogger
        _logger = getLogger(__name__)
        _logger.warning(
            "Arial font not found. Using fallback fonts (Helvetica/DejaVu Sans). "
            "For publication figures with Arial: sudo apt-get install ttf-mscorefonts-installer && fc-cache -fv"
        )
    except:
        pass  # Skip warning if logging not available

from ._tpl import termplot
from . import color
from . import utils
from . import ax
from . import presets

# Lazy import for subplots to avoid circular dependencies
_subplots = None
_figure = None
_crop = None

def subplots(*args, **kwargs):
    """Lazy-loaded subplots function."""
    global _subplots
    if _subplots is None:
        from ._subplots._SubplotsWrapper import subplots as _subplots_func
        _subplots = _subplots_func
    return _subplots(*args, **kwargs)

def figure(*args, **kwargs):
    """Lazy-loaded figure function that returns a FigWrapper."""
    global _figure
    if _figure is None:
        import matplotlib.pyplot as plt
        from ._subplots._FigWrapper import FigWrapper
        def _figure_func(*args, **kwargs):
            fig_mpl = plt.figure(*args, **kwargs)
            return FigWrapper(fig_mpl)
        _figure = _figure_func
    return _figure(*args, **kwargs)

def crop(input_path, output_path=None, margin=12, overwrite=False, verbose=False):
    """
    Auto-crop a figure to its content area.

    This function automatically detects the content area of a saved figure
    and crops it, removing excess whitespace. Designed for publication figures
    created with large margins.

    Parameters
    ----------
    input_path : str
        Path to the input image
    output_path : str, optional
        Path to save cropped image. If None and overwrite=True, overwrites input.
        If None and overwrite=False, adds '_cropped' suffix.
    margin : int, optional
        Margin in pixels around content (default: 12, ~1mm at 300 DPI)
    overwrite : bool, optional
        Overwrite input file (default: False)
    verbose : bool, optional
        Print detailed information (default: False)

    Returns
    -------
    str
        Path to the saved cropped image

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> stx.io.save(fig, "figure.png")
    >>> stx.plt.crop("figure.png", "figure_cropped.png")  # 1mm margin
    """
    global _crop
    if _crop is None:
        from .utils._crop import crop as _crop_func
        _crop = _crop_func
    return _crop(input_path, output_path, margin, overwrite, verbose)

def tight_layout(**kwargs):
    """
    Wrapper for matplotlib.pyplot.tight_layout that handles colorbar layout compatibility.

    This function calls tight_layout on the current figure and gracefully handles:
    1. UserWarning: "The figure layout has changed to tight" - informational only
    2. RuntimeError: Colorbar layout incompatibility - occurs when colorbars exist with old engine

    When a colorbar layout error occurs, the function silently continues as the layout
    is still functional even if the engine cannot be changed.

    Parameters
    ----------
    **kwargs
        All keyword arguments are passed to matplotlib.pyplot.tight_layout()
    """
    import warnings
    import matplotlib.pyplot as plt

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
        try:
            plt.tight_layout(**kwargs)
        except RuntimeError as e:
            # Silently handle colorbar layout engine incompatibility
            # This occurs when colorbars were created before tight_layout is called
            # The layout is still usable, so we can safely ignore this error
            if "Colorbar layout" not in str(e):
                raise


def colorbar(mappable=None, cax=None, ax=None, **kwargs):
    """
    Create a colorbar, automatically unwrapping SciTeX AxisWrapper objects.

    This function handles both regular matplotlib axes and SciTeX AxisWrapper
    objects transparently, making it a drop-in replacement for plt.colorbar().

    Parameters
    ----------
    mappable : ScalarMappable, optional
        The image, contour set, etc. to which the colorbar applies.
        If None, uses the current image.
    cax : Axes, optional
        Axes into which the colorbar will be drawn.
    ax : Axes or AxisWrapper or list thereof, optional
        Parent axes from which space for the colorbar will be stolen.
        If None, uses current axes.
    **kwargs
        Additional keyword arguments passed to matplotlib.pyplot.colorbar()

    Returns
    -------
    Colorbar
        The created colorbar object
    """
    import matplotlib.pyplot as plt

    # Unwrap ax if it's a SciTeX AxisWrapper
    if ax is not None:
        if hasattr(ax, '__iter__') and not isinstance(ax, str):
            # Handle list/array of axes
            ax = [a._axis_mpl if hasattr(a, '_axis_mpl') else a for a in ax]
        else:
            # Single axis
            ax = ax._axis_mpl if hasattr(ax, '_axis_mpl') else ax

    # Unwrap cax if provided
    if cax is not None:
        cax = cax._axis_mpl if hasattr(cax, '_axis_mpl') else cax

    # Call matplotlib's colorbar with unwrapped axes
    return plt.colorbar(mappable=mappable, cax=cax, ax=ax, **kwargs)

__all__ = [
    "termplot",
    "subplots",
    "figure",
    "tight_layout",
    "utils",
    "color",
    "presets",
]

def __getattr__(name):
    """
    Fallback to matplotlib.pyplot for any missing attributes.
    This makes scitex.plt a complete drop-in replacement for matplotlib.pyplot.
    """
    try:
        import matplotlib.pyplot as plt
        if hasattr(plt, name):
            return getattr(plt, name)
        else:
            raise AttributeError(f"module 'scitex.plt' has no attribute '{name}'")
    except ImportError:
        raise AttributeError(f"module 'scitex.plt' has no attribute '{name}' (matplotlib not available)")

def __dir__():
    """
    Provide comprehensive directory listing including matplotlib.pyplot functions.
    """
    # Get local attributes
    local_attrs = __all__.copy()

    # Add matplotlib.pyplot attributes
    try:
        import matplotlib.pyplot as plt
        mpl_attrs = [attr for attr in dir(plt) if not attr.startswith('_')]
        local_attrs.extend(mpl_attrs)
    except ImportError:
        pass

    return sorted(set(local_attrs))

# EOF
