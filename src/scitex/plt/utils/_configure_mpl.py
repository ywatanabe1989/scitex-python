#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-19 12:01:10 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_configure_mpl.py


from typing import Any
from typing import Dict
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scitex
from scitex.dict import DotDict


def configure_mpl(
    plt,
    fig_size_mm=(160, 100),
    fig_scale=1.0,
    dpi_display=100,
    dpi_save=300,
    # fontsize="medium",
    autolayout=True,
    n_ticks=4,
    hide_top_right_spines=True,
    line_width=1.0,  # Increased from 0.5 for better visibility
    alpha=0.85,  # Adjusted for better contrast
    enable_latex=False,  # Disable LaTeX, use Arial font instead
    latex_preamble=None,  # Custom LaTeX preamble
    verbose=False,
    **kwargs,
) -> Tuple[Any, Dict]:
    """Configures Matplotlib settings for publication-quality plots.

    Parameters
    ----------
    plt : matplotlib.pyplot
        Matplotlib pyplot module
    fig_size_mm : tuple of int, optional
        Figure width and height in millimeters, by default (160, 100)
    fig_scale : float, optional
        Scaling factor for figure size, by default 1.0
    dpi_display : int, optional
        Display resolution in DPI, by default 100
    dpi_save : int, optional
        Saving resolution in DPI, by default 300
    # fontsize : Union[str, int, float], optional
    #     Base font size ('xx-small' to 'xx-large' or points), by default 'medium'
    #     Other sizes are derived from this:
    #     - Title: 125% of base
    #     - Labels: 100% of base
    #     - Ticks/Legend: 85% of base
    autolayout : bool, optional
        Whether to enable automatic tight layout, by default True
    hide_top_right_spines : bool, optional
        Whether to hide top and right spines, by default True
    line_width : float, optional
        Default line width, by default 1.0
    alpha : float, optional
        Color transparency, by default 0.85
    n_ticks : int, optional
        Number of ticks on each axis, by default 4
    verbose : bool, optional
        Whether to print configuration details, by default False

    Returns
    -------
    tuple
        (plt, DotDict of RGBA colors) - Access as COLORS.blue or COLORS['blue']
    """
    # # Convert base font size
    # base_size = _convert_font_size(fontsize)

    # # Ensure minimum sizes for different elements with better proportions
    # title_size = max(base_size * 1.25, 10.0)  # Increased for better hierarchy
    # label_size = max(base_size * 1.0, 9.0)  # Minimum 9pt for good readability
    # small_size = max(base_size * 0.85, 8.0)  # Increased ratio for better legibility

    # Colors
    RGBA = {
        k: scitex.plt.color.update_alpha(v, alpha)
        for k, v in scitex.plt.color.PARAMS["RGBA"].items()
    }

    RGBA_NORM = {
        k: tuple(scitex.plt.color.update_alpha(v, alpha))
        for k, v in scitex.plt.color.PARAMS["RGBA_NORM"].items()
    }

    RGBA_NORM_FOR_CYCLE = {
        k: tuple(scitex.plt.color.update_alpha(v, alpha))
        for k, v in scitex.plt.color.PARAMS["RGBA_NORM_FOR_CYCLE"].items()
    }

    # Normalize figure size from mm to inches
    figsize_inch = (
        fig_size_mm[0] / 25.4 * fig_scale,
        fig_size_mm[1] / 25.4 * fig_scale,
    )

    # Prepare matplotlib configuration
    mpl_config = {
        # Resolution
        "figure.dpi": dpi_display,
        "savefig.dpi": dpi_save,
        # Figure Size
        "figure.figsize": figsize_inch,
        # Font Sizes (7pt for titles/labels, 6pt for legend)
        "font.size": 7,  # Base font size
        "axes.titlesize": 7,  # Title size (prevent "large" default)
        "axes.labelsize": 7,  # Axis label size
        "xtick.labelsize": 7,  # Tick label size
        "ytick.labelsize": 7,  # Tick label size
        # Legend configuration
        "legend.fontsize": 6,  # 6pt for legend labels
        "legend.frameon": False,  # No frame by default
        "legend.loc": "best",  # Auto-position to avoid overlap
        # Auto Layout
        "figure.autolayout": autolayout,
        # Top and Right Axes
        "axes.spines.top": not hide_top_right_spines,
        "axes.spines.right": not hide_top_right_spines,
        # Spine width
        "axes.linewidth": 0.8,  # Slightly thicker axes lines
        # Custom color cycle
        "axes.prop_cycle": plt.cycler(
            color=list(RGBA_NORM_FOR_CYCLE.values())
        ),
        # Line
        "lines.linewidth": line_width,
        "lines.markersize": 6.0,  # Better default marker size
        # Grid (if used)
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
    }

    # Configure LaTeX rendering if enabled with enhanced fallback
    if enable_latex:
        latex_success = False
        try:
            # Try to enable LaTeX rendering
            test_config = {
                "text.usetex": True,
                "text.latex.preamble": latex_preamble
                or r"\usepackage{amsmath}\usepackage{amssymb}",
                "font.family": "serif",
                "font.serif": ["Computer Modern Roman"],
                "mathtext.fontset": "cm",
            }

            # Test LaTeX capability before applying
            with plt.rc_context(test_config):
                try:
                    # Create a test figure to verify LaTeX works
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, r"$x^2 + y^2 = r^2$", usetex=True)
                    fig.canvas.draw()  # Force rendering
                    plt.close(fig)
                    latex_success = True
                except Exception as latex_error:
                    plt.close(fig)
                    if verbose:
                        print(f"⚠️  LaTeX test render failed: {latex_error}")
                    raise latex_error

            if latex_success:
                mpl_config.update(test_config)
                if verbose:
                    print("✅ LaTeX rendering enabled and tested")

        except Exception as e:
            if verbose:
                print(
                    f"⚠️  LaTeX rendering failed, falling back to mathtext: {e}"
                )
                print(
                    "    This may be due to missing LaTeX fonts or Node.js conflicts"
                )

            # Enhanced fallback to mathtext with better configuration
            mpl_config.update(
                {
                    "text.usetex": False,
                    "mathtext.default": "regular",
                    "font.family": "serif",
                    "mathtext.fontset": "cm",
                    "mathtext.fallback": "cm",  # Fallback font for missing symbols
                }
            )

            # Enable LaTeX fallback mode in the str module
            try:
                from scitex.str._latex_fallback import set_fallback_mode

                set_fallback_mode("force_mathtext")
                if verbose:
                    print("📝 Enabled automatic LaTeX fallback mode")
            except ImportError:
                if verbose:
                    print("⚠️  LaTeX fallback module not available")

    else:
        # Use sans-serif fonts without LaTeX
        # Try to enable Arial explicitly
        import matplotlib.font_manager as fm
        import os

        # Try to detect and register Arial fonts
        arial_enabled = False
        try:
            # First check if Arial is already available
            fm.findfont("Arial", fallback_to_default=False)
            arial_enabled = True
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
                    arial_enabled = True
                except Exception:
                    pass

        # Configure font family
        if arial_enabled:
            mpl_config.update(
                {
                    "text.usetex": False,
                    "font.family": "Arial",
                    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
                    "mathtext.fontset": "dejavusans",
                    "mathtext.default": "regular",
                }
            )
        else:
            # Fall back to sans-serif with Helvetica/DejaVu Sans
            mpl_config.update(
                {
                    "text.usetex": False,
                    "font.family": "sans-serif",
                    "font.sans-serif": ["Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
                    "mathtext.fontset": "dejavusans",
                    "mathtext.default": "regular",
                }
            )

            # Warn user about missing Arial using scitex.logging
            try:
                from scitex.logging import getLogger
                logger = getLogger(__name__)
                logger.warning(
                    "Arial font not found. Using fallback fonts (Helvetica/DejaVu Sans). "
                    "For publication figures with Arial: sudo apt-get install ttf-mscorefonts-installer && fc-cache -fv"
                )
            except ImportError:
                # Fallback to warnings if scitex.logging not available
                import warnings
                warnings.warn(
                    "Arial font not found on system. Using fallback fonts (Helvetica/DejaVu Sans). "
                    "For publication-quality figures with Arial, install Microsoft Core Fonts: "
                    "sudo apt-get install ttf-mscorefonts-installer && fc-cache -fv",
                    UserWarning,
                    stacklevel=2
                )

        # Suppress matplotlib's own font warnings
        import logging
        logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

        # Set fallback mode to mathtext
        try:
            from scitex.str._latex_fallback import set_fallback_mode

            set_fallback_mode("force_mathtext")
        except ImportError:
            pass

    # Update Matplotlib configuration
    plt.rcParams.update(mpl_config)

    if verbose:
        print("\n" + "-" * 40)
        print("Matplotlib has been configured as follows:\n")
        print(f"Figure DPI (Display): {dpi_display} DPI")
        print(f"Figure DPI (Save): {dpi_save} DPI")
        print(
            f"Figure Size (Not the Axis Size): "
            f"{fig_size_mm[0] * fig_scale:.1f} x "
            f"{fig_size_mm[1] * fig_scale:.1f} mm (width x height)"
        )
        # print("\nFont Sizes:")
        # print(f"  Base Size: {base_size:.1f}pt")
        # print(f"  Title: {title_size:.1f}pt (125% of base, min 10pt)")
        # print(f"  Axis Labels: {label_size:.1f}pt (100% of base, min 9pt)")
        # print(f"  Tick Labels: {small_size:.1f}pt (85% of base, min 8pt)")
        # print(f"  Legend: {small_size:.1f}pt (85% of base, min 8pt)")
        print(f"\nHide Top and Right Axes: {hide_top_right_spines}")
        print(f"Line Width: {line_width}")
        # print(f"Number of Ticks: {n_ticks}")
        print(f"\nCustom Colors (RGBA):")
        for color_str, rgba in RGBA.items():
            print(f"  {color_str}: {rgba}")
        print("-" * 40)

    # Store n_ticks for later use
    plt._n_ticks = n_ticks

    # Add a utility function to plt
    def set_n_ticks(ax, n=None):
        if n is None:
            n = plt._n_ticks
        ax.xaxis.set_major_locator(plt.MaxNLocator(n))
        ax.yaxis.set_major_locator(plt.MaxNLocator(n))

    plt.set_n_ticks = set_n_ticks

    # Convert to DotDict for convenient access (COLORS.blue or COLORS['blue'])
    return plt, DotDict(RGBA_NORM)


# def _convert_font_size(size: Union[str, int, float]) -> float:
#     """Converts various font size specifications to numerical values.

#     Parameters
#     ----------
#     size : Union[str, int, float]
#         Font size specification. Can be:
#         - String: 'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
#         - Numeric: direct point size value

#     Returns
#     -------
#     float
#         Font size in points
#     """
#     if isinstance(size, str):
#         size_map = {
#             "xx-small": 9,
#             "x-small": 11,
#             "small": 13,
#             "medium": 15,
#             "large": 18,
#             "x-large": 22,
#             "xx-large": 26,
#         }
#         return size_map.get(size.lower(), 15)
#     elif isinstance(size, (int, float)):
#         return max(float(size), 9.0)  # Ensure minimum size of 9
#     else:
#         raise ValueError(f"Unsupported font size type: {type(size)}")


if __name__ == "__main__":
    plt, CC = configure_mpl(plt)
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True)
    x = np.linspace(0, 10, 100)
    for idx_cc, cc_str in enumerate(CC):
        phase_shift = idx_cc * np.pi / len(CC)
        y = np.sin(x + phase_shift)
        axes[0].plot(x, y, label="Default color cycle")
        axes[1].plot(x, y, color=CC[cc_str], label=f"{cc_str}")
    axes[0].legend()
    axes[1].legend()
    plt.show()

# EOF
