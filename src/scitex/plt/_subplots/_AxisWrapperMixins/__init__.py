#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_AxisWrapperMixins/__init__.py

"""
AxisWrapper Mixins - Modular plotting API for SciTeX.

API Layers
==========

SciTeX provides three distinct API layers for plotting, each with different
purposes and trade-offs:

stx_* (SciTeX Canonical)
------------------------
- Input: ArrayLike / List / ndarray
- Output: (Axes, tracked_df, meta)
- Purpose: publication / reproducibility
- Features:
  * Full tracking and metadata support
  * Output connects to .pltz / .figz formats
  * Automatic styling according to SciTeX style
  * Primary API - recommended for final figures

  Examples:
    ax.stx_bar(x, height)
    ax.stx_scatter(x, y, label="Data")
    ax.stx_kde(data)

mpl_* (Matplotlib Compatibility)
--------------------------------
- Input: Same as matplotlib
- Output: matplotlib artists
- Purpose: compatibility / low-level control / escape hatch
- Features:
  * No tracking, no scitex processing
  * Direct matplotlib API access
  * Use for unsupported operations or migration

  Examples:
    ax.mpl_plot(x, y)
    ax.mpl_scatter(x, y)
    ax.mpl_raw("some_method", *args)  # Call any matplotlib method

sns_* (Seaborn / DataFrame-Centric)
-----------------------------------
- Input: DataFrame + column names (data=, x=, y=, hue=)
- Output: Axes (+ summarized df)
- Purpose: exploratory / grouped statistics
- Features:
  * DataFrame-centric interface
  * Statistical summaries and grouping
  * Familiar seaborn UX

  Examples:
    ax.sns_boxplot(data=df, x="group", y="value")
    ax.sns_histplot(data=df, x="measurement", hue="category")

Choosing an API Layer
=====================

Use stx_*:
  - For publication-ready figures
  - When you need reproducibility and tracking
  - As your default choice

Use mpl_*:
  - For low-level matplotlib control
  - When migrating existing matplotlib code
  - For matplotlib features not yet wrapped

Use sns_*:
  - For exploratory data analysis
  - When input is a DataFrame
  - For statistical visualization with grouping
"""

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

from ._AdjustmentMixin import AdjustmentMixin
from ._MatplotlibPlotMixin import MatplotlibPlotMixin
from ._RawMatplotlibMixin import RawMatplotlibMixin, MPL_METHODS
from ._SeabornMixin import SeabornMixin
from ._TrackingMixin import TrackingMixin
from ._UnitAwareMixin import UnitAwareMixin

# EOF
