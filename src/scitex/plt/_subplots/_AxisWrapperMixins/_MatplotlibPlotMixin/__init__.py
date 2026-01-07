#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: __init__.py - MatplotlibPlotMixin package

"""
MatplotlibPlotMixin - Modular plotting mixin for AxisWrapper.

This package provides plotting functionality split into logical submodules:
- _base: Core helper methods
- _scientific: Scientific/specialized plots (stx_image, stx_kde, stx_conf_mat, etc.)
- _statistical: Statistical plots (stx_line, stx_mean_std, stx_box, stx_violin, hist)
- _stx_aliases: stx_ prefixed aliases for standard matplotlib methods

API Layer Design:
-----------------
stx_* (SciTeX canonical):
  - Full tracking, metadata, and reproducibility support
  - Output connects to .plot / .figure format
  - Purpose: publication / reproducibility

mpl_* (Matplotlib compatibility - see _RawMatplotlibMixin):
  - Raw matplotlib API without scitex processing
  - Purpose: compatibility / low-level control / escape hatch

sns_* (Seaborn - see _SeabornMixin):
  - DataFrame-centric with data=, x=, y=, hue= interface
  - Purpose: exploratory / grouped stats
"""

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

from ._base import PlotBaseMixin
from ._scientific import ScientificPlotMixin
from ._statistical import StatisticalPlotMixin
from ._stx_aliases import StxAliasesMixin


class MatplotlibPlotMixin(
    PlotBaseMixin,
    ScientificPlotMixin,
    StatisticalPlotMixin,
    StxAliasesMixin,
):
    """Mixin class for basic plotting operations.

    Combines multiple specialized mixins:
    - PlotBaseMixin: Core helper methods (_get_ax_module, _apply_scitex_postprocess)
    - ScientificPlotMixin: Scientific plots (stx_image, stx_kde, stx_conf_mat, etc.)
    - StatisticalPlotMixin: Statistical line plots and distributions
    - StxAliasesMixin: stx_ prefixed matplotlib aliases
    """

    pass


# EOF
