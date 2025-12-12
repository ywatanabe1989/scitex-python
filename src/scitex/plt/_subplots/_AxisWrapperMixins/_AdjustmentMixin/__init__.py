#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: __init__.py - AdjustmentMixin package

"""
AdjustmentMixin - Modular axis adjustment mixin for AxisWrapper.

This package provides axis adjustment functionality split into logical submodules:
- _labels: Label rotation and legend positioning
- _metadata: Axis labels, titles, and scientific metadata
- _visual: Visual adjustments (ticks, spines, extend, shift)
"""

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

from ._labels import LabelsMixin
from ._metadata import MetadataMixin
from ._visual import VisualAdjustmentMixin


class AdjustmentMixin(LabelsMixin, MetadataMixin, VisualAdjustmentMixin):
    """Mixin class for matplotlib axis adjustments.

    Combines multiple specialized mixins:
    - LabelsMixin: Label rotation and legend positioning
    - MetadataMixin: Axis labels, titles, and scientific metadata
    - VisualAdjustmentMixin: Ticks, spines, extend, shift
    """
    pass


# EOF
