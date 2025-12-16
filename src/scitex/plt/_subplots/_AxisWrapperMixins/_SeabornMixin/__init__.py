#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: __init__.py - SeabornMixin package

"""
SeabornMixin - Modular seaborn integration mixin for AxisWrapper.

This package provides seaborn plotting functionality:
- _base: Helper methods and data preparation
- _wrappers: Individual seaborn plot wrappers (sns_xxx)
"""

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)

from ._base import SeabornBaseMixin, sns_copy_doc
from ._wrappers import SeabornWrappersMixin


class SeabornMixin(SeabornBaseMixin, SeabornWrappersMixin):
    """Mixin class for seaborn plotting integration.

    Combines:
    - SeabornBaseMixin: Helper methods for tracking and data preparation
    - SeabornWrappersMixin: Individual sns_ prefixed seaborn wrappers
    """
    pass


# EOF
