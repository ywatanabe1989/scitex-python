#!/usr/bin/env python3
# Timestamp: "2025-10-01 17:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/normality/_test_ks.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


"""
Functionalities:
  - Perform Kolmogorov-Smirnov test for distribution comparison
  - One-sample KS test (compare to reference distribution)
  - Two-sample KS test (compare two empirical distributions)
  - Generate CDF comparison plots
  - Support flexible output formats (dict or DataFrame)

Dependencies:
  - packages: numpy, pandas, scipy, matplotlib

IO:
  - input: One or two samples (arrays or Series)
  - output: Test results (dict or DataFrame) and optional figure

Note:
  - This module is a thin shim that re-exports from split implementation files
  - Actual implementations are in _test_ks_1samp.py and _test_ks_2samp.py
"""

# Re-export test functions from split modules
from ._test_ks_1samp import test_ks_1samp
from ._test_ks_2samp import test_ks_2samp

__all__ = ["test_ks_1samp", "test_ks_2samp"]

# EOF
