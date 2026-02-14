#!/usr/bin/env python3
# Timestamp: "2025-10-01 15:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/stats/tests/_test_ttest.py
# ----------------------------------------
"""
Backward compatibility shim for t-test functions.

This file re-exports test functions from split modules.
The original 808-line file was split into three separate files
to keep each under 512 lines for maintainability.

Use the individual modules directly for new code:
- _test_ttest_ind: Independent samples t-test
- _test_ttest_rel: Paired samples t-test
- _test_ttest_1samp: One-sample t-test
"""

from ._test_ttest_1samp import test_ttest_1samp
from ._test_ttest_ind import test_ttest_ind
from ._test_ttest_rel import test_ttest_rel

__all__ = ["test_ttest_ind", "test_ttest_rel", "test_ttest_1samp"]

# EOF
