#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/git/test_constants.py

"""Tests for git constants."""

import pytest
pytest.importorskip("git")

from scitex.git._constants import EXIT_FAILURE, EXIT_SUCCESS


class TestConstants:
    def test_exit_success_value(self):
        assert EXIT_SUCCESS == 0

    def test_exit_failure_value(self):
        assert EXIT_FAILURE == 1

    def test_exit_values_different(self):
        assert EXIT_SUCCESS != EXIT_FAILURE

# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_constants.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/constants.py
# 
# """
# Constants for git module.
# """
# 
# EXIT_SUCCESS = 0
# EXIT_FAILURE = 1
# 
# __all__ = [
#     "EXIT_SUCCESS",
#     "EXIT_FAILURE",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_constants.py
# --------------------------------------------------------------------------------
