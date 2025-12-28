#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/tests/scitex/git/test_utils.py

"""Tests for git utilities."""

import tempfile
from pathlib import Path

import pytest
pytest.importorskip("git")

from scitex.git._utils import _in_directory


class TestUtils:
    def test_in_directory_context(self):
        cwd_original = Path.cwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            with _in_directory(temp_path):
                assert Path.cwd() == temp_path

            assert Path.cwd() == cwd_original

    def test_in_directory_restore_on_exception(self):
        cwd_original = Path.cwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            try:
                with _in_directory(temp_path):
                    raise ValueError("test exception")
            except ValueError:
                pass

            assert Path.cwd() == cwd_original

# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_utils.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/git/ops.py
# 
# """
# Git operations utilities.
# """
# 
# import os
# from contextlib import contextmanager
# from pathlib import Path
# 
# 
# @contextmanager
# def _in_directory(path: Path):
#     cwd_original = Path.cwd()
#     try:
#         os.chdir(path)
#         yield
#     finally:
#         os.chdir(cwd_original)
# 
# 
# __all__ = [
#     "_in_directory",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/git/_utils.py
# --------------------------------------------------------------------------------
