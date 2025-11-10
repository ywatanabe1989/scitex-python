#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/conftest.py
# ----------------------------------------
"""
Pytest configuration to ensure tests import from local source.
"""

import sys
from pathlib import Path

# Add the src directory to Python path to ensure local imports
repo_root = Path(__file__).parent
src_path = repo_root / "src"

# Insert at the very beginning to override any installed package
if str(src_path) in sys.path:
    sys.path.remove(str(src_path))
sys.path.insert(0, str(src_path))

# Clear any already-imported scitex modules to force reload from correct path
modules_to_clear = [key for key in sys.modules.keys() if key.startswith('scitex')]
for module_name in modules_to_clear:
    del sys.modules[module_name]

# EOF
