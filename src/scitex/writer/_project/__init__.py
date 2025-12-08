#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_project/__init__.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_project/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Project initialization helpers for writer module.

Handles project creation, attachment, and validation.
"""

from ._create import ensure_project_exists
from ._validate import validate_structure
from ._trees import create_document_trees

__all__ = [
    "ensure_project_exists",
    "validate_structure",
    "create_document_trees",
]

# EOF
