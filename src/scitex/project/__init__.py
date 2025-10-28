#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/project/__init__.py

"""
SciTeX Project Management

This module provides standalone project management for SciTeX research projects.
Projects are self-contained with metadata stored in scitex/.metadata/ directory.

Core Features:
- Project creation and loading
- Metadata persistence (scitex/.metadata/)
- Storage calculation
- Name validation (GitHub/Gitea compatible)
- Git integration
- Works standalone without Django or Gitea

Examples:
    Create a new project:
        >>> from scitex.project import SciTeXProject
        >>> project = SciTeXProject.create(
        ...     name="My Research",
        ...     path=Path("/path/to/project"),
        ...     owner="ywatanabe",
        ...     description="Neural decoding research"
        ... )

    Load existing project:
        >>> project = SciTeXProject.load_from_directory(Path("/path/to/project"))

    Update storage:
        >>> project.update_storage_usage()
        >>> print(f"Storage: {project.storage_used / 1024**2:.2f} MB")

    Validate project name:
        >>> from scitex.project import validate_name
        >>> is_valid, error = validate_name("my-project")
"""

from .core import SciTeXProject
from .validators import (
    ProjectValidator,
    validate_name,
    generate_slug,
    extract_repo_name,
)
from .metadata import (
    ProjectMetadataStore,
    generate_project_id,
)
from .storage import (
    ProjectStorageCalculator,
    calculate_storage,
    format_size,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    'SciTeXProject',

    # Validators
    'ProjectValidator',
    'validate_name',
    'generate_slug',
    'extract_repo_name',

    # Metadata
    'ProjectMetadataStore',
    'generate_project_id',

    # Storage
    'ProjectStorageCalculator',
    'calculate_storage',
    'format_size',
]

# EOF
