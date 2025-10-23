#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/__init__.py

"""
SciTeX Writer - LaTeX Compilation System

Python wrapper around scitex-writer shell scripts for LaTeX compilation.

Features:
- Compile manuscript, supplementary, and revision documents
- Watch mode for auto-recompilation
- Exit code and output handling
- Integration with scitex.project

Examples:
    Compile manuscript:
        >>> from scitex.writer import compile_manuscript
        >>> result = compile_manuscript(project_dir=Path("/path/to/project"))
        >>> if result.success:
        ...     print(f"PDF: {result.output_pdf}")

    Watch mode:
        >>> from scitex.writer import watch_manuscript
        >>> watch_manuscript(project_dir=Path("/path/to/project"))

    With SciTeXProject:
        >>> from scitex.project import SciTeXProject
        >>> project = SciTeXProject.load_from_directory(Path("/path"))
        >>> writer_dir = project.get_scitex_directory('writer')
        >>> result = compile_manuscript(project_dir=writer_dir)
"""

from .compile import (
    compile_manuscript,
    compile_supplementary,
    compile_revision,
    CompilationResult,
)
from .watch import watch_manuscript
from .template import create_writer_project, copy_template
from .config import WriterConfig, find_writer_root

__version__ = "0.1.0"

__all__ = [
    # Compilation
    'compile_manuscript',
    'compile_supplementary',
    'compile_revision',
    'CompilationResult',

    # Watch mode
    'watch_manuscript',

    # Templates
    'create_writer_project',
    'copy_template',

    # Configuration
    'WriterConfig',
    'find_writer_root',
]

# EOF
