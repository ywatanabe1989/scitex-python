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

from .writer import Writer
from .types import (
    DocumentSection,
    Document,
    ManuscriptDocument,
    ManuscriptContents,
    SupplementaryDocument,
    SupplementaryContents,
    RevisionDocument,
    RevisionContents,
)
from .compile import (
    compile_manuscript,
    compile_supplementary,
    compile_revision,
    CompilationResult,
)
from .watch import watch_manuscript
from ._init_directory import init_directory
from .config import WriterConfig, find_writer_root
from .validate import (
    ProjectValidationError,
    validate_manuscript_structure,
    validate_supplementary_structure,
    validate_revision_structure,
    validate_all_documents,
    list_missing_files,
)
from .parse_latex import (
    LaTeXIssue,
    parse_latex_log,
    parse_compilation_output,
    format_issues,
    summarize_issues,
)
from .compile_async import (
    compile_manuscript_async,
    compile_supplementary_async,
    compile_revision_async,
    compile_all_async,
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    'Writer',

    # Type definitions
    'DocumentSection',
    'Document',

    # Manuscript types
    'ManuscriptDocument',
    'ManuscriptContents',

    # Supplementary types
    'SupplementaryDocument',
    'SupplementaryContents',

    # Revision types
    'RevisionDocument',
    'RevisionContents',

    # Initialization
    'init_directory',

    # Compilation (for advanced use)
    'compile_manuscript',
    'compile_supplementary',
    'compile_revision',
    'CompilationResult',

    # Watch mode (for advanced use)
    'watch_manuscript',

    # Configuration (for advanced use)
    'WriterConfig',
    'find_writer_root',

    # Validation (for advanced use)
    'ProjectValidationError',
    'validate_manuscript_structure',
    'validate_supplementary_structure',
    'validate_revision_structure',
    'validate_all_documents',
    'list_missing_files',

    # LaTeX error parsing (for advanced use)
    'LaTeXIssue',
    'parse_latex_log',
    'parse_compilation_output',
    'format_issues',
    'summarize_issues',

    # Async compilation (for advanced use)
    'compile_manuscript_async',
    'compile_supplementary_async',
    'compile_revision_async',
    'compile_all_async',
]

# EOF
