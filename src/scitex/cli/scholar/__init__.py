#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/cli/scholar/__init__.py
# ----------------------------------------

"""Scholar CLI commands.

This module provides the command-line interface for SciTeX Scholar.

Usage:
    scitex scholar fetch "10.1038/nature12373"
    scitex scholar fetch --from-bibtex papers.bib --project myresearch
    scitex scholar fetch "10.1038/nature12373" --async
    scitex scholar library
    scitex scholar config
    scitex scholar jobs list
    scitex scholar jobs status <job_id>
"""

from __future__ import annotations

import click

from ._fetch import fetch
from ._jobs import jobs
from ._library import config, library


@click.group()
def scholar():
    """
    Scientific paper management

    \b
    Fetch papers, manage your library, and track background jobs.

    \b
    Examples:
        scitex scholar fetch "10.1038/nature12373"
        scitex scholar fetch --from-bibtex refs.bib -p myproject
        scitex scholar library
        scitex scholar jobs list
    """
    pass


scholar.add_command(fetch)
scholar.add_command(library)
scholar.add_command(config)
scholar.add_command(jobs)

__all__ = ["scholar"]

# EOF
