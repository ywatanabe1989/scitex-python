#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX CLI Main Entry Point
"""

import click
from . import cloud, scholar


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option()
def cli():
    """
    SciTeX - Integrated Scientific Research Platform

    \b
    Commands:
      cloud     Git operations (via Gitea) - AVAILABLE NOW
      scholar   Literature management - AVAILABLE NOW
      code      Analysis execution (coming soon)
      viz       Visualization (coming soon)
      writer    Manuscript writing (coming soon)
      project   Integrated workflows (coming soon)

    \b
    Examples:
      scitex cloud login
      scitex cloud clone ywatanabe/my-project
      scitex scholar bibtex papers.bib --project myresearch
      scitex scholar single --doi "10.1038/nature12373"
    """
    pass


# Add command groups
cli.add_command(cloud.cloud)
cli.add_command(scholar.scholar)


if __name__ == '__main__':
    cli()
