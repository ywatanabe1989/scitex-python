#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX Scholar Commands - CLI wrapper for scitex.scholar module

Wraps existing Scholar implementation (browser automation, PDF download, metadata enrichment)
into the unified scitex CLI interface.
"""

import click
import sys
import asyncio
from pathlib import Path

from scitex.config import get_paths


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def scholar():
    """
    Literature management with browser automation

    \b
    Provides scientific literature search, PDF download, and metadata enrichment:
    - Single or batch paper processing
    - BibTeX enrichment
    - Browser automation for PDF downloads
    - OpenAthens/institutional access support

    \b
    Storage: $SCITEX_DIR/scholar/library/
    Backend: Browser automation + metadata APIs
    """
    pass


@scholar.command()
@click.option("--doi", help='DOI of the paper (e.g., "10.1038/nature12373")')
@click.option("--title", help="Paper title (will resolve DOI automatically)")
@click.option("--project", help="Project name for organizing papers")
@click.option(
    "--browser-mode",
    type=click.Choice(["stealth", "interactive"]),
    default="stealth",
    help="Browser mode for PDF download",
)
@click.option("--chrome-profile", default="system", help="Chrome profile name")
@click.option(
    "--force", "-f", is_flag=True, help="Force re-download even if files exist"
)
def single(doi, title, project, browser_mode, chrome_profile, force):
    """
    Process a single paper (DOI or title)

    Downloads PDF, enriches metadata, and stores in library.

    \b
    Examples:
        scitex scholar single --doi "10.1038/nature12373"
        scitex scholar single --title "Spike sorting" --project neuroscience
        scitex scholar single --doi "10.1016/j.neuron.2018.01.023" --force
    """
    if not doi and not title:
        click.echo("Error: Either --doi or --title is required", err=True)
        sys.exit(1)

    # Import here to avoid slow startup
    from scitex.scholar.pipelines.ScholarPipelineSingle import ScholarPipelineSingle

    doi_or_title = doi if doi else title

    click.echo(f"Processing paper: {doi_or_title}")

    async def run():
        pipeline = ScholarPipelineSingle(
            browser_mode=browser_mode,
            chrome_profile=chrome_profile,
        )

        paper, symlink_path = await pipeline.process_single_paper(
            doi_or_title=doi_or_title,
            project=project,
            force=force,
        )

        click.echo(f"✓ Paper processed successfully")
        if symlink_path:
            click.echo(f"  Location: {symlink_path}")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@scholar.command()
@click.option(
    "--dois", multiple=True, help="DOIs to process (can be specified multiple times)"
)
@click.option(
    "--titles",
    multiple=True,
    help="Paper titles to process (can be specified multiple times)",
)
@click.option("--project", help="Project name for organizing papers")
@click.option("--num-workers", type=int, default=4, help="Number of parallel workers")
@click.option(
    "--browser-mode",
    type=click.Choice(["stealth", "interactive"]),
    default="stealth",
    help="Browser mode for all workers",
)
@click.option(
    "--chrome-profile", default="system", help="Base Chrome profile to sync from"
)
def parallel(dois, titles, project, num_workers, browser_mode, chrome_profile):
    """
    Process multiple papers in parallel

    Uses multiple browser workers for efficient batch processing.

    \b
    Examples:
        scitex scholar parallel --dois 10.1038/nature12373 --dois 10.1016/j.neuron.2018.01.023
        scitex scholar parallel --titles "Spike sorting" --titles "Neural networks" --num-workers 8
        scitex scholar parallel --dois 10.1038/nature12373 --project neuroscience
    """
    if not dois and not titles:
        click.echo("Error: Either --dois or --titles is required", err=True)
        sys.exit(1)

    # Import here to avoid slow startup
    from scitex.scholar.pipelines.ScholarPipelineParallel import ScholarPipelineParallel

    # Combine DOIs and titles
    queries = list(dois) + list(titles)

    click.echo(f"Processing {len(queries)} papers with {num_workers} workers...")

    async def run():
        pipeline = ScholarPipelineParallel(
            num_workers=num_workers,
            browser_mode=browser_mode,
            base_chrome_profile=chrome_profile,
        )

        papers = await pipeline.process_papers_from_list_async(
            doi_or_title_list=queries,
            project=project,
        )

        click.echo(f"✓ {len(papers)} papers processed successfully")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@scholar.command()
@click.argument("bibtex_file", type=click.Path(exists=True))
@click.option("--project", help="Project name for organizing papers")
@click.option(
    "--output", help="Output path for enriched BibTeX (default: {input}_processed.bib)"
)
@click.option("--num-workers", type=int, default=4, help="Number of parallel workers")
@click.option(
    "--browser-mode",
    type=click.Choice(["stealth", "interactive"]),
    default="stealth",
    help="Browser mode for all workers",
)
@click.option(
    "--chrome-profile", default="system", help="Base Chrome profile to sync from"
)
def bibtex(bibtex_file, project, output, num_workers, browser_mode, chrome_profile):
    """
    Process papers from BibTeX file

    Enriches BibTeX entries with metadata and downloads PDFs in parallel.

    \b
    Examples:
        scitex scholar bibtex papers.bib --project myresearch
        scitex scholar bibtex refs.bib --output enriched.bib --num-workers 8
        scitex scholar bibtex library.bib --browser-mode interactive

    \b
    TIP: Get BibTeX files from Scholar QA (https://scholarqa.allen.ai/chat/)
         Ask questions → Export All Citations → Save as .bib file
    """
    # Import here to avoid slow startup
    from scitex.scholar.pipelines.ScholarPipelineBibTeX import ScholarPipelineBibTeX

    bibtex_path = Path(bibtex_file)

    click.echo(f"Processing BibTeX file: {bibtex_path}")

    async def run():
        pipeline = ScholarPipelineBibTeX(
            num_workers=num_workers,
            browser_mode=browser_mode,
            base_chrome_profile=chrome_profile,
        )

        papers = await pipeline.process_bibtex_file_async(
            bibtex_path=bibtex_path,
            project=project,
            output_bibtex_path=output,
        )

        click.echo(f"✓ {len(papers)} papers processed from BibTeX")
        if output:
            click.echo(f"  Enriched BibTeX saved to: {output}")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@scholar.command()
@click.option("--project", help="Show library for specific project")
def library(project):
    """
    Show your Scholar library

    \b
    Examples:
        scitex scholar library
        scitex scholar library --project neuroscience
    """
    library_path = get_paths().scholar_library

    if not library_path.exists():
        click.echo("No library found. Process some papers first!")
        return

    if project:
        project_path = library_path / project
        if not project_path.exists():
            click.echo(f"Project '{project}' not found in library")
            return

        click.echo(f"\nProject: {project}")
        click.echo(f"Location: {project_path}")
        click.echo("\nPapers:")

        # List symlinks in project directory
        for item in sorted(project_path.iterdir()):
            if item.is_symlink():
                click.echo(f"  - {item.name}")
    else:
        # List all projects
        master_path = library_path / "MASTER"
        project_dirs = [
            d for d in library_path.iterdir() if d.is_dir() and d.name != "MASTER"
        ]

        if master_path.exists():
            num_papers = len(list(master_path.iterdir()))
            click.echo(f"\nTotal papers in library: {num_papers}")

        if project_dirs:
            click.echo(f"\nProjects ({len(project_dirs)}):")
            for proj_dir in sorted(project_dirs):
                num_papers = len([p for p in proj_dir.iterdir() if p.is_symlink()])
                click.echo(f"  - {proj_dir.name} ({num_papers} papers)")
        else:
            click.echo("\nNo projects yet. Use --project when processing papers.")


@scholar.command()
def config():
    """
    Show Scholar configuration

    Displays library location, browser settings, and authentication status.
    """
    library_path = get_paths().scholar_library

    click.echo("\n=== SciTeX Scholar Configuration ===\n")
    click.echo(f"Library location: {library_path}")
    click.echo(f"Library exists: {'Yes' if library_path.exists() else 'No'}")

    if library_path.exists():
        master_path = library_path / "MASTER"
        if master_path.exists():
            num_papers = len(list(master_path.iterdir()))
            click.echo(f"Papers in library: {num_papers}")

    # Check for Chrome profiles
    chrome_config_path = Path.home() / ".config" / "google-chrome"
    click.echo(f"\nChrome config: {chrome_config_path}")
    click.echo(f"Chrome available: {'Yes' if chrome_config_path.exists() else 'No'}")

    click.echo("\nFor more info, see:")
    click.echo("  https://github.com/ywatanabe1989/scitex-code")


# EOF
