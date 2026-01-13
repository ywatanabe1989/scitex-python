#!/usr/bin/env python3
"""
SciTeX Scholar Commands - CLI for literature management.

Fetch papers to your library by DOI or title. Downloads PDFs, enriches metadata,
and organizes everything in a searchable library.
"""

import asyncio
import sys
from pathlib import Path

import click

from scitex import logging
from scitex.config import get_paths

logger = logging.getLogger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def scholar():
    """
    Fetch papers to your library

    \b
    Downloads PDFs, enriches metadata, and organizes papers:
    - Resolve DOI/title to full metadata
    - Download PDF via browser automation
    - Store in organized library structure

    \b
    Storage: ~/.scitex/scholar/library/
    """
    pass


@scholar.command()
@click.argument("papers", nargs=-1)
@click.option(
    "--from-bibtex",
    "bibtex_file",
    type=click.Path(exists=True),
    help="Import papers from BibTeX file",
)
@click.option("--project", "-p", help="Project name for organizing papers")
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: auto)",
)
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
@click.option(
    "--output",
    "-o",
    help="Output path for enriched BibTeX (only with --from-bibtex)",
)
def fetch(
    papers, bibtex_file, project, workers, browser_mode, chrome_profile, force, output
):
    """
    Fetch papers to your library

    Provide DOIs or titles as arguments. Papers are downloaded, metadata is
    enriched, and everything is stored in your library.

    \b
    Examples:
        scitex scholar fetch "10.1038/nature12373"
        scitex scholar fetch "10.1038/nature12373" "10.1016/j.neuron.2018.01.023"
        scitex scholar fetch "Spike sorting methods" --project neuroscience
        scitex scholar fetch --from-bibtex papers.bib --project myresearch
        scitex scholar fetch "10.1038/nature12373" --force

    \b
    TIP: Get BibTeX files from Scholar QA (https://scholarqa.allen.ai/chat/)
    """
    # Validate input
    if not papers and not bibtex_file:
        click.echo("Error: Provide DOIs/titles or use --from-bibtex", err=True)
        click.echo("\nUsage: scitex scholar fetch <doi_or_title>...", err=True)
        click.echo("       scitex scholar fetch --from-bibtex papers.bib", err=True)
        sys.exit(1)

    if papers and bibtex_file:
        click.echo(
            "Error: Cannot mix positional arguments with --from-bibtex", err=True
        )
        sys.exit(1)

    # Handle BibTeX import
    if bibtex_file:
        _add_from_bibtex(
            bibtex_file=bibtex_file,
            project=project,
            workers=workers,
            browser_mode=browser_mode,
            chrome_profile=chrome_profile,
            output=output,
        )
        return

    # Handle DOIs/titles
    papers = list(papers)
    num_papers = len(papers)

    # Auto-determine workers
    if workers is None:
        workers = min(num_papers, 4)  # Max 4 by default

    if num_papers == 1:
        _add_single(
            doi_or_title=papers[0],
            project=project,
            browser_mode=browser_mode,
            chrome_profile=chrome_profile,
            force=force,
        )
    else:
        _add_multiple(
            papers=papers,
            project=project,
            workers=workers,
            browser_mode=browser_mode,
            chrome_profile=chrome_profile,
        )


def _add_single(doi_or_title, project, browser_mode, chrome_profile, force):
    """Add a single paper to library."""
    from scitex.scholar.pipelines.ScholarPipelineSingle import ScholarPipelineSingle

    logger.info(f"Fetching: {doi_or_title}")

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

        logger.success("Paper fetched")
        if symlink_path:
            logger.info(f"  Location: {symlink_path}")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


def _add_multiple(papers, project, workers, browser_mode, chrome_profile):
    """Add multiple papers to library in parallel."""
    from scitex.scholar.pipelines.ScholarPipelineParallel import ScholarPipelineParallel

    logger.info(f"Fetching {len(papers)} papers ({workers} workers)")

    async def run():
        pipeline = ScholarPipelineParallel(
            num_workers=workers,
            browser_mode=browser_mode,
            base_chrome_profile=chrome_profile,
        )

        results = await pipeline.process_papers_from_list_async(
            doi_or_title_list=papers,
            project=project,
        )

        logger.success(f"{len(results)} papers fetched")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


def _add_from_bibtex(
    bibtex_file, project, workers, browser_mode, chrome_profile, output
):
    """Fetch papers from BibTeX file."""
    from scitex.scholar.pipelines.ScholarPipelineBibTeX import ScholarPipelineBibTeX

    bibtex_path = Path(bibtex_file)
    workers = workers or 4

    logger.info(f"Fetching papers from: {bibtex_path.name}")

    async def run():
        pipeline = ScholarPipelineBibTeX(
            num_workers=workers,
            browser_mode=browser_mode,
            base_chrome_profile=chrome_profile,
        )

        results = await pipeline.process_bibtex_file_async(
            bibtex_path=bibtex_path,
            project=project,
            output_bibtex_path=output,
        )

        logger.success(f"{len(results)} papers fetched from BibTeX")
        if output:
            logger.info(f"  Enriched BibTeX: {output}")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)


@scholar.command()
@click.option("--project", "-p", help="Show specific project")
def library(project):
    """
    Show your paper library

    \b
    Examples:
        scitex scholar library
        scitex scholar library --project neuroscience
    """
    library_path = get_paths().scholar_library

    if not library_path.exists():
        click.echo("Library is empty. Fetch papers with: scitex scholar fetch <doi>")
        return

    if project:
        project_path = library_path / project
        if not project_path.exists():
            click.echo(f"Project '{project}' not found")
            return

        click.echo(f"\nProject: {project}")
        click.echo(f"Location: {project_path}")
        click.echo("\nPapers:")

        for item in sorted(project_path.iterdir()):
            if item.is_symlink():
                click.echo(f"  {item.name}")
    else:
        master_path = library_path / "MASTER"
        project_dirs = [
            d for d in library_path.iterdir() if d.is_dir() and d.name != "MASTER"
        ]

        if master_path.exists():
            num_papers = len(list(master_path.iterdir()))
            click.echo(f"\nTotal papers: {num_papers}")

        if project_dirs:
            click.echo(f"\nProjects ({len(project_dirs)}):")
            for proj_dir in sorted(project_dirs):
                num_papers = len([p for p in proj_dir.iterdir() if p.is_symlink()])
                click.echo(f"  {proj_dir.name} ({num_papers} papers)")
        else:
            click.echo("\nNo projects. Use --project/-p when adding papers.")


@scholar.command()
def config():
    """
    Show Scholar configuration
    """
    library_path = get_paths().scholar_library

    click.echo("\n=== SciTeX Scholar ===\n")
    click.echo(f"Library: {library_path}")

    if library_path.exists():
        master_path = library_path / "MASTER"
        if master_path.exists():
            num_papers = len(list(master_path.iterdir()))
            click.echo(f"Papers:  {num_papers}")

    chrome_config_path = Path.home() / ".config" / "google-chrome"
    click.echo(
        f"\nChrome:  {'Available' if chrome_config_path.exists() else 'Not found'}"
    )


# EOF
