#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/scholar/jobs/_executors.py
# ----------------------------------------

"""Job executors for Scholar operations.

These functions wrap Scholar pipelines to work with the JobManager.
Each executor accepts a progress_callback for real-time progress updates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from ._errors import create_structured_error


async def fetch_single_executor(
    doi_or_title: str,
    project: str | None = None,
    browser_mode: str = "stealth",
    chrome_profile: str = "system",
    force: bool = False,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Execute a single paper fetch.

    Args:
        doi_or_title: DOI or title to fetch
        project: Project name for organizing
        browser_mode: Browser mode (stealth/interactive)
        chrome_profile: Chrome profile to use
        force: Force re-download
        progress_callback: Callback for progress updates

    Returns
    -------
        Result dictionary with paper info and status
    """
    from scitex.scholar.pipelines import ScholarPipelineSingle

    # Record start time for screenshot collection
    start_timestamp = datetime.now().isoformat()

    if progress_callback:
        progress_callback(total=1, completed=0, current_item=doi_or_title)

    pipeline = ScholarPipelineSingle(
        browser_mode=browser_mode,
        chrome_profile=chrome_profile,
    )

    try:
        paper, symlink_path = await pipeline.process_single_paper(
            doi_or_title=doi_or_title,
            project=project,
            force=force,
        )

        # Granular success flags
        has_doi = bool(paper and paper.metadata.id.doi)
        has_metadata = bool(paper and paper.metadata.basic.title)
        has_pdf = bool(symlink_path)
        has_content = bool(
            paper
            and hasattr(paper, "container")
            and paper.container.pdf_size_bytes
            and paper.container.pdf_size_bytes > 0
        )
        pdf_method = None
        if paper and paper.metadata.path.pdfs_engines:
            pdf_method = paper.metadata.path.pdfs_engines[0]

        if progress_callback:
            progress_callback(
                completed=1, message="Completed" if has_pdf else "Completed (no PDF)"
            )

        return {
            "success": has_pdf,  # Overall success = PDF obtained
            "success_doi": has_doi,
            "success_metadata": has_metadata,
            "success_pdf": has_pdf,
            "success_content": has_content,
            "pdf_method": pdf_method,
            "message": "Paper fetched"
            if has_pdf
            else "Metadata fetched but PDF not downloaded",
            "doi": paper.metadata.id.doi if paper else None,
            "title": paper.metadata.basic.title if paper else None,
            "path": str(symlink_path) if symlink_path else None,
            "has_pdf": has_pdf,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        if progress_callback:
            progress_callback(failed=1, message=f"Failed: {e}")

        # Create structured error with categorization and screenshots
        structured_error = create_structured_error(
            error=e,
            doi_or_title=doi_or_title,
            since_timestamp=start_timestamp,
            include_screenshots=True,
        )

        return {
            "success": False,
            "error": str(e),
            "structured_error": structured_error.to_dict(),
            "doi_or_title": doi_or_title,
            "timestamp": datetime.now().isoformat(),
        }


async def fetch_multiple_executor(
    papers: list[str],
    project: str | None = None,
    workers: int = 4,
    browser_mode: str = "stealth",
    chrome_profile: str = "system",
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Execute multiple paper fetches in parallel.

    Args:
        papers: List of DOIs or titles to fetch
        project: Project name for organizing
        workers: Number of parallel workers
        browser_mode: Browser mode
        chrome_profile: Chrome profile
        progress_callback: Callback for progress updates

    Returns
    -------
        Result dictionary with stats and paper list
    """
    from scitex.scholar.pipelines import ScholarPipelineParallel

    # Record start time for screenshot collection
    start_timestamp = datetime.now().isoformat()

    total = len(papers)
    if progress_callback:
        progress_callback(total=total, completed=0, message="Starting...")

    pipeline = ScholarPipelineParallel(
        num_workers=workers,
        browser_mode=browser_mode,
        base_chrome_profile=chrome_profile,
    )

    # Track progress
    completed_count = 0
    failed_count = 0
    results_list = []

    async def process_with_progress():
        nonlocal completed_count, failed_count

        papers_result = await pipeline.process_papers_from_list_async(
            doi_or_title_list=papers,
            project=project,
        )

        for paper in papers_result:
            if paper:
                completed_count += 1
                results_list.append(
                    {
                        "doi": paper.metadata.id.doi,
                        "title": paper.metadata.basic.title,
                        "success": True,
                    }
                )
            else:
                failed_count += 1
                results_list.append({"success": False})

            if progress_callback:
                progress_callback(
                    completed=completed_count,
                    failed=failed_count,
                    message=f"Processing {completed_count + failed_count}/{total}",
                )

        return papers_result

    try:
        await process_with_progress()

        if progress_callback:
            progress_callback(message="Completed")

        return {
            "success": True,
            "total": total,
            "completed": completed_count,
            "failed": failed_count,
            "papers": results_list,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        # Create structured error with categorization and screenshots
        structured_error = create_structured_error(
            error=e,
            since_timestamp=start_timestamp,
            include_screenshots=True,
        )

        return {
            "success": False,
            "error": str(e),
            "structured_error": structured_error.to_dict(),
            "total": total,
            "completed": completed_count,
            "failed": failed_count,
            "timestamp": datetime.now().isoformat(),
        }


async def fetch_bibtex_executor(
    bibtex_path: str,
    project: str | None = None,
    output_path: str | None = None,
    workers: int = 4,
    browser_mode: str = "stealth",
    chrome_profile: str = "system",
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Execute paper fetch from BibTeX file.

    Args:
        bibtex_path: Path to BibTeX file
        project: Project name
        output_path: Output path for enriched BibTeX
        workers: Number of parallel workers
        browser_mode: Browser mode
        chrome_profile: Chrome profile
        progress_callback: Callback for progress updates

    Returns
    -------
        Result dictionary with stats
    """
    from pathlib import Path

    from scitex.scholar.pipelines import ScholarPipelineBibTeX

    # Record start time for screenshot collection
    start_timestamp = datetime.now().isoformat()

    bibtex_file = Path(bibtex_path)

    if progress_callback:
        progress_callback(
            total=0, completed=0, message=f"Loading {bibtex_file.name}..."
        )

    pipeline = ScholarPipelineBibTeX(
        num_workers=workers,
        browser_mode=browser_mode,
        base_chrome_profile=chrome_profile,
    )

    try:
        results = await pipeline.process_bibtex_file_async(
            bibtex_path=bibtex_file,
            project=project,
            output_bibtex_path=output_path,
        )

        if progress_callback:
            progress_callback(
                total=len(results),
                completed=len(results),
                message="Completed",
            )

        return {
            "success": True,
            "bibtex_path": str(bibtex_file),
            "output_path": output_path,
            "total": len(results),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        # Create structured error with categorization and screenshots
        structured_error = create_structured_error(
            error=e,
            since_timestamp=start_timestamp,
            include_screenshots=True,
        )

        return {
            "success": False,
            "error": str(e),
            "structured_error": structured_error.to_dict(),
            "bibtex_path": str(bibtex_file),
            "timestamp": datetime.now().isoformat(),
        }


async def enrich_bibtex_executor(
    bibtex_path: str,
    output_path: str | None = None,
    add_abstracts: bool = True,
    add_citations: bool = True,
    add_impact_factors: bool = True,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Execute BibTeX enrichment.

    Args:
        bibtex_path: Path to BibTeX file
        output_path: Output path for enriched BibTeX
        add_abstracts: Add missing abstracts
        add_citations: Add citation counts
        add_impact_factors: Add impact factors
        progress_callback: Callback for progress updates

    Returns
    -------
        Result dictionary with enrichment stats
    """
    from scitex.scholar import Scholar

    # Record start time for screenshot collection
    start_timestamp = datetime.now().isoformat()

    scholar = Scholar()

    if progress_callback:
        progress_callback(message="Loading BibTeX...")

    try:
        papers = scholar.load_bibtex(bibtex_path)
        total = len(papers)

        if progress_callback:
            progress_callback(total=total, completed=0, message="Enriching metadata...")

        enriched = scholar.enrich_papers(papers)

        # Save output
        out_path = output_path or bibtex_path.replace(".bib", "-enriched.bib")
        scholar.save_papers_as_bibtex(enriched, out_path)

        if progress_callback:
            progress_callback(completed=total, message="Completed")

        return {
            "success": True,
            "input_path": bibtex_path,
            "output_path": out_path,
            "total": total,
            "enriched": len(enriched),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        # Create structured error with categorization and screenshots
        structured_error = create_structured_error(
            error=e,
            since_timestamp=start_timestamp,
            include_screenshots=True,
        )

        return {
            "success": False,
            "error": str(e),
            "structured_error": structured_error.to_dict(),
            "bibtex_path": bibtex_path,
            "timestamp": datetime.now().isoformat(),
        }


# Executor mapping for JobManager
EXECUTORS = {
    "fetch": fetch_single_executor,
    "fetch_multiple": fetch_multiple_executor,
    "fetch_bibtex": fetch_bibtex_executor,
    "enrich": enrich_bibtex_executor,
}


def get_executor(job_type: str):
    """Get executor function for a job type."""
    return EXECUTORS.get(job_type)


# EOF
