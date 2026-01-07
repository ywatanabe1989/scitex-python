#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/scholar/_mcp_handlers.py
# ----------------------------------------

"""Handler implementations for the scitex-scholar MCP server."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

__all__ = [
    "search_papers_handler",
    "resolve_dois_handler",
    "enrich_bibtex_handler",
    "download_pdf_handler",
    "download_pdfs_batch_handler",
    "get_library_status_handler",
    "parse_bibtex_handler",
    "validate_pdfs_handler",
    "resolve_openurls_handler",
    "authenticate_handler",
    "export_papers_handler",
]


def _get_scholar_dir() -> Path:
    """Get the scholar data directory."""
    base_dir = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
    scholar_dir = base_dir / "scholar"
    scholar_dir.mkdir(parents=True, exist_ok=True)
    return scholar_dir


def _ensure_scholar():
    """Ensure Scholar module is available and return instance."""
    try:
        from scitex.scholar import Scholar

        return Scholar()
    except ImportError as e:
        raise RuntimeError(f"Scholar module not available: {e}")


async def search_papers_handler(
    query: str,
    sources: list[str] | None = None,
    limit: int = 20,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict:
    """Search for scientific papers across multiple databases."""
    try:
        from scitex.scholar import Scholar

        loop = asyncio.get_event_loop()
        scholar = Scholar()

        def do_search():
            kwargs = {"limit": limit}
            if sources:
                kwargs["sources"] = sources
            if year_min:
                kwargs["year_min"] = year_min
            if year_max:
                kwargs["year_max"] = year_max

            papers = scholar.search(query, **kwargs)
            return papers

        papers = await loop.run_in_executor(None, do_search)

        results = []
        for paper in papers:
            results.append(
                {
                    "title": paper.title,
                    "authors": paper.authors[:5] if paper.authors else [],
                    "year": paper.year,
                    "doi": paper.doi,
                    "journal": paper.journal,
                    "abstract": (
                        paper.abstract[:300] + "..."
                        if paper.abstract and len(paper.abstract) > 300
                        else paper.abstract
                    ),
                    "citation_count": paper.citation_count,
                    "impact_factor": paper.impact_factor,
                }
            )

        return {
            "success": True,
            "count": len(results),
            "query": query,
            "papers": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def resolve_dois_handler(
    bibtex_path: str | None = None,
    titles: list[str] | None = None,
    resume: bool = True,
    project: str | None = None,
) -> dict:
    """Resolve DOIs from paper titles."""
    try:
        from scitex.scholar import Scholar

        loop = asyncio.get_event_loop()
        scholar = Scholar(project=project) if project else Scholar()

        def do_resolve():
            if bibtex_path:
                # Load papers from BibTeX and resolve DOIs
                papers = scholar.from_bibtex(bibtex_path)
                resolved = []
                failed = []

                for paper in papers:
                    if not paper.doi:
                        # Try to resolve DOI from title
                        try:
                            doi = scholar.resolve_doi(
                                paper.title, paper.authors, paper.year
                            )
                            if doi:
                                paper.doi = doi
                                resolved.append({"title": paper.title, "doi": doi})
                            else:
                                failed.append(
                                    {"title": paper.title, "reason": "No DOI found"}
                                )
                        except Exception as e:
                            failed.append({"title": paper.title, "reason": str(e)})
                    else:
                        resolved.append({"title": paper.title, "doi": paper.doi})

                return {"resolved": resolved, "failed": failed, "total": len(papers)}

            elif titles:
                resolved = []
                failed = []

                for title in titles:
                    try:
                        doi = scholar.resolve_doi(title)
                        if doi:
                            resolved.append({"title": title, "doi": doi})
                        else:
                            failed.append({"title": title, "reason": "No DOI found"})
                    except Exception as e:
                        failed.append({"title": title, "reason": str(e)})

                return {"resolved": resolved, "failed": failed, "total": len(titles)}

            else:
                return {"error": "Either bibtex_path or titles required"}

        result = await loop.run_in_executor(None, do_resolve)

        return {
            "success": True,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def enrich_bibtex_handler(
    bibtex_path: str,
    output_path: str | None = None,
    add_abstracts: bool = True,
    add_citations: bool = True,
    add_impact_factors: bool = True,
) -> dict:
    """Enrich BibTeX entries with metadata."""
    try:
        from scitex.scholar import Scholar

        loop = asyncio.get_event_loop()
        scholar = Scholar()

        def do_enrich():
            papers = scholar.from_bibtex(bibtex_path)
            enriched_count = 0

            for paper in papers:
                enriched = False

                if add_abstracts and not paper.abstract:
                    # Try to fetch abstract
                    pass  # Will be handled by scholar.enrich()

                if add_citations and not paper.citation_count:
                    pass

                if add_impact_factors and not paper.impact_factor:
                    pass

                if enriched:
                    enriched_count += 1

            # Use scholar's enrich method
            papers = scholar.enrich(papers)

            # Save to output
            out_path = output_path or bibtex_path.replace(".bib", "-enriched.bib")
            papers.save(out_path)

            summary = {
                "total": len(papers),
                "with_doi": sum(1 for p in papers if p.doi),
                "with_abstract": sum(1 for p in papers if p.abstract),
                "with_citations": sum(1 for p in papers if p.citation_count),
                "with_impact_factor": sum(1 for p in papers if p.impact_factor),
            }

            return {"output_path": out_path, "summary": summary}

        result = await loop.run_in_executor(None, do_enrich)

        return {
            "success": True,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def download_pdf_handler(
    doi: str,
    output_dir: str = "./pdfs",
    auth_method: str = "none",
    use_browser: bool = False,
) -> dict:
    """Download a single PDF."""
    try:
        from scitex.scholar import Scholar
        from scitex.scholar.core import Paper

        loop = asyncio.get_event_loop()
        scholar = Scholar()

        def do_download():
            paper = Paper(doi=doi)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            result = scholar.download_pdf(
                paper,
                output_dir=output_dir,
                use_browser=use_browser,
            )

            return result

        result = await loop.run_in_executor(None, do_download)

        if result:
            return {
                "success": True,
                "doi": doi,
                "path": str(result),
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {
                "success": False,
                "doi": doi,
                "error": "Download failed",
            }

    except Exception as e:
        return {"success": False, "doi": doi, "error": str(e)}


async def download_pdfs_batch_handler(
    dois: list[str] | None = None,
    bibtex_path: str | None = None,
    project: str | None = None,
    output_dir: str | None = None,
    max_concurrent: int = 3,
    resume: bool = True,
) -> dict:
    """Download PDFs for multiple papers."""
    try:
        from scitex.scholar import Scholar
        from scitex.scholar.core import Paper

        loop = asyncio.get_event_loop()
        scholar = Scholar(project=project) if project else Scholar()

        def do_batch_download():
            papers = []

            if bibtex_path:
                papers = scholar.from_bibtex(bibtex_path)
            elif dois:
                papers = [Paper(doi=d) for d in dois]
            else:
                return {"error": "Either dois or bibtex_path required"}

            out_dir = output_dir or str(
                _get_scholar_dir() / "library" / (project or "default") / "pdfs"
            )
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            results = {
                "total": len(papers),
                "downloaded": [],
                "failed": [],
                "skipped": [],
            }

            for paper in papers:
                if not paper.doi:
                    results["skipped"].append(
                        {"title": paper.title, "reason": "No DOI"}
                    )
                    continue

                try:
                    pdf_path = scholar.download_pdf(paper, output_dir=out_dir)
                    if pdf_path:
                        results["downloaded"].append(
                            {
                                "doi": paper.doi,
                                "path": str(pdf_path),
                            }
                        )
                    else:
                        results["failed"].append(
                            {
                                "doi": paper.doi,
                                "reason": "Download returned None",
                            }
                        )
                except Exception as e:
                    results["failed"].append(
                        {
                            "doi": paper.doi,
                            "reason": str(e),
                        }
                    )

            return results

        result = await loop.run_in_executor(None, do_batch_download)

        return {
            "success": True,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_library_status_handler(
    project: str | None = None,
    include_details: bool = False,
) -> dict:
    """Get library status."""
    try:

        library_dir = _get_scholar_dir() / "library"

        if project:
            project_dir = library_dir / project
        else:
            project_dir = library_dir

        if not project_dir.exists():
            return {
                "success": True,
                "exists": False,
                "message": f"Library directory not found: {project_dir}",
            }

        # Count PDFs
        pdf_files = list(project_dir.rglob("*.pdf"))
        metadata_files = list(project_dir.rglob("metadata.json"))

        status = {
            "success": True,
            "exists": True,
            "path": str(project_dir),
            "pdf_count": len(pdf_files),
            "entry_count": len(metadata_files),
            "timestamp": datetime.now().isoformat(),
        }

        if include_details:
            entries = []
            for meta_file in metadata_files[:50]:  # Limit to 50 for performance
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)
                    pdf_exists = any(
                        (meta_file.parent / f).exists()
                        for f in meta_file.parent.glob("*.pdf")
                    )
                    entries.append(
                        {
                            "id": meta_file.parent.name,
                            "title": meta.get("title", "Unknown"),
                            "doi": meta.get("doi"),
                            "has_pdf": pdf_exists,
                        }
                    )
                except Exception:
                    pass

            status["entries"] = entries

        return status

    except Exception as e:
        return {"success": False, "error": str(e)}


async def parse_bibtex_handler(bibtex_path: str) -> dict:
    """Parse a BibTeX file."""
    try:
        from scitex.scholar import Scholar

        loop = asyncio.get_event_loop()
        scholar = Scholar()

        def do_parse():
            papers = scholar.from_bibtex(bibtex_path)
            return papers

        papers = await loop.run_in_executor(None, do_parse)

        results = []
        for paper in papers:
            results.append(
                {
                    "title": paper.title,
                    "authors": paper.authors[:5] if paper.authors else [],
                    "year": paper.year,
                    "doi": paper.doi,
                    "journal": paper.journal,
                    "bibtex_key": getattr(paper, "bibtex_key", None),
                }
            )

        return {
            "success": True,
            "count": len(results),
            "path": bibtex_path,
            "papers": results,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def validate_pdfs_handler(
    project: str | None = None,
    pdf_paths: list[str] | None = None,
) -> dict:
    """Validate PDF files."""
    try:
        from PyPDF2 import PdfReader

        if pdf_paths:
            paths = [Path(p) for p in pdf_paths]
        elif project:
            library_dir = _get_scholar_dir() / "library" / project
            paths = list(library_dir.rglob("*.pdf"))
        else:
            library_dir = _get_scholar_dir() / "library"
            paths = list(library_dir.rglob("*.pdf"))

        results = {
            "total": len(paths),
            "valid": [],
            "invalid": [],
        }

        for pdf_path in paths:
            try:
                reader = PdfReader(str(pdf_path))
                page_count = len(reader.pages)

                # Check if it has text content
                has_text = False
                if page_count > 0:
                    text = reader.pages[0].extract_text()
                    has_text = bool(text and len(text.strip()) > 100)

                results["valid"].append(
                    {
                        "path": str(pdf_path),
                        "pages": page_count,
                        "has_text": has_text,
                        "size_kb": round(pdf_path.stat().st_size / 1024, 2),
                    }
                )
            except Exception as e:
                results["invalid"].append(
                    {
                        "path": str(pdf_path),
                        "error": str(e),
                    }
                )

        return {
            "success": True,
            **results,
            "valid_count": len(results["valid"]),
            "invalid_count": len(results["invalid"]),
            "timestamp": datetime.now().isoformat(),
        }

    except ImportError:
        return {"success": False, "error": "PyPDF2 not installed"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def resolve_openurls_handler(
    dois: list[str],
    resolver_url: str | None = None,
    resume: bool = True,
) -> dict:
    """Resolve OpenURLs for DOIs."""
    try:
        from scitex.scholar import Scholar

        loop = asyncio.get_event_loop()
        scholar = Scholar()

        def do_resolve():
            results = {
                "resolved": [],
                "failed": [],
            }

            for doi in dois:
                try:
                    # Use scholar's OpenURL resolver
                    url = scholar.resolve_openurl(doi, resolver_url=resolver_url)
                    if url:
                        results["resolved"].append({"doi": doi, "url": url})
                    else:
                        results["failed"].append({"doi": doi, "reason": "No URL found"})
                except Exception as e:
                    results["failed"].append({"doi": doi, "reason": str(e)})

            return results

        result = await loop.run_in_executor(None, do_resolve)

        return {
            "success": True,
            **result,
            "total": len(dois),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def authenticate_handler(
    method: str,
    institution: str | None = None,
) -> dict:
    """Authenticate with institutional access."""
    try:
        from scitex.scholar import ScholarAuthManager

        loop = asyncio.get_event_loop()

        def do_auth():
            auth_manager = ScholarAuthManager()

            if method == "openathens":
                success = auth_manager.authenticate_openathens(institution)
            elif method == "shibboleth":
                success = auth_manager.authenticate_shibboleth(institution)
            else:
                return {"error": f"Unknown auth method: {method}"}

            return {"authenticated": success}

        result = await loop.run_in_executor(None, do_auth)

        return {
            "success": True,
            "method": method,
            "institution": institution,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


async def export_papers_handler(
    output_path: str,
    project: str | None = None,
    format: str = "bibtex",
    filter_has_pdf: bool = False,
) -> dict:
    """Export papers to various formats."""
    try:
        from scitex.scholar import Scholar

        loop = asyncio.get_event_loop()
        scholar = Scholar(project=project) if project else Scholar()

        def do_export():
            # Get papers from library
            papers = scholar.get_library_papers(project=project)

            if filter_has_pdf:
                papers = [p for p in papers if hasattr(p, "pdf_path") and p.pdf_path]

            # Export based on format
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "bibtex":
                papers.save(str(out_path))
            elif format == "json":
                with open(out_path, "w") as f:
                    json.dump([p.to_dict() for p in papers], f, indent=2)
            elif format == "csv":
                import csv

                with open(out_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["title", "authors", "year", "doi", "journal"]
                    )
                    writer.writeheader()
                    for p in papers:
                        writer.writerow(
                            {
                                "title": p.title,
                                "authors": (
                                    "; ".join(p.authors[:3]) if p.authors else ""
                                ),
                                "year": p.year,
                                "doi": p.doi,
                                "journal": p.journal,
                            }
                        )
            elif format == "ris":
                papers.save(str(out_path), format="ris")

            return {"count": len(papers), "path": str(out_path)}

        result = await loop.run_in_executor(None, do_export)

        return {
            "success": True,
            "format": format,
            **result,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# EOF
