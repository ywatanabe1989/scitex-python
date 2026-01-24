#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: src/scitex/cli/scholar/_crossref_scitex.py
"""CrossRef-SciTeX CLI commands for scitex scholar.

Provides access to the local CrossRef database (167M+ papers) via crossref-local.
Branded as crossref-scitex to distinguish from official CrossRef API.
"""

from __future__ import annotations

import json
import sys

import click


@click.group("crossref-scitex")
def crossref_scitex():
    """
    CrossRef-SciTeX database search (167M+ papers)

    \b
    Search and query the local CrossRef database via crossref-local.
    Supports both direct DB access and HTTP API mode.

    \b
    Examples:
        scitex scholar crossref-scitex search "deep learning"
        scitex scholar crossref-scitex get 10.1038/nature12373
        scitex scholar crossref-scitex count "epilepsy seizure"
        scitex scholar crossref-scitex info
    """
    pass


@crossref_scitex.command("search")
@click.argument("query")
@click.option("-n", "--limit", default=20, help="Maximum results (default: 20)")
@click.option("--offset", default=0, help="Skip N results for pagination")
@click.option("--year-min", type=int, help="Minimum publication year")
@click.option("--year-max", type=int, help="Maximum publication year")
@click.option("--enrich", is_flag=True, help="Add citation counts and references")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def search_cmd(query, limit, offset, year_min, year_max, enrich, as_json):
    """
    Search for papers in CrossRef database.

    \b
    Examples:
        scitex scholar crossref-scitex search "hippocampal sharp wave ripples"
        scitex scholar crossref search "deep learning" --limit 50
        scitex scholar crossref search "CRISPR" --year-min 2020 --enrich
    """
    try:
        from scitex.scholar import crossref_scitex as crossref
    except ImportError:
        click.secho(
            "crossref-local not installed. Install with: pip install crossref-local",
            fg="red",
        )
        sys.exit(1)

    try:
        results = crossref.search(query, limit=limit, offset=offset)

        if enrich:
            results = crossref.enrich(results)

        papers = []
        for work in results:
            if year_min and work.year and work.year < year_min:
                continue
            if year_max and work.year and work.year > year_max:
                continue
            papers.append(work)
            if len(papers) >= limit:
                break

        if as_json:
            output = {
                "query": query,
                "total": results.total,
                "count": len(papers),
                "papers": [
                    {
                        "doi": p.doi,
                        "title": p.title,
                        "authors": p.authors,
                        "year": p.year,
                        "journal": p.journal,
                        "citation_count": p.citation_count,
                    }
                    for p in papers
                ],
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.secho(
                f"Found {results.total} papers for: {query}", fg="green", bold=True
            )
            click.echo()

            for i, paper in enumerate(papers, 1):
                authors = ", ".join(paper.authors[:3]) if paper.authors else "Unknown"
                if paper.authors and len(paper.authors) > 3:
                    authors += " et al."

                click.secho(f"{i}. {paper.title}", fg="cyan", bold=True)
                click.echo(f"   Authors: {authors}")
                click.echo(
                    f"   Year: {paper.year or 'N/A'} | Journal: {paper.journal or 'N/A'}"
                )
                click.echo(f"   DOI: {paper.doi}")
                if paper.citation_count:
                    click.echo(f"   Citations: {paper.citation_count}")
                click.echo()

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@crossref_scitex.command("get")
@click.argument("doi")
@click.option("--citations", is_flag=True, help="Include citing papers")
@click.option("--references", is_flag=True, help="Include referenced papers")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def get_cmd(doi, citations, references, as_json):
    """
    Get paper details by DOI.

    \b
    Examples:
        scitex scholar crossref get 10.1038/nature12373
        scitex scholar crossref get 10.1126/science.aax0758 --citations
    """
    try:
        from scitex.scholar import crossref_scitex as crossref
    except ImportError:
        click.secho(
            "crossref-local not installed. Install with: pip install crossref-local",
            fg="red",
        )
        sys.exit(1)

    try:
        work = crossref.get(doi)

        if work is None:
            click.secho(f"DOI not found: {doi}", fg="red")
            sys.exit(1)

        if as_json:
            output = {
                "doi": work.doi,
                "title": work.title,
                "authors": work.authors,
                "year": work.year,
                "journal": work.journal,
                "abstract": work.abstract,
                "citation_count": work.citation_count,
                "reference_count": work.reference_count,
                "type": work.type,
                "publisher": work.publisher,
            }
            if citations:
                output["citing_dois"] = crossref.get_citing(doi)
            if references:
                output["referenced_dois"] = crossref.get_cited(doi)
            click.echo(json.dumps(output, indent=2))
        else:
            click.secho(work.title, fg="cyan", bold=True)
            click.echo()

            if work.authors:
                click.echo(f"Authors: {', '.join(work.authors)}")
            click.echo(f"Year: {work.year or 'N/A'}")
            click.echo(f"Journal: {work.journal or 'N/A'}")
            click.echo(f"DOI: {work.doi}")
            click.echo(f"Type: {work.type or 'N/A'}")
            click.echo(f"Publisher: {work.publisher or 'N/A'}")
            click.echo(f"Citations: {work.citation_count or 0}")
            click.echo(f"References: {work.reference_count or 0}")

            if work.abstract:
                click.echo()
                click.secho("Abstract:", bold=True)
                click.echo(
                    work.abstract[:500] + "..."
                    if len(work.abstract) > 500
                    else work.abstract
                )

            if citations:
                citing = crossref.get_citing(doi)
                click.echo()
                click.secho(f"Citing papers ({len(citing)}):", bold=True)
                for c_doi in citing[:10]:
                    click.echo(f"  - {c_doi}")
                if len(citing) > 10:
                    click.echo(f"  ... and {len(citing) - 10} more")

            if references:
                cited = crossref.get_cited(doi)
                click.echo()
                click.secho(f"References ({len(cited)}):", bold=True)
                for r_doi in cited[:10]:
                    click.echo(f"  - {r_doi}")
                if len(cited) > 10:
                    click.echo(f"  ... and {len(cited) - 10} more")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@crossref_scitex.command("count")
@click.argument("query")
def count_cmd(query):
    """
    Count papers matching a query.

    \b
    Examples:
        scitex scholar crossref count "machine learning"
        scitex scholar crossref count "CRISPR gene editing"
    """
    try:
        from scitex.scholar import crossref_scitex as crossref
    except ImportError:
        click.secho(
            "crossref-local not installed. Install with: pip install crossref-local",
            fg="red",
        )
        sys.exit(1)

    try:
        count = crossref.count(query)
        click.echo(f"{count:,} papers match: {query}")
    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


@crossref_scitex.command("info")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def info_cmd(as_json):
    """
    Show CrossRef database configuration and status.

    \b
    Examples:
        scitex scholar crossref info
        scitex scholar crossref info --json
    """
    try:
        from scitex.scholar import crossref_scitex as crossref
    except ImportError:
        click.secho(
            "crossref-local not installed. Install with: pip install crossref-local",
            fg="red",
        )
        sys.exit(1)

    try:
        info = crossref.info()
        mode = crossref.get_mode()

        if as_json:
            info["mode"] = mode
            click.echo(json.dumps(info, indent=2))
        else:
            click.secho("CrossRef Database Status", fg="cyan", bold=True)
            click.echo()
            click.echo(f"Mode: {mode}")
            click.echo(f"Status: {info.get('status', 'unknown')}")

            if "version" in info:
                click.echo(f"Version: {info['version']}")

            if "work_count" in info:
                click.echo(f"Papers: {info['work_count']:,}")

            if "db_path" in info:
                click.echo(f"Database: {info['db_path']}")

            if "api_url" in info:
                click.echo(f"API URL: {info['api_url']}")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red")
        sys.exit(1)


# EOF
