#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/scholar.py
# ----------------------------------------

"""Template for stx.scholar literature management module usage."""

TEMPLATE = {
    "name": "Scholar Module",
    "description": "stx.scholar usage for literature management, BibTeX enrichment, and PDF downloads",
    "filename": "scholar_script.py",
    "priority": 4,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.scholar - Literature Management
===================================

stx.scholar provides:
- BibTeX enrichment (DOIs, abstracts, citations, impact factors)
- PDF downloading with institutional access support
- Paper search across multiple databases
- Project-based paper organization
- CrossRef local database (167M+ papers)

Usage Patterns
--------------
1. CLI: scitex scholar <command> [options]
2. MCP: scholar_* tools for Claude Code integration
3. Python API: stx.scholar.* functions

Typical Workflow
----------------
1. Create project: scitex scholar project create myresearch
2. Add papers from BibTeX: scitex scholar add --bibtex refs.bib --project myresearch
3. Enrich metadata: scitex scholar enrich --project myresearch
4. Download PDFs: scitex scholar fetch --project myresearch
5. Search for more: scitex scholar search "machine learning EEG"
"""

import scitex as stx

# ============================================================
# Pattern 1: CLI Usage (Recommended)
# ============================================================

"""
CLI Commands:
-------------

# Project Management
scitex scholar project create myresearch       # Create new project
scitex scholar project list                    # List all projects

# Add Papers
scitex scholar add --bibtex refs.bib --project myresearch
scitex scholar add --doi "10.1038/nature12373" --project myresearch

# Enrich BibTeX (add DOIs, abstracts, citations)
scitex scholar enrich --bibtex refs.bib --output enriched.bib
scitex scholar enrich --project myresearch

# Download PDFs
scitex scholar fetch --project myresearch
scitex scholar fetch --doi "10.1038/nature12373" --output ./pdfs/

# Search Databases
scitex scholar search "machine learning EEG" --limit 50
scitex scholar search "CRISPR" --source crossref --year-min 2020

# Check Library Status
scitex scholar status --project myresearch
"""

# ============================================================
# Pattern 2: CrossRef Local Search (167M+ papers)
# ============================================================

"""
CrossRef Local Database:
------------------------
Fast full-text search across 167M+ academic papers.
Supports FTS5 query syntax: AND, OR, NOT, "exact phrases"

CLI:
----
scitex scholar crossref search "neural network AND hippocampus"
scitex scholar crossref get 10.1038/nature12373
scitex scholar crossref citations 10.1038/nature12373

MCP Tools:
----------
crossref_search(query, limit)      # Search papers
crossref_get(doi)                  # Get metadata by DOI
crossref_citations(doi, direction) # Get citing/cited papers
"""

# ============================================================
# Pattern 3: Paper Caching for Research Topics
# ============================================================

"""
Paper Cache Workflow:
---------------------
Build reusable collections for specific research topics.

# Create cache from search query
scitex scholar cache create epilepsy "epilepsy seizure prediction" --limit 1000

# Query cached papers
scitex scholar cache query epilepsy --year-min 2020 --limit 50

# Get top cited papers
scitex scholar cache top-cited epilepsy --n 20

# Export to formats
scitex scholar cache export epilepsy --format bibtex --output refs.bib
scitex scholar cache export epilepsy --format csv --output papers.csv
"""

# ============================================================
# Pattern 4: Institutional Authentication
# ============================================================

"""
Institutional Access (OpenAthens/Shibboleth):
---------------------------------------------
For downloading papers requiring institutional credentials.

# Check authentication status
scitex scholar auth status

# Login via OpenAthens
scitex scholar auth login --method openathens

# Logout and clear session
scitex scholar auth logout
"""

# ============================================================
# Pattern 5: Python API Usage
# ============================================================

def python_api_example():
    """Direct Python API usage."""
    from scitex.scholar import (
        search_papers,
        enrich_bibtex,
        download_pdf,
        parse_bibtex,
    )

    # Search for papers
    results = search_papers(
        query="machine learning EEG",
        search_mode="crossref",
        limit=20
    )

    # Parse BibTeX file
    papers = parse_bibtex("references.bib")

    # Enrich with metadata
    enrich_bibtex(
        bibtex_path="references.bib",
        output_path="enriched.bib",
        add_abstracts=True,
        add_citations=True,
        add_impact_factors=True
    )

    # Download a specific PDF
    download_pdf(
        doi="10.1038/nature12373",
        output_dir="./pdfs/"
    )

# ============================================================
# Pattern 6: With @stx.session (Research Workflow)
# ============================================================

@stx.session
def main(
    topic="neural networks",
    n_papers=50,
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Research literature workflow."""
    from scitex.scholar import search_papers, enrich_bibtex

    # 1. Search for papers
    logger.info(f"Searching for papers on: {{topic}}")
    results = search_papers(
        query=topic,
        search_mode="crossref",
        limit=n_papers
    )

    # 2. Save search results
    stx.io.save(results, CONFIG.SDIR_OUT / "search_results.json")

    # 3. If BibTeX exists, enrich it
    bibtex_path = CONFIG.SDIR_OUT / "references.bib"
    if bibtex_path.exists():
        enrich_bibtex(
            bibtex_path=str(bibtex_path),
            output_path=str(CONFIG.SDIR_OUT / "enriched.bib")
        )
        logger.info("BibTeX enriched with metadata")

    logger.info(f"Found {{len(results)}} papers")
    return 0

# ============================================================
# MCP Tools Reference
# ============================================================

"""
MCP Tools for Claude Code:
--------------------------

Search & Discovery:
- scholar_search_papers(query, limit, search_mode)
- scholar_crossref_search(query, limit, year_min, year_max)
- scholar_crossref_get(doi)
- scholar_crossref_citations(doi, direction)

Library Management:
- scholar_create_project(project_name, description)
- scholar_list_projects()
- scholar_add_papers_to_project(project, dois, bibtex_path)
- scholar_get_library_status(project)

Enrichment & Download:
- scholar_enrich_bibtex(bibtex_path, add_abstracts, add_citations)
- scholar_fetch_papers(project, papers, async_mode)
- scholar_download_pdf(doi, output_dir)

BibTeX Operations:
- scholar_parse_bibtex(bibtex_path)
- scholar_resolve_dois(titles, bibtex_path)
- scholar_export_papers(output_path, format)

Authentication:
- scholar_authenticate(method, institution)
- scholar_check_auth_status(method)
- scholar_logout(method)

Cache Operations:
- cache_create(name, query, limit)
- cache_query(name, fields, year_min, year_max)
- cache_top_cited(name, n)
- cache_export(name, output_path, format)
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
