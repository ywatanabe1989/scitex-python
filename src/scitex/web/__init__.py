#!/usr/bin/env python3
"""Web-related utilities module for scitex."""

from ._scraping import get_image_urls, get_urls
from ._search_pubmed import (
    _fetch_details,
    _get_citation,
    _parse_abstract_xml,
    _search_pubmed,
    get_crossref_metrics,
    search_pubmed,
)
from ._search_pubmed import (
    batch__fetch_details as _batch__fetch_details,
)
from ._search_pubmed import (
    fetch_async as _fetch_async,
)
from ._search_pubmed import (
    format_bibtex as _format_bibtex,
)
from ._search_pubmed import (
    parse_args as _parse_args,
)
from ._search_pubmed import (
    run_main as _run_main,
)
from ._search_pubmed import (
    save_bibtex as _save_bibtex,
)
from ._summarize_url import (
    crawl_to_json,
    crawl_url,
    summarize_url,
)
from ._summarize_url import (
    extract_main_content as _extract_main_content,
)
from ._summarize_url import (
    summarize_all as _summarize_all,
)
from .download_images import download_images

__all__ = [
    # Public API
    "search_pubmed",
    "get_crossref_metrics",
    "summarize_url",
    "crawl_url",
    "crawl_to_json",
    "get_urls",
    "download_images",
    "get_image_urls",
]
