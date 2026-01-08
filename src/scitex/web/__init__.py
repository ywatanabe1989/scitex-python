#!/usr/bin/env python3
"""Web-related utilities module for scitex."""

from ._scraping import get_image_urls, get_urls
from ._search_pubmed import (
    _fetch_details,
    _get_citation,
    _parse_abstract_xml,
    _search_pubmed,
    batch__fetch_details,
    fetch_async,
    format_bibtex,
    get_crossref_metrics,
    parse_args,
    run_main,
    save_bibtex,
    search_pubmed,
)
from ._summarize_url import (
    crawl_to_json,
    crawl_url,
    extract_main_content,
    summarize_all,
    summarize_url,
)
from .download_images import download_images

__all__ = [
    "search_pubmed",
    "_search_pubmed",
    "_fetch_details",
    "_parse_abstract_xml",
    "_get_citation",
    "get_crossref_metrics",
    "save_bibtex",
    "format_bibtex",
    "fetch_async",
    "batch__fetch_details",
    "parse_args",
    "run_main",
    "summarize_url",
    "extract_main_content",
    "crawl_url",
    "crawl_to_json",
    "summarize_all",
    "get_urls",
    "download_images",
    "get_image_urls",
]
