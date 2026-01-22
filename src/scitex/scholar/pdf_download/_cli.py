#!/usr/bin/env python3
# Timestamp: "2026-01-22 (ywatanabe)"
# File: src/scitex/scholar/pdf_download/_cli.py
"""CLI entry point for ScholarPDFDownloader."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scitex import logging

logger = logging.getLogger(__name__)

__FILE__ = __file__


async def main_async(args):
    """Example usage showing decoupled URL resolution and downloading."""
    from scitex.scholar import (
        ScholarAuthManager,
        ScholarBrowserManager,
        ScholarURLFinder,
    )
    from scitex.scholar.auth import AuthenticationGateway
    from scitex.scholar.pdf_download import ScholarPDFDownloader

    # Authenticated Browser and Context
    auth_manager = ScholarAuthManager()
    browser_manager = ScholarBrowserManager(
        chrome_profile_name="system",
        browser_mode=args.browser_mode,
        auth_manager=auth_manager,
        use_zenrows_proxy=False,
    )
    (
        browser,
        context,
    ) = await browser_manager.get_authenticated_browser_and_context_async()

    # Authentication Gateway
    auth_gateway = AuthenticationGateway(
        auth_manager=auth_manager,
        browser_manager=browser_manager,
    )
    url_context = await auth_gateway.prepare_context_async(
        doi=args.doi, context=context
    )

    # URL Resolution
    url_finder = ScholarURLFinder(context)
    resolved_url = url_context.url if url_context else None
    if resolved_url:
        logger.info(f"Using resolved URL from auth_gateway: {resolved_url}")
        urls = await url_finder.find_pdf_urls(resolved_url)
    else:
        logger.info(f"No resolved URL, using DOI: {args.doi}")
        urls = await url_finder.find_pdf_urls(args.doi)

    # Extract URL strings from list of dicts
    pdf_urls = []
    for entry in urls:
        if isinstance(entry, dict):
            pdf_urls.append(entry.get("url"))
        elif isinstance(entry, str):
            pdf_urls.append(entry)

    if not pdf_urls:
        logger.error(f"No PDF URLs found for DOI: {args.doi}")
        return

    logger.info(f"Found {len(pdf_urls)} PDF URL(s) for DOI: {args.doi}")

    # PDF Download
    pdf_downloader = ScholarPDFDownloader(context)
    if len(pdf_urls) == 1:
        await pdf_downloader.download_from_url(pdf_urls[0], args.output)
    else:
        output_dir = Path(args.output).parent
        await pdf_downloader.download_from_urls(
            pdf_urls, output_dir=output_dir, max_concurrent=3
        )


def main(args):
    asyncio.run(main_async(args))
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a PDF using DOI with authentication support"
    )
    parser.add_argument(
        "--doi",
        type=str,
        required=True,
        help="DOI of the paper (e.g., 10.1088/1741-2552/aaf92e)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="~/.scitex/scholar/library/downloads/downloaded_paper.pdf",
        help="Output path for the PDF",
    )
    parser.add_argument(
        "--browser-mode",
        type=str,
        choices=["stealth", "interactive"],
        default="stealth",
        help="Browser mode (default: stealth)",
    )
    return parser.parse_args()


def run_main() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng

    import sys

    import matplotlib.pyplot as plt

    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


if __name__ == "__main__":
    run_main()

# EOF
