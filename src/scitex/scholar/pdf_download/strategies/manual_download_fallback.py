#!/usr/bin/env python3
# Timestamp: "2025-10-13 08:00:08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/scholar/pdf_download/strategies/manual_download_fallback.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = "./src/scitex/scholar/pdf_download/strategies/manual_download_fallback.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
"""Manual Download Fallback Strategy"""

from pathlib import Path
from typing import Optional

from playwright.async_api import BrowserContext

from scitex import logging
from scitex.scholar import ScholarConfig
from scitex.scholar.browser import browser_logger
from scitex.scholar.pdf_download.strategies.manual_download_utils import (
    DownloadMonitorAndSync,
)

logger = logging.getLogger(__name__)


async def try_download_manual_async(
    context: BrowserContext,
    pdf_url: str,
    output_path: Path,
    func_name: str = "try_download_manual_async",
    config: ScholarConfig = None,
    doi: Optional[str] = None,
) -> Optional[Path]:
    """Manual download fallback strategy.

    Opens PDF URL in browser, shows instructions, and monitors downloads directory.
    When user manually downloads the PDF, it automatically detects and organizes it.

    NOTE: This method should NOT check the _scitex_is_manual_mode flag because
    it IS the manual mode implementation!

    Args:
        context: Browser context
        pdf_url: URL of the PDF to download
        output_path: Where to save the final PDF
        func_name: Name for logging
        config: Scholar configuration
        doi: Optional DOI for filename generation

    Returns
    -------
        Path to downloaded file, or None if failed
    """
    config = config or ScholarConfig()
    page = None

    try:
        # Create new page and navigate to PDF
        page = await context.new_page()

        await browser_logger.info(
            page,
            f"{func_name}: Opening PDF for manual download...",
        )

        await page.goto(pdf_url, timeout=30000, wait_until="domcontentloaded")

        await browser_logger.info(
            page,
            f"{func_name}: Please download the PDF manually from this page",
        )

        # Setup monitoring
        downloads_dir = config.get_library_downloads_dir()
        master_dir = config.get_library_master_dir()
        monitor = DownloadMonitorAndSync(downloads_dir, master_dir)

        # Progress logger
        def log_progress(msg: str):
            logger.info(f"{func_name}: {msg}")

        # Extract DOI from URL if not provided
        if not doi and "doi.org/" in pdf_url:
            doi = pdf_url.split("doi.org/")[-1].split("?")[0].split("#")[0]
        elif not doi and "/doi/" in pdf_url:
            # Try to extract DOI from URL like /doi/10.1212/...
            import re

            match = re.search(r"/doi/(10\.\d+/[^\s?#]+)", pdf_url)
            if match:
                doi = match.group(1)

        # Show instructions and start monitoring
        log_progress(f"Monitoring {downloads_dir} for new PDFs...")
        log_progress("Please download the PDF manually from the browser")

        # Monitor for download (2 minutes timeout to prevent process accumulation)
        temp_file = await monitor.monitor_for_new_download_async(
            timeout_sec=120,  # 2 minutes
            check_interval_sec=1.0,
            logger_func=log_progress,
        )

        if not temp_file:
            await browser_logger.error(
                page,
                f"{func_name}: No new PDF detected in 120 seconds",
            )
            logger.error(f"{func_name}: Download monitoring timeout")
            await page.close()
            return None

        await browser_logger.info(
            page,
            f"{func_name}: Detected: {temp_file.name} ({temp_file.stat().st_size / 1e6:.1f} MB)",
        )

        # Sync to library
        final_path = monitor.sync_to_final_destination(
            temp_file,
            doi=doi,
            url=pdf_url,
            content_type="main",
        )

        await browser_logger.info(
            page,
            f"{func_name}: Synced to library: {final_path.name}",
        )

        # Copy to requested output path
        if final_path and final_path.exists():
            import shutil

            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(final_path), str(output_path))

            await browser_logger.info(
                page,
                f"{func_name}: Manual download complete!",
            )

            logger.info(f"{func_name}: Manual download saved to {output_path}")
            await page.close()
            return output_path

        await page.close()
        return None

    except Exception as e:
        logger.error(f"{func_name}: Manual download failed: {e}")
        if page:
            try:
                await browser_logger.error(
                    page,
                    f"{func_name}: Error: {type(e).__name__}",
                )
                await page.close()
            except Exception:
                pass
        return None


async def handle_manual_download_on_page_async(
    page,
    pdf_url: str,
    output_path: Path,
    func_name: str = "handle_manual_download",
    config: ScholarConfig = None,
    doi: Optional[str] = None,
) -> Optional[Path]:
    """Handle manual download on an already-open page.

    Unlike try_download_manual_async, this uses an existing page
    (e.g., from the stop automation button workflow).

    Args:
        page: Already-open Playwright page
        pdf_url: URL of the PDF
        output_path: Target output path
        config: Scholar configuration
        doi: Optional DOI for metadata

    Returns
    -------
        Path to downloaded file, or None if failed
    """
    config = config or ScholarConfig()
    downloads_dir = config.get_library_downloads_dir()

    # Extract DOI from URL if not provided
    if not doi and "doi.org/" in pdf_url:
        doi = pdf_url.split("doi.org/")[-1].split("?")[0].split("#")[0]

    await browser_logger.info(page, f"{func_name}: Manual download mode activated")
    await browser_logger.info(
        page, f"{func_name}: Please download the PDF manually from this page"
    )

    # Monitor for download
    monitor = DownloadMonitorAndSync(downloads_dir, downloads_dir)

    def log_progress(msg: str):
        logger.info(f"{func_name}: {msg}")

    temp_file = await monitor.monitor_for_new_download_async(
        timeout_sec=120, logger_func=log_progress
    )

    if not temp_file:
        await browser_logger.error(
            page, f"{func_name}: No new PDF detected in downloads directory"
        )
        return None

    await browser_logger.info(
        page,
        f"{func_name}: Detected PDF: {temp_file.name} ({temp_file.stat().st_size / 1e6:.1f} MB)",
    )

    # Save minimal metadata
    if doi:
        import json

        metadata_file = temp_file.parent / f"{temp_file.name}.meta.json"
        metadata = {"doi": doi, "pdf_url": pdf_url, "pdf_file": temp_file.name}
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    await browser_logger.info(
        page, f"{func_name}: Manual download complete - saved in downloads/"
    )

    logger.info(f"{func_name}: PDF: {temp_file}")
    if doi:
        logger.info(f"{func_name}: DOI: {doi} (saved in {temp_file.name}.meta.json)")

    return temp_file


# EOF
