#!/usr/bin/env python3
"""Test Scholar PubMed search specifically."""

import os
import sys

sys.path.insert(0, "src")

# Set environment variables
os.environ["SCITEX_ENTREZ_EMAIL"] = "ywata1989@gmail.com"

import pytest

pytest.importorskip("aiohttp")

from scitex import logging
from scitex.scholar import Scholar

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)


def test_scholar_pubmed():
    """Test Scholar PubMed search with debugging."""
    print("Testing Scholar PubMed search with debugging...")

    # Initialize Scholar
    scholar = Scholar(email="ywata1989@gmail.com", enrich_by_default=False)

    # Test different queries
    queries = ["machine learning", "deep learning neuroscience", "cancer immunotherapy"]

    for query in queries:
        print(f"\n\nSearching PubMed for: '{query}'")

        try:
            papers = scholar.search(
                query, limit=3, sources=["pubmed"], show_progress=True
            )

            print(f"Found {len(papers)} papers")

            if len(papers) > 0:
                for i, paper in enumerate(papers[:2], 1):
                    print(f"\n{i}. {paper.title}")
                    print(f"   PMID: {paper.pmid}")
                    print(f"   Year: {paper.year}")
            else:
                print("No papers found - checking search details...")

                # Try all sources to compare
                all_papers = scholar.search(
                    query, limit=3, sources="all", show_progress=False
                )
                print(f"  All sources returned: {len(all_papers)} papers")

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_scholar_pubmed()
