#!/usr/bin/env python3
"""Debug PubMed search functionality."""

import os
import sys

sys.path.insert(0, "src")

# Set environment variables
os.environ["SCITEX_ENTREZ_EMAIL"] = "ywata1989@gmail.com"

import asyncio

import pytest

pytest.importorskip("aiohttp")

from scitex.scholar._paper_acquisition import PaperAcquisition


async def test_pubmed_direct():
    """Test PubMed search directly through PaperAcquisition."""
    print("Testing direct PubMed search through PaperAcquisition...")

    acquisition = PaperAcquisition(email="ywata1989@gmail.com")

    try:
        # Search PubMed directly
        papers = await acquisition.search(
            query="deep learning", sources=["pubmed"], max_results=5
        )

        print(f"\nFound {len(papers)} papers from PubMed")

        for i, paper in enumerate(papers[:3], 1):
            print(f"\n{i}. {paper.title}")
            print(
                f"   Authors: {', '.join(paper.authors[:3]) if paper.authors else 'N/A'}"
            )
            print(f"   Year: {paper.year}")
            print(f"   Journal: {paper.journal}")
            print(f"   PMID: {paper.pmid}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


async def test_pubmed_engine():
    """Test PubMed engine directly."""
    from scitex.scholar._pubmed_engine import PubMedEngine

    print("\n\nTesting PubMed engine directly...")

    engine = PubMedEngine(email="ywata1989@gmail.com")

    try:
        papers = await engine.search("machine learning cancer", max_results=3)

        print(f"\nFound {len(papers)} papers")

        for i, paper in enumerate(papers, 1):
            print(f"\n{i}. {paper.title}")
            print(f"   PMID: {paper.pmid}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run async tests
    asyncio.run(test_pubmed_direct())
    asyncio.run(test_pubmed_engine())
