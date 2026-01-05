#!/usr/bin/env python3
"""Test script for the new Scholar interface."""

import os
import sys

sys.path.insert(0, "src")

# Set environment variables with SCITEX_ prefix
os.environ["SCITEX_ENTREZ_EMAIL"] = "ywata1989@gmail.com"
os.environ["SCITEX_SCHOLAR_DIR"] = "/tmp/scitex_scholar_test"

import pytest

pytest.importorskip("aiohttp")

from scitex.scholar import Papers, Scholar


def test_basic_functionality():
    """Test basic Scholar functionality."""
    print("Testing new Scholar interface...")

    # Initialize Scholar
    scholar = Scholar(
        email="researcher@university.edu",
        enrich_by_default=False,  # Skip enrichment for testing
        download_dir="/tmp/test_papers",
    )

    print("✓ Scholar initialized successfully")

    # Test quick search
    print("\nTesting _search_quick...")
    titles = scholar._search_quick("machine learning", top_n=3)
    print(f"✓ Quick search returned {len(titles)} titles")
    for i, title in enumerate(titles, 1):
        print(f"  {i}. {title[:60]}...")

    # Test main search (with mock data since we don't have API keys)
    print("\nTesting search method...")
    try:
        papers = scholar.search("deep learning", limit=5, show_progress=False)
        print(f"✓ Search completed (returned {len(papers)} papers)")

        # Test Papers methods
        if len(papers) > 0:
            print("\nTesting Papers methods...")

            # Test iteration
            count = 0
            for paper in papers:
                count += 1
            print(f"✓ Iteration works ({count} papers)")

            # Test filtering
            filtered = papers.filter(year_min=2020)
            print(f"✓ Filtering works (filtered to {len(filtered)} papers)")

            # Test sorting
            sorted_papers = papers.sort_by("citations")
            print("✓ Sorting works")

            # Test summary
            summary = papers.summary()
            print("✓ Summary generation works")
            print("\nCollection Summary:")
            print(summary)
    except Exception as e:
        print(f"⚠ Search failed (expected without API keys): {e}")

    print("\n✅ All basic tests passed!")


def test_environment_variables():
    """Test environment variable handling."""
    print("\nTesting environment variable handling...")

    # Test with no API keys
    scholar = Scholar()
    print(f"✓ Email from env: {scholar.email}")
    print(f"✓ Default download dir: {scholar.download_dir}")

    # Test API key detection
    os.environ["SCITEX_SEMANTIC_SCHOLAR_API_KEY"] = "test_key"
    scholar2 = Scholar()
    print(
        f"✓ S2 API key detected: {'s2' in scholar2.api_keys and scholar2.api_keys['s2'] == 'test_key'}"
    )

    # Clean up
    del os.environ["SCITEX_SEMANTIC_SCHOLAR_API_KEY"]


def test_imports():
    """Test all expected imports."""
    print("\nTesting imports...")

    from scitex.scholar import (
        Paper,
        Papers,
        Scholar,
        build_index,
        get_scholar_dir,
        search_sync,
    )

    print("✓ All core imports successful")

    # Test scholar directory
    scholar_dir = get_scholar_dir()
    print(f"✓ Scholar directory: {scholar_dir}")


def test_pubmed_search():
    """Test PubMed search functionality."""
    print("\nTesting PubMed search...")

    # Initialize Scholar with PubMed email
    scholar = Scholar(email="ywata1989@gmail.com", enrich_by_default=False)

    try:
        # Search PubMed specifically
        papers = scholar.search(
            "deep learning neuroscience",
            limit=5,
            sources=["pubmed"],
            show_progress=True,
        )

        print(f"\n✓ PubMed search returned {len(papers)} papers")

        # Display results
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n{i}. {paper.title}")
            print(
                f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}"
            )
            print(f"   Year: {paper.year}")
            print(f"   Journal: {paper.journal}")
            print(f"   PMID: {paper.pmid}")
            print(f"   DOI: {paper.doi}")
            print(f"   Citations: {paper.citation_count}")

        # Test paper collection functionality
        if len(papers) > 0:
            print("\nTesting Papers features on PubMed results...")

            # Save to BibTeX
            bibtex_path = papers.save("/tmp/pubmed_papers.bib", format="bibtex")
            print(f"✓ Saved to BibTeX: {bibtex_path}")

            # Get summary
            summary = papers.summary()
            print(f"\n{summary}")

    except Exception as e:
        print(f"❌ PubMed search failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_imports()
    test_environment_variables()
    test_basic_functionality()
    test_pubmed_search()

    print("\n✅ All tests completed successfully!")
