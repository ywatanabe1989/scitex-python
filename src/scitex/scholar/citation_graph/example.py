#!/usr/bin/env python3
"""
Example usage of the citation_graph module.

Run this from the scitex-code root:
    python -m scitex.scholar.citation_graph.example
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scitex.scholar.citation_graph import CitationGraphBuilder


def main():
    # Database path (adjust to your setup)
    db_path = Path.home() / "proj/crossref_local/data/crossref.db"

    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("Please update the db_path in this script.")
        return 1

    print("=" * 70)
    print("  Citation Graph Example")
    print("=" * 70)
    print(f"\nDatabase: {db_path}")

    # Initialize builder
    builder = CitationGraphBuilder(str(db_path))

    # Example DOI (a well-cited paper)
    seed_doi = "10.1001/2013.jamapsychiatry.4"

    # Get paper summary
    print(f"\n1. Getting paper summary for {seed_doi}...")
    summary = builder.get_paper_summary(seed_doi)

    if summary:
        print(f"\nPaper: {summary['title']}")
        print(f"Authors: {', '.join(summary['authors'][:3])}")
        print(f"Year: {summary['year']}")
        print(f"Journal: {summary['journal']}")
        print(f"References: {summary['reference_count']}")
        print(f"Citations: {summary['citation_count']}")
    else:
        print("Paper not found in database")
        return 1

    # Build citation network
    print(f"\n2. Building citation network (top 20 papers)...")
    graph = builder.build(seed_doi, top_n=20)

    print(f"\nNetwork built:")
    print(f"  Nodes: {graph.node_count}")
    print(f"  Edges: {graph.edge_count}")

    # Show top papers by similarity
    print(f"\nTop 10 most similar papers:")
    print(f"{'Rank':<5} {'Score':<7} {'Year':<6} {'Title':<60}")
    print("-" * 85)

    sorted_nodes = sorted(graph.nodes, key=lambda n: n.similarity_score, reverse=True)

    for i, node in enumerate(sorted_nodes[:11], 1):
        if node.doi.lower() == seed_doi.lower():
            continue
        print(
            f"{i:<5} {node.similarity_score:<7.1f} {node.year:<6} {node.title[:60]:<60}"
        )

    # Export to JSON
    output_path = Path(__file__).parent / "example_output.json"
    builder.export_json(graph, str(output_path))
    print(f"\n3. Network exported to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n✅ Example complete!")
    print("\nNext steps:")
    print("  - Open example_output.json to see the graph data")
    print("  - Use this JSON with D3.js, vis.js, or Cytoscape for visualization")
    print("  - Integrate with scitex-cloud for API endpoints")

    return 0


if __name__ == "__main__":
    sys.exit(main())
