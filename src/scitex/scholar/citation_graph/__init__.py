"""
Citation Graph Module

Build and analyze citation networks for academic papers using CrossRef data.

This module provides tools to:
- Extract citation relationships
- Calculate paper similarity (co-citation, bibliographic coupling)
- Build citation network graphs
- Export for visualization (D3.js, vis.js, Cytoscape)

Example:
    >>> from scitex.scholar.citation_graph import CitationGraphBuilder
    >>>
    >>> builder = CitationGraphBuilder(db_path="/path/to/crossref.db")
    >>> graph = builder.build("10.1038/s41586-020-2008-3", top_n=20)
    >>> builder.export_json(graph, "network.json")
"""

from .builder import CitationGraphBuilder
from .models import PaperNode, CitationEdge, CitationGraph

__version__ = "0.1.0"
__all__ = [
    "CitationGraphBuilder",
    "PaperNode",
    "CitationEdge",
    "CitationGraph",
]
