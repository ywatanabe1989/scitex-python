"""
Data models for citation graphs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PaperNode:
    """Represents a paper in the citation network."""

    doi: str
    title: str = ""
    year: int = 0
    authors: List[str] = field(default_factory=list)
    journal: str = ""
    citation_count: int = 0
    similarity_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "id": self.doi,
            "title": self.title,
            "year": self.year,
            "authors": self.authors,
            "journal": self.journal,
            "citation_count": self.citation_count,
            "similarity_score": self.similarity_score,
        }


@dataclass
class CitationEdge:
    """Represents a citation relationship between papers."""

    source: str  # DOI of citing paper
    target: str  # DOI of cited paper
    edge_type: str = "cites"  # 'cites', 'cited_by', 'similar'
    weight: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "weight": self.weight,
        }


@dataclass
class CitationGraph:
    """Represents a complete citation network."""

    seed_doi: str
    nodes: List[PaperNode] = field(default_factory=list)
    edges: List[CitationEdge] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON export."""
        return {
            "seed": self.seed_doi,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": self.metadata,
        }

    @property
    def node_count(self) -> int:
        """Number of nodes in graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of edges in graph."""
        return len(self.edges)
