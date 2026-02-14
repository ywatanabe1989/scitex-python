#!/usr/bin/env python3
# Timestamp: "2026-02-11 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/stats/auto/_decision_tree.py

"""
Decision Tree for Statistical Test Selection.

Provides a flowchart-style decision tree that guides users through
"Which statistical test should I use?" decisions. Reusable across
MCP, CLI, and Django web interface.

The tree mirrors classic statistics decision flowcharts:
    Outcome type? -> # Groups? -> Paired? -> Normal? -> TEST
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class DecisionNode:
    """A node in the statistical test decision tree.

    Parameters
    ----------
    id : str
        Unique identifier (used as SVG element ID for click handling).
    label : str
        Display text in the flowchart.
    shape : str
        Node shape: "diamond" for decisions, "rounded" for tests/leaves.
    emphasis : str
        Visual style: "primary" (decision), "success" (test), "warning" (note).
    children : list of (str, str)
        List of (edge_label, child_id) tuples for outgoing edges.
    test_id : str or None
        If this is a leaf node, the test identifier matching TEST_REGISTRY keys.
    """

    id: str
    label: str
    shape: str = "diamond"
    emphasis: str = "primary"
    children: List[Tuple[str, str]] = field(default_factory=list)
    test_id: Optional[str] = None


# =============================================================================
# Decision Tree Structure
# =============================================================================

DECISION_TREE: Dict[str, DecisionNode] = {
    # Root
    "start": DecisionNode(
        "start",
        "What is your\noutcome type?",
        "diamond",
        "primary",
        [
            ("Continuous /\nOrdinal", "continuous"),
            ("Categorical /\nBinary", "categorical"),
            ("Correlation", "correlation"),
        ],
    ),
    # --- Continuous / Ordinal Branch ---
    "continuous": DecisionNode(
        "continuous",
        "How many\ngroups?",
        "diamond",
        "primary",
        [
            ("2 groups", "two_groups"),
            ("3+ groups", "multi_groups"),
            ("1 sample", "one_sample"),
        ],
    ),
    "one_sample": DecisionNode(
        "one_sample",
        "Normality\ncheck",
        "diamond",
        "primary",
        [("Check", "shapiro")],
    ),
    "shapiro": DecisionNode(
        "shapiro",
        "Shapiro-Wilk",
        "rounded",
        "success",
        [],
        test_id="shapiro",
    ),
    # -- Two groups --
    "two_groups": DecisionNode(
        "two_groups",
        "Independent or\npaired?",
        "diamond",
        "primary",
        [("Independent", "indep_2"), ("Paired", "paired_2")],
    ),
    "indep_2": DecisionNode(
        "indep_2",
        "Normal\ndistribution?",
        "diamond",
        "primary",
        [("Yes", "ttest_ind"), ("No / Unknown", "indep_2_nonparam")],
    ),
    "ttest_ind": DecisionNode(
        "ttest_ind",
        "t-test\n(Independent)",
        "rounded",
        "success",
        [],
        test_id="ttest_ind",
    ),
    "indep_2_nonparam": DecisionNode(
        "indep_2_nonparam",
        "Brunner-Munzel\n(Recommended)",
        "rounded",
        "success",
        [("Alternative", "indep_2_alt")],
        test_id="brunnermunzel",
    ),
    "indep_2_alt": DecisionNode(
        "indep_2_alt",
        "Mann-Whitney U",
        "rounded",
        "normal",
        [],
        test_id="mannwhitneyu",
    ),
    "paired_2": DecisionNode(
        "paired_2",
        "Normal\ndistribution?",
        "diamond",
        "primary",
        [("Yes", "ttest_paired"), ("No / Unknown", "wilcoxon")],
    ),
    "ttest_paired": DecisionNode(
        "ttest_paired",
        "t-test\n(Paired)",
        "rounded",
        "success",
        [],
        test_id="ttest_paired",
    ),
    "wilcoxon": DecisionNode(
        "wilcoxon",
        "Wilcoxon\nSigned-Rank",
        "rounded",
        "success",
        [],
        test_id="wilcoxon",
    ),
    # -- Multiple groups --
    "multi_groups": DecisionNode(
        "multi_groups",
        "Independent or\nrepeated?",
        "diamond",
        "primary",
        [("Independent", "multi_indep"), ("Repeated", "multi_paired")],
    ),
    "multi_indep": DecisionNode(
        "multi_indep",
        "Normal &\nequal variance?",
        "diamond",
        "primary",
        [("Yes", "anova"), ("No", "kruskal")],
    ),
    "anova": DecisionNode(
        "anova",
        "ANOVA",
        "rounded",
        "success",
        [("Post-hoc", "posthoc_param")],
        test_id="anova",
    ),
    "kruskal": DecisionNode(
        "kruskal",
        "Kruskal-Wallis",
        "rounded",
        "success",
        [("Post-hoc", "posthoc_nonparam")],
        test_id="kruskal",
    ),
    "multi_paired": DecisionNode(
        "multi_paired",
        "Friedman test\nnot yet in UI",
        "rounded",
        "warning",
        [],
    ),
    # -- Post-hoc --
    "posthoc_param": DecisionNode(
        "posthoc_param",
        "Which\npost-hoc?",
        "diamond",
        "primary",
        [
            ("Equal var.", "tukey"),
            ("Unequal var.", "games_howell"),
            ("vs Control", "dunnett"),
        ],
    ),
    "tukey": DecisionNode(
        "tukey",
        "Tukey HSD",
        "rounded",
        "success",
        [],
        test_id="tukey",
    ),
    "games_howell": DecisionNode(
        "games_howell",
        "Games-Howell",
        "rounded",
        "success",
        [],
        test_id="games_howell",
    ),
    "dunnett": DecisionNode(
        "dunnett",
        "Dunnett",
        "rounded",
        "success",
        [],
        test_id="dunnett",
    ),
    "posthoc_nonparam": DecisionNode(
        "posthoc_nonparam",
        "Dunn test\n(not yet in UI)",
        "rounded",
        "warning",
        [],
    ),
    # --- Categorical Branch ---
    "categorical": DecisionNode(
        "categorical",
        "Paired or\nindependent?",
        "diamond",
        "primary",
        [("Independent", "cat_indep"), ("Paired", "cat_paired")],
    ),
    "cat_indep": DecisionNode(
        "cat_indep",
        "Chi-Square",
        "rounded",
        "success",
        [],
        test_id="chi2",
    ),
    "cat_paired": DecisionNode(
        "cat_paired",
        "McNemar\n(not yet in UI)",
        "rounded",
        "warning",
        [],
    ),
    # --- Correlation Branch ---
    "correlation": DecisionNode(
        "correlation",
        "Normal\ndistribution?",
        "diamond",
        "primary",
        [("Yes", "pearson"), ("No / Unknown", "spearman")],
    ),
    "pearson": DecisionNode(
        "pearson",
        "Pearson\nCorrelation",
        "rounded",
        "success",
        [],
        test_id="pearson",
    ),
    "spearman": DecisionNode(
        "spearman",
        "Spearman\nCorrelation",
        "rounded",
        "success",
        [],
        test_id="spearman",
    ),
}


def get_decision_tree() -> Dict[str, Any]:
    """Return the decision tree as a JSON-serializable dictionary.

    Returns
    -------
    dict
        Tree with node IDs as keys, each containing label, shape,
        emphasis, children, and test_id fields.
    """
    return {
        node_id: {
            "id": node.id,
            "label": node.label,
            "shape": node.shape,
            "emphasis": node.emphasis,
            "children": [{"label": lbl, "target": tgt} for lbl, tgt in node.children],
            "test_id": node.test_id,
        }
        for node_id, node in DECISION_TREE.items()
    }


def render_flowchart_mermaid() -> str:
    """Render the decision tree as Mermaid markup text.

    Returns Mermaid text suitable for client-side rendering with Mermaid.js.
    This avoids the need for mermaid-cli (mmdc) on the server.

    Returns
    -------
    str
        Mermaid markup string.
    """
    import figrecipe._diagram as _fr_diagram

    d = _fr_diagram.Diagram(type="decision", title="Which Statistical Test?")

    for node in DECISION_TREE.values():
        d.add_node(node.id, node.label, shape=node.shape, emphasis=node.emphasis)
        for edge_label, child_id in node.children:
            d.add_edge(node.id, child_id, label=edge_label)

    return d.to_mermaid()  # type: ignore[no-any-return]


def render_flowchart_svg(output_path: Optional[Union[str, Path]] = None) -> str:
    """Render the decision tree as an SVG string using figrecipe Diagram.

    Parameters
    ----------
    output_path : str or Path, optional
        If provided, also write SVG to this file path.

    Returns
    -------
    str
        SVG content string.
    """
    import figrecipe._diagram as _fr_diagram

    d = _fr_diagram.Diagram(type="decision", title="Which Statistical Test?")

    for node in DECISION_TREE.values():
        d.add_node(node.id, node.label, shape=node.shape, emphasis=node.emphasis)
        for edge_label, child_id in node.children:
            d.add_edge(node.id, child_id, label=edge_label)

    if output_path:
        output_path = Path(output_path)
        d.render(output_path, format="svg")
        return output_path.read_text(encoding="utf-8")

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        d.render(tmp_path, format="svg")
        return tmp_path.read_text(encoding="utf-8")
    finally:
        tmp_path.unlink(missing_ok=True)


def get_leaf_node_test_ids() -> Dict[str, str]:
    """Return mapping of node_id -> test_id for all leaf nodes.

    Returns
    -------
    dict
        Mapping like {"ttest_ind": "ttest_ind", "indep_2_nonparam": "brunnermunzel"}.
    """
    return {
        node_id: node.test_id
        for node_id, node in DECISION_TREE.items()
        if node.test_id is not None
    }


__all__ = [
    "DecisionNode",
    "DECISION_TREE",
    "get_decision_tree",
    "render_flowchart_mermaid",
    "render_flowchart_svg",
    "get_leaf_node_test_ids",
]

# EOF
