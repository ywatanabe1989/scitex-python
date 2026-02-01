#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_viz/_plotly.py
"""Plotly-based interactive visualization for verification DAG."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .._chain import verify_chain, verify_run
from .._db import get_db


def generate_plotly_dag(
    session_id: Optional[str] = None,
    target_file: Optional[str] = None,
    title: str = "Verification DAG",
) -> go.Figure:
    """
    Generate interactive Plotly figure for verification DAG.

    Parameters
    ----------
    session_id : str, optional
        Start from this session
    target_file : str, optional
        Start from session that produced this file
    title : str, optional
        Title for the figure

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly required: pip install plotly")

    db = get_db()
    nodes = []
    edges = []
    node_colors = []
    node_texts = []

    if target_file:
        chain = verify_chain(target_file)
        for i, run in enumerate(chain.runs):
            script_name = Path(run.script_path).name if run.script_path else "unknown"
            nodes.append(run.session_id)
            node_texts.append(f"{script_name}<br>{run.session_id[:20]}...")
            node_colors.append("#90EE90" if run.is_verified else "#FFB6C1")

        for i in range(len(chain.runs) - 1):
            edges.append((i + 1, i))  # parent -> child

    elif session_id:
        chain_ids = db.get_chain(session_id)
        for i, sid in enumerate(chain_ids):
            run = db.get_run(sid)
            verification = verify_run(sid)
            script_name = (
                Path(run["script_path"]).name
                if run and run.get("script_path")
                else "unknown"
            )
            nodes.append(sid)
            node_texts.append(f"{script_name}<br>{sid[:20]}...")
            node_colors.append("#90EE90" if verification.is_verified else "#FFB6C1")

        for i in range(len(chain_ids) - 1):
            edges.append((i + 1, i))

    if not nodes:
        nodes = ["No data"]
        node_texts = ["No runs found"]
        node_colors = ["#CCCCCC"]

    # Create layout positions (vertical flow)
    n = len(nodes)
    x_pos = [0.5] * n
    y_pos = [1 - i / max(n - 1, 1) for i in range(n)]

    # Create edge traces
    edge_x = []
    edge_y = []
    for src, dst in edges:
        edge_x.extend([x_pos[src], x_pos[dst], None])
        edge_y.extend([y_pos[src], y_pos[dst], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=2, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Create node trace
    node_trace = go.Scatter(
        x=x_pos,
        y=y_pos,
        mode="markers+text",
        hoverinfo="text",
        text=node_texts,
        textposition="middle right",
        marker=dict(
            size=30,
            color=node_colors,
            line=dict(width=2, color="#333"),
        ),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="white",
            annotations=[
                dict(
                    text="🟢 Verified | 🔴 Failed",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0,
                    y=-0.1,
                    font=dict(size=12),
                )
            ],
        ),
    )

    return fig


def render_plotly_dag(
    output_path: Union[str, Path],
    session_id: Optional[str] = None,
    target_file: Optional[str] = None,
    title: str = "Verification DAG",
) -> Path:
    """
    Render verification DAG using Plotly.

    Parameters
    ----------
    output_path : str or Path
        Output file path (.html or .png)
    session_id : str, optional
        Start from this session
    target_file : str, optional
        Start from session that produced this file
    title : str, optional
        Title for the visualization

    Returns
    -------
    Path
        Path to the generated file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = generate_plotly_dag(
        session_id=session_id,
        target_file=target_file,
        title=title,
    )

    ext = output_path.suffix.lower()

    if ext == ".html":
        fig.write_html(str(output_path))
    elif ext == ".png":
        fig.write_image(str(output_path))
    elif ext == ".svg":
        fig.write_image(str(output_path))
    else:
        fig.write_html(str(output_path.with_suffix(".html")))
        return output_path.with_suffix(".html")

    return output_path


# EOF
