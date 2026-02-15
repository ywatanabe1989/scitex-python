#!/usr/bin/env python3
# Timestamp: "2026-02-15 15:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/scripts/assets/workflow.py


"""Generate SciTeX overview diagram using figrecipe.diagram."""

from figrecipe._diagram import Diagram

import scitex as stx


@stx.session
def main(
    CONFIG=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    d = Diagram(title="SciTeX\nPython Engine for Reproducible Research", gap_mm=10)

    # -- Input --
    d.add_box("scripts", "Existing .py scripts", shape="stadium", emphasis="muted")
    d.add_box("linter", "linter", emphasis="warning")
    d.add_box("source", "Source Data", shape="stadium")
    d.add_box("disk", "Disk", subtitle="Storage", shape="cylinder", emphasis="muted")

    # -- Analysis (consolidated with bullet content lists) --
    d.add_box(
        "core",
        "Core",
        content=["io", "config", "logger", "session", "template", "repro", "rng"],
        emphasis="primary",
        bullet="circle",
    )
    d.add_box(
        "tools",
        "Tools",
        content=["plt \u00b7 figrecipe", "stats", "dsp", "ml", "nn"],
        emphasis="primary",
        bullet="circle",
    )

    # -- Publication (consolidated with bullet content lists) --
    d.add_box(
        "writer",
        "Writer",
        content=["LaTeX editor", "PDF preview", "version ctrl"],
        emphasis="success",
        bullet="circle",
    )
    d.add_box(
        "scholar",
        "Scholar",
        content=["Literature \u00b7 BibTeX", "CrossRef DB", "OpenAlex DB"],
        emphasis="success",
        bullet="circle",
    )

    # -- Output --
    d.add_box(
        "paper", "Paper", subtitle="manuscript", shape="document", emphasis="success"
    )
    d.add_box("cloud", "Cloud", subtitle="scitex.ai", shape="stadium", emphasis="muted")

    # -- External --
    d.add_box("user", "User")
    d.add_box(
        "ai",
        "AI Layer",
        subtitle="Agent \u00b7 LLM \u00b7 MCP tools",
        emphasis="warning",
    )

    # -- Containers --

    # Input (column)
    d.add_container(
        "input_grp",
        title="Input",
        children=["scripts", "linter", "source", "disk"],
        direction="column",
    )

    # Analysis = Core + Tools side-by-side (row)
    d.add_container(
        "analysis_grp",
        title="Analysis \u00b7 scitex-code",
        children=["core", "tools"],
        direction="row",
    )

    # Publication = Writer + Scholar stacked (column)
    d.add_container(
        "pub_grp",
        title="Publication",
        children=["writer", "scholar"],
        direction="column",
    )

    # Output (column)
    d.add_container(
        "output_grp",
        title="Output",
        children=["paper", "cloud"],
        direction="column",
    )

    # Top-level row (invisible — layout-only grouping)
    d.add_container(
        "main_row",
        children=["input_grp", "analysis_grp", "pub_grp", "output_grp"],
        direction="row",
        fill_color="none",
        border_color="none",
    )

    # AI container (row: User + AI side-by-side, below main)
    d.add_container(
        "ai_grp",
        title="AI",
        children=["user", "ai"],
        direction="row",
    )

    # -- Arrows --

    # Input flow
    d.add_arrow("scripts", "linter")
    d.add_arrow("linter", "core", label="migrate")
    d.add_arrow("source", "core")

    # Analysis -> Publication
    d.add_arrow("tools", "writer")
    d.add_arrow("tools", "scholar")

    # Publication -> Paper
    d.add_arrow("writer", "paper")

    # Clew verification (3 segmented reverse arrows)
    d.add_arrow("paper", "writer", style="dashed", label="Clew")
    d.add_arrow("writer", "core", style="dashed")
    d.add_arrow("core", "source", style="dashed")

    # User -> AI -> modules
    d.add_arrow("user", "ai", style="dashed")
    d.add_arrow("ai", "core", style="dashed")
    d.add_arrow("ai", "tools", style="dashed")
    d.add_arrow("ai", "writer", style="dashed")

    # Render (auto_fix handles arrow routing, sizing, etc.)
    fig, ax = d.render(auto_fix=True)
    stx.io.save(fig, CONFIG.SDIR_OUT / "workflow.png", dpi=300)

    return 0


if __name__ == "__main__":
    main()

# EOF
