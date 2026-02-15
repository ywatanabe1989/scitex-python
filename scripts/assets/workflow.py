#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2026-02-16 02:49:17 (ywatanabe)"
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
    d = Diagram(title="SciTeX Ecosystem", gap_mm=10)

    # -- Input --
    d.add_box("scripts", "Python Script", shape="document", emphasis="muted")
    d.add_box(
        "linter",
        "Automatic translation",
        content=["scitex.linter", "Smooth Migration to SciTeX"],
        padding_mm=8,
        fill_color="#E6A01419",
        border_color="#E6A014",
        width_mm=50,
    )
    d.add_box("source", "Source Data", shape="cylinder", height_mm=40)

    # -- Analysis & Visualization (nested containers) --
    # Core infrastructure
    d.add_box(
        "core",
        "Core",
        content=[
            "scitex.io",
            "scitex.config",
            "scitex.logger",
            "scitex.session",
            "scitex.template",
            "scitex.repro",
            "scitex.rng",
        ],
        fill_color="#0080C019",
        border_color="#0080C0",
        bullet="circle",
    )

    # Domain-specific analysis
    d.add_box(
        "domain",
        "Domain-specific",
        content=[
            "scitex.plt",
            "scitex.stats",
            "scitex.dsp",
            "scitex.ml",
            "scitex.nn",
        ],
        fill_color="#14B41419",
        border_color="#14B414",
        bullet="circle",
    )

    # -- Publication (consolidated with bullet content lists) --
    d.add_box(
        "writer",
        "Writer",
        subtitle="LaTeX Compilation",
        content=["scitex.writer"],
        fill_color="#FF463219",
        border_color="#FF4632",
        bullet="circle",
    )
    d.add_box(
        "scholar",
        "Scholar",
        subtitle="Literature management",
        content=[
            "scitex.scholar",
            "CrossRef Local Database",
            "OpenAlex Local Database",
        ],
        fill_color="#C832FF19",
        border_color="#C832FF",
        bullet="circle",
    )

    # Spacers for layout alignment
    d.add_box("spacer", "", fill_color="none", border_color="none")
    d.add_box("source_pad", "", fill_color="none", border_color="none")
    d.add_box("paper_pad", "", fill_color="none", border_color="none")

    # -- Output --
    d.add_box(
        "paper",
        "Manuscript",
        content=["Manuscript", "Supplementary materials", "Revision Letter"],
        bullet="circle",
        shape="document",
        emphasis="muted",
        height_mm=40,
    )
    d.add_box(
        "cloud",
        "Cloud",
        subtitle="Web Interface and collaboration",
        content=["scitex.cloud", "https://scitex.ai", "self-host"],
        bullet="circle",
        shape="stadium",
        emphasis="muted",
    )

    # -- Containers --

    # Upper row: scripts → linter → spacer → cloud
    d.add_container(
        "upper_row",
        children=["scripts", "linter", "spacer", "cloud"],
        direction="row",
        fill_color="none",
        border_color="none",
    )

    # Inner container: Analysis & Visualization = Core + Domain-specific (row)
    d.add_container(
        "analysis_grp",
        title="Analysis & Visualization",
        title_loc="upper center",
        children=["core", "domain"],
        direction="row",
        # fill_color="#E8E8E8",
    )

    # FIX 2: Reversed column order — bottom-up data flow:
    #   analysis_grp (bottom) → scholar (middle) → writer (top)
    d.add_container(
        "main_grp",
        children=["analysis_grp", "scholar", "writer"],
        direction="column",
    )

    # Source at top, paper at bottom (column wrappers with spacers)
    d.add_container(
        "source_col",
        children=["source", "source_pad"],
        direction="column",
        fill_color="none",
        border_color="none",
    )
    d.add_container(
        "paper_col",
        children=["paper_pad", "paper"],
        direction="column",
        fill_color="none",
        border_color="none",
    )

    # Main row: source (top-aligned) → analysis → paper (bottom-aligned)
    d.add_container(
        "main_row",
        children=["source_col", "main_grp", "paper_col"],
        direction="row",
        fill_color="none",
        border_color="none",
    )

    # -- Arrows --

    # Input stage
    d.add_arrow("scripts", "linter")
    # FIX 3: Reduced curve for linter→core to avoid wild arc
    d.add_arrow("linter", "core", curve=0.3)
    d.add_arrow("source", "core", curve=0)

    # Analysis flow (bottom-up): core → domain → scholar → writer
    d.add_arrow("domain", "writer", curve=-0.5)
    d.add_arrow("scholar", "writer", curve=0)

    # Output
    # FIX 4: Reduced curve magnitude for writer→paper
    d.add_arrow("writer", "paper", curve=0.3, target_anchor="lower-left")

    # # Cloud connection
    # d.add_arrow("writer", "cloud", curve=-0.3)

    # Render
    fig, ax = d.render(auto_fix=True)
    stx.io.save(fig, CONFIG.SDIR_OUT / "workflow.png", dpi=300)

    return 0


if __name__ == "__main__":
    main()

# EOF
