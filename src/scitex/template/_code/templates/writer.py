#!/usr/bin/env python3
# Timestamp: 2026-01-25
# File: src/scitex/template/_templates/writer.py
# ----------------------------------------

"""Template for stx.writer LaTeX manuscript compilation module usage."""

TEMPLATE = {
    "name": "Writer Module",
    "description": "stx.writer usage for LaTeX manuscript compilation and paper writing",
    "filename": "writer_script.py",
    "priority": 9,
    "content": '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: {timestamp}
# File: {filepath}

"""
stx.writer - LaTeX Manuscript Compilation
=========================================

stx.writer provides:
- LaTeX to PDF compilation
- BibTeX bibliography processing
- Figure and table management
- Word count and statistics
- Manuscript structure scaffolding

Usage Patterns
--------------
1. CLI: scitex writer <command> [options]
2. MCP: writer_usage tool for documentation
3. Python API: stx.writer.* functions

Typical Workflow
----------------
1. Clone paper template: scitex template clone paper ./manuscript
2. Write manuscript: Edit .tex files
3. Compile: scitex writer compile ./manuscript
4. Check: scitex writer wordcount ./manuscript
"""

import scitex as stx

# ============================================================
# Pattern 1: CLI Usage (Recommended)
# ============================================================

"""
CLI Commands:
-------------

# Compile manuscript to PDF
scitex writer compile ./manuscript
scitex writer compile ./manuscript --output paper.pdf

# Compile with bibliography
scitex writer compile ./manuscript --bib

# Clean auxiliary files
scitex writer clean ./manuscript

# Word count
scitex writer wordcount ./manuscript

# Check for common LaTeX errors
scitex writer check ./manuscript

# Watch for changes and auto-compile
scitex writer watch ./manuscript
"""

# ============================================================
# Pattern 2: Project Structure
# ============================================================

"""
Recommended Manuscript Structure:
---------------------------------

manuscript/
  main.tex              # Main document
  sections/
    abstract.tex        # Abstract
    introduction.tex    # Introduction
    methods.tex         # Methods
    results.tex         # Results
    discussion.tex      # Discussion
    references.tex      # References section
  figures/
    fig1.png            # Figure files
    fig2.pdf
  tables/
    table1.tex          # LaTeX tables
  references.bib        # BibTeX bibliography
  config/
    preamble.tex        # Custom packages/macros

Create this structure with:
scitex template clone paper ./manuscript
"""

# ============================================================
# Pattern 3: LaTeX Best Practices
# ============================================================

"""
LaTeX Writing Guidelines:
-------------------------

1. Document Structure:
   \\\\documentclass[article]
   \\\\input[config/preamble]
   \\\\begin[document]
   \\\\input[sections/abstract]
   \\\\input[sections/introduction]
   \\\\input[sections/methods]
   \\\\input[sections/results]
   \\\\input[sections/discussion]
   \\\\bibliography[references]
   \\\\end[document]

2. Figure Inclusion:
   \\\\begin[figure][htbp]
       \\\\centering
       \\\\includegraphics[width=0.8\\\\linewidth][figures/fig1.png]
       \\\\caption[Description of the figure.]
       \\\\label[fig:main]
   \\\\end[figure]

3. Table Inclusion:
   \\\\begin[table][htbp]
       \\\\centering
       \\\\caption[Description of the table.]
       \\\\label[tab:results]
       \\\\input[tables/table1]
   \\\\end[table]

4. Citations:
   As shown in previous work~\\\\cite[author2023]...
   Multiple citations~\\\\cite[ref1,ref2,ref3]...

5. Cross-references:
   See Figure~\\\\ref[fig:main]...
   Table~\\\\ref[tab:results] shows...
   Section~\\\\ref[sec:methods] describes...

Note: Replace [] with curly braces in actual LaTeX code.
"""

# ============================================================
# Pattern 4: BibTeX Management
# ============================================================

"""
BibTeX Best Practices:
----------------------

# references.bib structure
@article key=author2023,
  author  = Smith, John and Doe, Jane,
  title   = Title of the Paper,
  journal = Journal Name,
  year    = 2023,
  volume  = 10,
  pages   = 1--15,
  doi     = 10.1000/example


# Enrich with scitex scholar
scitex scholar enrich --bibtex references.bib

# This adds:
# - Missing DOIs
# - Abstracts
# - Citation counts
# - Impact factors
"""

# ============================================================
# Pattern 5: Integration with SciTeX
# ============================================================

@stx.session
def main(
    manuscript_dir="./manuscript",
    CONFIG=stx.INJECTED,
    logger=stx.INJECTED,
):
    """Manuscript preparation workflow."""
    from pathlib import Path

    ms_dir = Path(manuscript_dir)

    # 1. Generate figures from analysis
    logger.info("Generating figures...")
    # (figures created with stx.plt would go in ms_dir/figures/)

    # 2. Generate tables from data
    logger.info("Generating tables...")
    # (tables created with stx.io.save() in LaTeX format)

    # 3. Copy to manuscript directory
    import shutil
    if (CONFIG.SDIR_OUT / "figures").exists():
        for fig in (CONFIG.SDIR_OUT / "figures").glob("*.png"):
            shutil.copy(fig, ms_dir / "figures" / fig.name)
            logger.info("Copied figure to manuscript")

    logger.info("Figures ready in manuscript/figures/")
    return 0

# ============================================================
# Pattern 6: Python API
# ============================================================

def python_api_example():
    """Direct Python API for writer."""
    from scitex.writer import (
        compile_manuscript,
        clean_auxiliary,
        word_count,
        check_manuscript,
    )

    # Compile
    result = compile_manuscript(
        manuscript_dir="./manuscript",
        output="paper.pdf",
        bibliography=True
    )
    print("Compilation complete")

    # Word count
    stats = word_count("./manuscript")
    print("Word count complete")

    # Check for issues
    issues = check_manuscript("./manuscript")
    for issue in issues:
        print("Warning found")

    # Clean
    clean_auxiliary("./manuscript")

# ============================================================
# Pattern 7: Automated Paper Generation
# ============================================================

"""
Automated Workflow:
-------------------

1. Run analysis scripts:
   python scripts/analysis.py

2. Figures auto-generated to:
   script_out/FINISHED_SUCCESS/<session>/

3. Copy figures to manuscript:
   cp script_out/FINISHED_SUCCESS/*/figures/* manuscript/figures/

4. Compile manuscript:
   scitex writer compile ./manuscript

5. Check word count:
   scitex writer wordcount ./manuscript
"""

# ============================================================
# MCP Tools Reference
# ============================================================

"""
MCP Tools:
----------

Documentation:
- writer_usage()  # Get complete usage guide

Note: Most writer operations are done via CLI:
- scitex writer compile <dir>
- scitex writer clean <dir>
- scitex writer wordcount <dir>
- scitex writer check <dir>
- scitex writer watch <dir>
"""

if __name__ == "__main__":
    main()
''',
}

__all__ = ["TEMPLATE"]

# EOF
