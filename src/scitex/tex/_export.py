#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-11 16:00:00
# File: /home/ywatanabe/proj/scitex-code/src/scitex/tex/_export.py

"""
Export SciTeX writer documents to LaTeX format.

This module converts the intermediate document format (from scitex.msword
or scitex.writer) into LaTeX source files.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Journal-specific document class configurations
JOURNAL_PRESETS = {
    "article": {
        "document_class": "article",
        "class_options": [],
        "required_packages": [],
    },
    "ieee": {
        "document_class": "IEEEtran",
        "class_options": ["conference"],
        "required_packages": ["cite", "amsmath", "algorithmic"],
    },
    "elsevier": {
        "document_class": "elsarticle",
        "class_options": ["preprint", "12pt"],
        "required_packages": ["lineno", "hyperref"],
    },
    "springer": {
        "document_class": "svjour3",
        "class_options": ["smallextended"],
        "required_packages": [],
    },
    "aps": {
        "document_class": "revtex4-2",
        "class_options": ["aps", "prl", "preprint"],
        "required_packages": [],
    },
    "mdpi": {
        "document_class": "article",
        "class_options": [],
        "required_packages": ["mdpi"],
    },
    "acm": {
        "document_class": "acmart",
        "class_options": ["sigconf"],
        "required_packages": [],
    },
}


def export_tex(
    writer_doc: Dict[str, Any],
    output_path: str | Path,
    document_class: str = "article",
    packages: Optional[List[str]] = None,
    preamble: Optional[str] = None,
    image_dir: Optional[str | Path] = None,
    export_images: bool = True,
    journal_preset: Optional[str] = None,
    class_options: Optional[List[str]] = None,
    use_bibtex: bool = False,
) -> Path:
    """
    Export a SciTeX writer document to LaTeX format.

    Parameters
    ----------
    writer_doc : dict
        SciTeX writer document structure containing:
        - blocks: List of document blocks (headings, paragraphs, captions, etc.)
        - metadata: Document metadata (title, author, etc.)
        - images: Image references with binary data
        - references: Bibliography entries
    output_path : str | Path
        Output path for the .tex file.
    document_class : str
        LaTeX document class (article, report, book, etc.).
        Overridden if journal_preset is specified.
    packages : list[str] | None
        Additional LaTeX packages to include.
    preamble : str | None
        Additional preamble content.
    image_dir : str | Path | None
        Directory to save extracted images. If None, uses
        "{output_stem}_figures/" next to the output .tex file.
        Set export_images=False to skip image export.
    export_images : bool
        Whether to export images to files. Default True.
    journal_preset : str | None
        Use a journal-specific preset: "ieee", "elsevier", "springer",
        "aps", "mdpi", "acm". Sets document_class and required packages.
    class_options : list[str] | None
        Document class options (e.g., ["12pt", "twocolumn"]).
    use_bibtex : bool
        If True, generate \\bibliography{} instead of thebibliography.
        Creates a .bib file alongside the .tex file.

    Returns
    -------
    Path
        The path to the written .tex file.

    Examples
    --------
    >>> from scitex.msword import load_docx
    >>> from scitex.tex import export_tex
    >>> doc = load_docx("manuscript.docx")
    >>> export_tex(doc, "manuscript.tex")
    PosixPath('manuscript.tex')

    >>> # Export for IEEE conference
    >>> export_tex(doc, "manuscript.tex", journal_preset="ieee")

    >>> # Export with custom image directory
    >>> export_tex(doc, "manuscript.tex", image_dir="./figures")
    """
    output_path = Path(output_path)

    # Apply journal preset if specified
    effective_class = document_class
    effective_options = class_options or []
    extra_packages = []

    if journal_preset and journal_preset in JOURNAL_PRESETS:
        preset = JOURNAL_PRESETS[journal_preset]
        effective_class = preset["document_class"]
        effective_options = preset["class_options"] + (class_options or [])
        extra_packages = preset["required_packages"]

    # Extract components from writer_doc
    blocks = writer_doc.get("blocks", [])
    metadata = writer_doc.get("metadata", {})
    references = writer_doc.get("references", [])
    images = writer_doc.get("images", [])

    # Handle image export
    image_map: Dict[str, str] = {}  # hash -> relative path
    if export_images and images:
        if image_dir is None:
            image_dir = output_path.parent / f"{output_path.stem}_figures"
        else:
            image_dir = Path(image_dir)

        image_dir.mkdir(parents=True, exist_ok=True)
        image_map = _write_images_to_dir(images, image_dir, output_path.parent)

    # Combine packages
    all_packages = extra_packages + (packages or [])

    # Build LaTeX content
    latex_content = _build_latex_document(
        blocks=blocks,
        metadata=metadata,
        references=references,
        document_class=effective_class,
        class_options=effective_options,
        packages=all_packages if all_packages else None,
        preamble=preamble,
        image_map=image_map,
        use_bibtex=use_bibtex,
        output_stem=output_path.stem,
    )

    # Write to file
    output_path.write_text(latex_content, encoding="utf-8")

    # Generate .bib file if using bibtex
    if use_bibtex and references:
        bib_path = output_path.with_suffix(".bib")
        bib_content = _generate_bibtex(references)
        bib_path.write_text(bib_content, encoding="utf-8")

    return output_path


def _generate_bibtex(references: List[Dict[str, Any]]) -> str:
    """Generate BibTeX content from references."""
    entries = []
    for ref in references:
        num = ref.get("number", len(entries) + 1)
        text = ref.get("text", ref.get("raw", ""))

        # Basic entry - in practice, would parse author/title/year
        entry = f"""@misc{{ref{num},
  note = {{{text}}}
}}"""
        entries.append(entry)

    return "\n\n".join(entries)


def _write_images_to_dir(
    images: List[Dict[str, Any]],
    image_dir: Path,
    tex_parent: Path,
) -> Dict[str, str]:
    """
    Write images to directory and return hash->relative_path mapping.

    Parameters
    ----------
    images : list
        List of image dicts with 'hash', 'extension', 'data' keys.
    image_dir : Path
        Directory to write images to.
    tex_parent : Path
        Parent directory of the .tex file (for relative paths).

    Returns
    -------
    dict
        Mapping from image hash to relative path for LaTeX.
    """
    image_map = {}
    fig_counter = 0

    for img in images:
        img_hash = img.get("hash")
        ext = img.get("extension", ".png")
        data = img.get("data")

        if data is None or img_hash is None:
            continue

        # Skip duplicates (same hash = same image content)
        if img_hash in image_map:
            continue

        fig_counter += 1
        filename = f"fig_{fig_counter}{ext}"
        filepath = image_dir / filename

        # Write image data
        filepath.write_bytes(data)

        # Store relative path from tex file location
        try:
            rel_path = filepath.relative_to(tex_parent)
        except ValueError:
            rel_path = filepath

        image_map[img_hash] = str(rel_path)

    return image_map


def _build_latex_document(
    blocks: List[Dict[str, Any]],
    metadata: Dict[str, Any],
    references: List[Dict[str, Any]],
    document_class: str,
    class_options: Optional[List[str]] = None,
    packages: Optional[List[str]] = None,
    preamble: Optional[str] = None,
    image_map: Optional[Dict[str, str]] = None,
    use_bibtex: bool = False,
    output_stem: str = "document",
) -> str:
    """Build complete LaTeX document content."""
    if image_map is None:
        image_map = {}
    lines = []

    # Document class with options
    if class_options:
        opts = ",".join(class_options)
        lines.append(f"\\documentclass[{opts}]{{{document_class}}}")
    else:
        lines.append(f"\\documentclass{{{document_class}}}")
    lines.append("")

    # Default packages
    default_packages = [
        "inputenc",
        "fontenc",
        "amsmath",
        "amssymb",
        "graphicx",
        "hyperref",
    ]

    # Package options
    package_options = {
        "inputenc": "utf8",
        "fontenc": "T1",
    }

    for pkg in default_packages:
        opt = package_options.get(pkg)
        if opt:
            lines.append(f"\\usepackage[{opt}]{{{pkg}}}")
        else:
            lines.append(f"\\usepackage{{{pkg}}}")

    # Additional packages
    if packages:
        for pkg in packages:
            if pkg not in default_packages:
                lines.append(f"\\usepackage{{{pkg}}}")

    lines.append("")

    # Metadata
    if metadata.get("title"):
        title = _escape_latex(metadata["title"])
        lines.append(f"\\title{{{title}}}")
    if metadata.get("author"):
        author = _escape_latex(metadata["author"])
        lines.append(f"\\author{{{author}}}")

    lines.append("")

    # Additional preamble
    if preamble:
        lines.append(preamble)
        lines.append("")

    # Begin document
    lines.append("\\begin{document}")
    lines.append("")

    # Title
    if metadata.get("title"):
        lines.append("\\maketitle")
        lines.append("")

    # Track list state for proper itemize/enumerate environments
    in_list = False
    list_type = None

    # Process blocks
    for i, block in enumerate(blocks):
        btype = block.get("type")

        # Handle list transitions
        if btype == "list-item":
            item_list_type = block.get("list_type", "unordered")
            if not in_list:
                env = "enumerate" if item_list_type == "ordered" else "itemize"
                lines.append(f"\\begin{{{env}}}")
                in_list = True
                list_type = item_list_type
        elif in_list:
            # Close list environment
            env = "enumerate" if list_type == "ordered" else "itemize"
            lines.append(f"\\end{{{env}}}")
            lines.append("")
            in_list = False
            list_type = None

        block_latex = _convert_block_to_latex(block, image_map)
        if block_latex:
            lines.append(block_latex)

    # Close any open list
    if in_list:
        env = "enumerate" if list_type == "ordered" else "itemize"
        lines.append(f"\\end{{{env}}}")
        lines.append("")

    # References section
    if references:
        lines.append("")
        if use_bibtex:
            lines.append(f"\\bibliographystyle{{plain}}")
            lines.append(f"\\bibliography{{{output_stem}}}")
        else:
            lines.append("\\begin{thebibliography}{99}")
            for ref in references:
                ref_latex = _convert_reference_to_latex(ref)
                if ref_latex:
                    lines.append(ref_latex)
            lines.append("\\end{thebibliography}")

    # End document
    lines.append("")
    lines.append("\\end{document}")

    return "\n".join(lines)


def _convert_block_to_latex(
    block: Dict[str, Any],
    image_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Convert a single block to LaTeX."""
    if image_map is None:
        image_map = {}

    btype = block.get("type", "paragraph")
    text = block.get("text", "")

    if not text and btype not in ("table", "image", "caption", "equation"):
        return None

    if btype == "heading":
        return _convert_heading(block)
    elif btype == "paragraph":
        return _convert_paragraph(block)
    elif btype == "caption":
        return _convert_caption(block, image_map)
    elif btype == "table":
        return _convert_table(block)
    elif btype == "image":
        return _convert_image(block, image_map)
    elif btype == "list-item":
        return _convert_list_item(block)
    elif btype == "equation":
        return _convert_equation(block)
    elif btype == "reference-paragraph":
        # Skip - handled separately in references section
        return None
    else:
        # Default: treat as paragraph
        return _escape_latex(text) + "\n"


def _convert_equation(block: Dict[str, Any]) -> str:
    """Convert an equation block to LaTeX."""
    latex = block.get("latex", "")
    text = block.get("text", "")

    if latex:
        # Use the converted LaTeX from OMML
        return f"\\begin{{equation}}\n{latex}\n\\end{{equation}}\n"
    elif text:
        # Fallback: wrap text in equation environment
        return f"\\begin{{equation}}\n{_escape_latex(text)}\n\\end{{equation}}\n"
    return ""


def _convert_heading(block: Dict[str, Any]) -> str:
    """Convert a heading block to LaTeX."""
    level = block.get("level", 1)
    text = _escape_latex(block.get("text", ""))

    # Map heading levels to LaTeX commands
    level_commands = {
        1: "section",
        2: "subsection",
        3: "subsubsection",
        4: "paragraph",
        5: "subparagraph",
    }

    command = level_commands.get(level, "paragraph")
    return f"\\{command}{{{text}}}\n"


def _convert_paragraph(block: Dict[str, Any]) -> str:
    """Convert a paragraph block to LaTeX."""
    runs = block.get("runs", [])

    if runs:
        # Build paragraph from formatted runs
        parts = []
        for run in runs:
            run_text = _escape_latex(run.get("text", ""))
            if run.get("bold"):
                run_text = f"\\textbf{{{run_text}}}"
            if run.get("italic"):
                run_text = f"\\textit{{{run_text}}}"
            if run.get("underline"):
                run_text = f"\\underline{{{run_text}}}"
            parts.append(run_text)
        return "".join(parts) + "\n"
    else:
        return _escape_latex(block.get("text", "")) + "\n"


def _convert_caption(
    block: Dict[str, Any],
    image_map: Optional[Dict[str, str]] = None,
) -> str:
    """Convert a caption block to LaTeX figure/table environment."""
    if image_map is None:
        image_map = {}

    caption_type = block.get("caption_type", "")
    number = block.get("number", "")
    caption_text = _escape_latex(block.get("caption_text", block.get("text", "")))
    image_hash = block.get("image_hash")

    if caption_type == "figure":
        # Check if we have an associated image
        image_path = None
        if image_hash and image_hash in image_map:
            image_path = image_map[image_hash]

        lines = [
            "\\begin{figure}[htbp]",
            "\\centering",
        ]

        if image_path:
            # Remove extension for includegraphics
            image_path_no_ext = image_path.rsplit(".", 1)[0] if "." in image_path else image_path
            lines.append(f"\\includegraphics[width=0.8\\textwidth]{{{image_path_no_ext}}}")
        else:
            lines.append(f"% Image placeholder for Figure {number}")

        lines.extend([
            f"\\caption{{{caption_text}}}",
            f"\\label{{fig:{number}}}",
            "\\end{figure}",
            "",
        ])
        return "\n".join(lines)

    elif caption_type == "table":
        # Table captions - typically above the table
        return f"% Table {number}: {caption_text}\n"

    else:
        return f"% Caption: {caption_text}\n"


def _convert_image(
    block: Dict[str, Any],
    image_map: Optional[Dict[str, str]] = None,
) -> str:
    """Convert an image block to LaTeX includegraphics."""
    if image_map is None:
        image_map = {}

    image_hash = block.get("image_hash") or block.get("hash")
    width = block.get("width", "0.8\\textwidth")

    if image_hash and image_hash in image_map:
        image_path = image_map[image_hash]
        # Remove extension for includegraphics
        image_path_no_ext = image_path.rsplit(".", 1)[0] if "." in image_path else image_path

        lines = [
            "\\begin{figure}[htbp]",
            "\\centering",
            f"\\includegraphics[width={width}]{{{image_path_no_ext}}}",
            "\\end{figure}",
            "",
        ]
        return "\n".join(lines)

    return "% Image placeholder\n"


def _convert_table(block: Dict[str, Any]) -> str:
    """Convert a table block to LaTeX."""
    rows = block.get("rows", [])
    if not rows:
        return ""

    num_cols = len(rows[0]) if rows else 0
    col_spec = "|" + "c|" * num_cols

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]

    for i, row in enumerate(rows):
        escaped_cells = [_escape_latex(str(cell)) for cell in row]
        lines.append(" & ".join(escaped_cells) + " \\\\")
        lines.append("\\hline")

    lines.extend([
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])

    return "\n".join(lines)


def _convert_list_item(block: Dict[str, Any]) -> str:
    """Convert a list item to LaTeX."""
    text = _escape_latex(block.get("text", ""))
    return f"\\item {text}\n"


def _convert_reference_to_latex(ref: Dict[str, Any]) -> str:
    """Convert a reference entry to LaTeX bibitem."""
    number = ref.get("number")
    text = _escape_latex(ref.get("text", ref.get("raw", "")))

    if number:
        return f"\\bibitem{{ref{number}}} {text}"
    else:
        return f"\\bibitem{{}} {text}"


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    if not text:
        return ""

    # Characters that need escaping in LaTeX
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]

    # Apply replacements (order matters - backslash first)
    result = text
    for old, new in replacements:
        # Skip if already escaped
        if old == "\\":
            # Don't escape existing LaTeX commands
            result = re.sub(r'(?<!\\)\\(?![a-zA-Z{])', new, result)
        else:
            result = result.replace(old, new)

    return result


@dataclass
class CompileResult:
    """Result of LaTeX compilation.

    Attributes
    ----------
    success : bool
        Whether compilation succeeded.
    pdf_path : Path | None
        Path to generated PDF, or None if failed.
    exit_code : int
        Process exit code.
    stdout : str
        Standard output from compiler.
    stderr : str
        Standard error from compiler.
    log_content : str
        Content of .log file if available.
    errors : list[str]
        Extracted error messages.
    warnings : list[str]
        Extracted warning messages.
    """
    success: bool
    pdf_path: Optional[Path]
    exit_code: int
    stdout: str
    stderr: str
    log_content: str = ""
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


def compile_tex(
    tex_path: str | Path,
    output_dir: Optional[str | Path] = None,
    compiler: str = "pdflatex",
    runs: int = 2,
    clean: bool = True,
    timeout: int = 120,
) -> CompileResult:
    """
    Compile a LaTeX file to PDF.

    Parameters
    ----------
    tex_path : str | Path
        Path to the .tex file.
    output_dir : str | Path | None
        Output directory for PDF. If None, uses same directory as tex file.
    compiler : str
        LaTeX compiler to use: "pdflatex", "xelatex", "lualatex", or "latexmk".
        Default is "pdflatex".
    runs : int
        Number of compilation passes (for references/ToC). Default is 2.
        Ignored if compiler is "latexmk".
    clean : bool
        Remove auxiliary files (.aux, .log, .out, etc.) after compilation.
        Default is True.
    timeout : int
        Timeout in seconds for each compilation pass. Default is 120.

    Returns
    -------
    CompileResult
        Compilation result with success status, PDF path, and logs.

    Examples
    --------
    >>> from scitex.tex import compile_tex
    >>> result = compile_tex("manuscript.tex")
    >>> if result.success:
    ...     print(f"PDF created: {result.pdf_path}")
    ... else:
    ...     print(f"Errors: {result.errors}")

    >>> # Use latexmk for automatic multi-pass compilation
    >>> result = compile_tex("manuscript.tex", compiler="latexmk")

    Notes
    -----
    Requires LaTeX to be installed on the system (texlive, miktex, etc.).
    """
    tex_path = Path(tex_path).absolute()

    if not tex_path.exists():
        return CompileResult(
            success=False,
            pdf_path=None,
            exit_code=1,
            stdout="",
            stderr=f"File not found: {tex_path}",
            errors=[f"File not found: {tex_path}"],
        )

    # Determine output directory
    if output_dir is None:
        output_dir = tex_path.parent
    else:
        output_dir = Path(output_dir).absolute()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Check if compiler is available
    compiler_cmd = shutil.which(compiler)
    if compiler_cmd is None:
        return CompileResult(
            success=False,
            pdf_path=None,
            exit_code=127,
            stdout="",
            stderr=f"Compiler not found: {compiler}",
            errors=[f"Compiler not found: {compiler}. Install texlive or miktex."],
        )

    # Build command
    if compiler == "latexmk":
        cmd = [
            compiler,
            "-pdf",
            "-interaction=nonstopmode",
            f"-output-directory={output_dir}",
            str(tex_path),
        ]
        runs = 1  # latexmk handles multi-pass
    else:
        cmd = [
            compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output_dir}",
            str(tex_path),
        ]

    # Run compilation
    stdout_all = []
    stderr_all = []
    exit_code = 0

    for run_num in range(runs):
        try:
            result = subprocess.run(
                cmd,
                cwd=tex_path.parent,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout_all.append(f"=== Pass {run_num + 1} ===\n{result.stdout}")
            stderr_all.append(result.stderr)
            exit_code = result.returncode

            # If compilation failed, don't continue
            if exit_code != 0:
                break

        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                pdf_path=None,
                exit_code=124,
                stdout="\n".join(stdout_all),
                stderr=f"Compilation timed out after {timeout} seconds",
                errors=[f"Compilation timed out after {timeout} seconds"],
            )
        except Exception as e:
            return CompileResult(
                success=False,
                pdf_path=None,
                exit_code=1,
                stdout="\n".join(stdout_all),
                stderr=str(e),
                errors=[str(e)],
            )

    # Check for output PDF
    pdf_name = tex_path.stem + ".pdf"
    pdf_path = output_dir / pdf_name

    # Read log file for detailed errors/warnings
    log_path = output_dir / (tex_path.stem + ".log")
    log_content = ""
    errors = []
    warnings = []

    if log_path.exists():
        try:
            log_content = log_path.read_text(encoding="utf-8", errors="replace")
            errors, warnings = _parse_latex_log(log_content)
        except Exception:
            pass

    # Clean auxiliary files
    if clean:
        aux_extensions = [".aux", ".log", ".out", ".toc", ".lof", ".lot",
                         ".bbl", ".blg", ".fls", ".fdb_latexmk", ".synctex.gz"]
        for ext in aux_extensions:
            aux_file = output_dir / (tex_path.stem + ext)
            if aux_file.exists():
                try:
                    aux_file.unlink()
                except Exception:
                    pass

    success = exit_code == 0 and pdf_path.exists()

    return CompileResult(
        success=success,
        pdf_path=pdf_path if pdf_path.exists() else None,
        exit_code=exit_code,
        stdout="\n".join(stdout_all),
        stderr="\n".join(stderr_all),
        log_content=log_content,
        errors=errors,
        warnings=warnings,
    )


def _parse_latex_log(log_content: str) -> Tuple[List[str], List[str]]:
    """Parse LaTeX log file for errors and warnings."""
    errors = []
    warnings = []

    lines = log_content.split("\n")

    for i, line in enumerate(lines):
        # Error patterns
        if line.startswith("!"):
            # Collect multi-line error message
            error_lines = [line]
            for j in range(i + 1, min(i + 5, len(lines))):
                if lines[j].startswith("l.") or lines[j].strip() == "":
                    break
                error_lines.append(lines[j])
            errors.append(" ".join(error_lines))

        elif "Error:" in line or "Fatal error" in line:
            errors.append(line.strip())

        # Warning patterns
        elif "Warning:" in line:
            warnings.append(line.strip())
        elif "Underfull" in line or "Overfull" in line:
            warnings.append(line.strip())

    return errors, warnings


__all__ = ["export_tex", "compile_tex", "CompileResult"]
