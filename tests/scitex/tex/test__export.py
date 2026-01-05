#!/usr/bin/env python3
# Time-stamp: "2026-01-05 14:00:00 (ywatanabe)"
# File: ./tests/scitex/tex/test__export.py

"""Comprehensive tests for tex._export module."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from scitex.tex._export import (
    JOURNAL_PRESETS,
    CompileResult,
    _build_latex_document,
    _convert_caption,
    _convert_equation,
    _convert_heading,
    _convert_image,
    _convert_list_item,
    _convert_paragraph,
    _convert_reference_to_latex,
    _convert_table,
    _escape_latex,
    _generate_bibtex,
    _parse_latex_log,
    _write_images_to_dir,
    compile_tex,
    export_tex,
)


class TestEscapeLatex:
    """Tests for _escape_latex helper function."""

    def test_empty_string(self):
        """Test with empty string."""
        assert _escape_latex("") == ""

    def test_plain_text(self):
        """Test with plain text (no special chars)."""
        assert _escape_latex("Hello World") == "Hello World"

    def test_ampersand(self):
        """Test escaping ampersand."""
        assert _escape_latex("A & B") == r"A \& B"

    def test_percent(self):
        """Test escaping percent sign."""
        assert _escape_latex("100%") == r"100\%"

    def test_dollar(self):
        """Test escaping dollar sign."""
        assert _escape_latex("$100") == r"\$100"

    def test_hash(self):
        """Test escaping hash/pound sign."""
        assert _escape_latex("Item #1") == r"Item \#1"

    def test_underscore(self):
        """Test escaping underscore."""
        assert _escape_latex("var_name") == r"var\_name"

    def test_braces(self):
        """Test escaping curly braces."""
        assert _escape_latex("{text}") == r"\{text\}"

    def test_tilde(self):
        """Test escaping tilde."""
        result = _escape_latex("~")
        assert "textasciitilde" in result

    def test_caret(self):
        """Test escaping caret."""
        result = _escape_latex("^")
        assert "textasciicircum" in result

    def test_multiple_special_chars(self):
        """Test escaping multiple special characters."""
        result = _escape_latex("$100 & 50%")
        assert r"\$" in result
        assert r"\&" in result
        assert r"\%" in result

    def test_none_input(self):
        """Test with None-like falsy input."""
        # Function handles empty string
        assert _escape_latex("") == ""


class TestConvertHeading:
    """Tests for _convert_heading helper function."""

    def test_level_1_heading(self):
        """Test level 1 heading (section)."""
        block = {"type": "heading", "level": 1, "text": "Introduction"}
        result = _convert_heading(block)
        assert r"\section{Introduction}" in result

    def test_level_2_heading(self):
        """Test level 2 heading (subsection)."""
        block = {"type": "heading", "level": 2, "text": "Methods"}
        result = _convert_heading(block)
        assert r"\subsection{Methods}" in result

    def test_level_3_heading(self):
        """Test level 3 heading (subsubsection)."""
        block = {"type": "heading", "level": 3, "text": "Data Collection"}
        result = _convert_heading(block)
        assert r"\subsubsection{Data Collection}" in result

    def test_level_4_heading(self):
        """Test level 4 heading (paragraph)."""
        block = {"type": "heading", "level": 4, "text": "Details"}
        result = _convert_heading(block)
        assert r"\paragraph{Details}" in result

    def test_level_5_heading(self):
        """Test level 5 heading (subparagraph)."""
        block = {"type": "heading", "level": 5, "text": "Note"}
        result = _convert_heading(block)
        assert r"\subparagraph{Note}" in result

    def test_default_level(self):
        """Test heading with missing level defaults to section."""
        block = {"type": "heading", "text": "Title"}
        result = _convert_heading(block)
        assert r"\section{Title}" in result

    def test_heading_with_special_chars(self):
        """Test heading with special LaTeX characters."""
        block = {"type": "heading", "level": 1, "text": "Results & Discussion"}
        result = _convert_heading(block)
        assert r"\&" in result


class TestConvertParagraph:
    """Tests for _convert_paragraph helper function."""

    def test_simple_paragraph(self):
        """Test simple paragraph without formatting."""
        block = {"type": "paragraph", "text": "This is a paragraph."}
        result = _convert_paragraph(block)
        assert "This is a paragraph." in result

    def test_paragraph_with_runs(self):
        """Test paragraph with formatted runs."""
        block = {
            "type": "paragraph",
            "runs": [
                {"text": "Normal "},
                {"text": "bold", "bold": True},
                {"text": " text"},
            ],
        }
        result = _convert_paragraph(block)
        assert r"\textbf{bold}" in result
        assert "Normal" in result

    def test_paragraph_with_italic(self):
        """Test paragraph with italic text."""
        block = {
            "type": "paragraph",
            "runs": [{"text": "emphasis", "italic": True}],
        }
        result = _convert_paragraph(block)
        assert r"\textit{emphasis}" in result

    def test_paragraph_with_underline(self):
        """Test paragraph with underlined text."""
        block = {
            "type": "paragraph",
            "runs": [{"text": "underlined", "underline": True}],
        }
        result = _convert_paragraph(block)
        assert r"\underline{underlined}" in result

    def test_paragraph_with_combined_formatting(self):
        """Test paragraph with bold, italic, and underline."""
        block = {
            "type": "paragraph",
            "runs": [
                {"text": "styled", "bold": True, "italic": True, "underline": True}
            ],
        }
        result = _convert_paragraph(block)
        assert r"\textbf" in result
        assert r"\textit" in result
        assert r"\underline" in result


class TestConvertTable:
    """Tests for _convert_table helper function."""

    def test_simple_table(self):
        """Test simple 2x2 table."""
        block = {
            "type": "table",
            "rows": [["A", "B"], ["C", "D"]],
        }
        result = _convert_table(block)
        assert r"\begin{table}" in result
        assert r"\begin{tabular}" in result
        assert r"\end{tabular}" in result
        assert r"\end{table}" in result
        assert "A & B" in result
        assert "C & D" in result

    def test_empty_table(self):
        """Test empty table."""
        block = {"type": "table", "rows": []}
        result = _convert_table(block)
        assert result == ""

    def test_table_with_special_chars(self):
        """Test table with special LaTeX characters."""
        block = {
            "type": "table",
            "rows": [["100%", "$50"]],
        }
        result = _convert_table(block)
        assert r"\%" in result
        assert r"\$" in result

    def test_table_column_spec(self):
        """Test table column specification."""
        block = {
            "type": "table",
            "rows": [["A", "B", "C"]],
        }
        result = _convert_table(block)
        # Should have |c|c|c| for 3 columns
        assert "|c|c|c|" in result


class TestConvertCaption:
    """Tests for _convert_caption helper function."""

    def test_figure_caption(self):
        """Test figure caption."""
        block = {
            "type": "caption",
            "caption_type": "figure",
            "number": "1",
            "caption_text": "A sample figure",
        }
        result = _convert_caption(block)
        assert r"\begin{figure}" in result
        assert r"\caption{A sample figure}" in result
        assert r"\label{fig:1}" in result
        assert r"\end{figure}" in result

    def test_figure_caption_with_image(self):
        """Test figure caption with associated image."""
        block = {
            "type": "caption",
            "caption_type": "figure",
            "number": "2",
            "caption_text": "With image",
            "image_hash": "abc123",
        }
        image_map = {"abc123": "figures/fig_1.png"}
        result = _convert_caption(block, image_map)
        assert r"\includegraphics" in result
        assert "figures/fig_1" in result

    def test_table_caption(self):
        """Test table caption."""
        block = {
            "type": "caption",
            "caption_type": "table",
            "number": "1",
            "caption_text": "Sample table",
        }
        result = _convert_caption(block)
        assert "Table 1" in result
        assert "Sample table" in result

    def test_generic_caption(self):
        """Test generic caption without type."""
        block = {
            "type": "caption",
            "caption_text": "Some caption",
        }
        result = _convert_caption(block)
        assert "Caption:" in result


class TestConvertImage:
    """Tests for _convert_image helper function."""

    def test_image_with_hash(self):
        """Test image conversion with valid hash."""
        block = {"type": "image", "image_hash": "hash123"}
        image_map = {"hash123": "figures/fig_1.png"}
        result = _convert_image(block, image_map)
        assert r"\includegraphics" in result
        assert "figures/fig_1" in result

    def test_image_placeholder(self):
        """Test image placeholder when no hash match."""
        block = {"type": "image", "image_hash": "unknown"}
        result = _convert_image(block, {})
        assert "placeholder" in result.lower()

    def test_image_custom_width(self):
        """Test image with custom width."""
        block = {"type": "image", "image_hash": "hash123", "width": "0.5\\textwidth"}
        image_map = {"hash123": "figures/fig_1.png"}
        result = _convert_image(block, image_map)
        assert "width=0.5" in result


class TestConvertListItem:
    """Tests for _convert_list_item helper function."""

    def test_simple_list_item(self):
        """Test simple list item."""
        block = {"type": "list-item", "text": "First item"}
        result = _convert_list_item(block)
        assert r"\item First item" in result

    def test_list_item_with_special_chars(self):
        """Test list item with special characters."""
        block = {"type": "list-item", "text": "Item with $100"}
        result = _convert_list_item(block)
        assert r"\$" in result


class TestConvertEquation:
    """Tests for _convert_equation helper function."""

    def test_equation_with_latex(self):
        """Test equation with LaTeX content."""
        block = {"type": "equation", "latex": "E = mc^2"}
        result = _convert_equation(block)
        assert r"\begin{equation}" in result
        assert "E = mc^2" in result
        assert r"\end{equation}" in result

    def test_equation_with_text_fallback(self):
        """Test equation with text fallback."""
        block = {"type": "equation", "text": "x + y = z"}
        result = _convert_equation(block)
        assert r"\begin{equation}" in result

    def test_empty_equation(self):
        """Test equation with no content."""
        block = {"type": "equation"}
        result = _convert_equation(block)
        assert result == ""


class TestConvertReference:
    """Tests for _convert_reference_to_latex helper function."""

    def test_numbered_reference(self):
        """Test numbered reference."""
        ref = {"number": 1, "text": "Author, Title, Year"}
        result = _convert_reference_to_latex(ref)
        assert r"\bibitem{ref1}" in result
        assert "Author" in result

    def test_unnumbered_reference(self):
        """Test unnumbered reference."""
        ref = {"text": "Anonymous reference"}
        result = _convert_reference_to_latex(ref)
        assert r"\bibitem{}" in result


class TestGenerateBibtex:
    """Tests for _generate_bibtex helper function."""

    def test_single_reference(self):
        """Test generating bibtex for single reference."""
        refs = [{"number": 1, "text": "Smith, J. Paper Title. 2020"}]
        result = _generate_bibtex(refs)
        assert "@misc{ref1" in result
        assert "note" in result

    def test_multiple_references(self):
        """Test generating bibtex for multiple references."""
        refs = [
            {"number": 1, "text": "First ref"},
            {"number": 2, "text": "Second ref"},
        ]
        result = _generate_bibtex(refs)
        assert "@misc{ref1" in result
        assert "@misc{ref2" in result


class TestParseLatexLog:
    """Tests for _parse_latex_log helper function."""

    def test_parse_error(self):
        """Test parsing LaTeX error from log."""
        log = """
! Undefined control sequence.
l.15 \\badcommand
"""
        errors, warnings = _parse_latex_log(log)
        assert len(errors) >= 1
        assert "Undefined control sequence" in errors[0]

    def test_parse_warning(self):
        """Test parsing LaTeX warning from log."""
        log = """
LaTeX Warning: Reference `fig:1' on page 1 undefined
"""
        errors, warnings = _parse_latex_log(log)
        assert len(warnings) >= 1

    def test_parse_overfull(self):
        """Test parsing overfull hbox warning."""
        log = """
Overfull \\hbox (10.0pt too wide) in paragraph
"""
        errors, warnings = _parse_latex_log(log)
        assert any("Overfull" in w for w in warnings)

    def test_empty_log(self):
        """Test parsing empty log."""
        errors, warnings = _parse_latex_log("")
        assert errors == []
        assert warnings == []


class TestJournalPresets:
    """Tests for JOURNAL_PRESETS configuration."""

    def test_presets_exist(self):
        """Test that journal presets are defined."""
        assert "article" in JOURNAL_PRESETS
        assert "ieee" in JOURNAL_PRESETS
        assert "elsevier" in JOURNAL_PRESETS
        assert "springer" in JOURNAL_PRESETS
        assert "aps" in JOURNAL_PRESETS
        assert "mdpi" in JOURNAL_PRESETS
        assert "acm" in JOURNAL_PRESETS

    def test_preset_structure(self):
        """Test that presets have required keys."""
        for name, preset in JOURNAL_PRESETS.items():
            assert "document_class" in preset
            assert "class_options" in preset
            assert "required_packages" in preset

    def test_ieee_preset(self):
        """Test IEEE preset configuration."""
        preset = JOURNAL_PRESETS["ieee"]
        assert preset["document_class"] == "IEEEtran"
        assert "conference" in preset["class_options"]


class TestCompileResult:
    """Tests for CompileResult dataclass."""

    def test_create_success_result(self):
        """Test creating successful compile result."""
        result = CompileResult(
            success=True,
            pdf_path=Path("/tmp/test.pdf"),
            exit_code=0,
            stdout="Output",
            stderr="",
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.errors == []
        assert result.warnings == []

    def test_create_failure_result(self):
        """Test creating failed compile result."""
        result = CompileResult(
            success=False,
            pdf_path=None,
            exit_code=1,
            stdout="",
            stderr="Error",
            errors=["Compile error"],
        )
        assert result.success is False
        assert result.pdf_path is None
        assert "Compile error" in result.errors

    def test_default_lists(self):
        """Test that errors and warnings default to empty lists."""
        result = CompileResult(
            success=True,
            pdf_path=None,
            exit_code=0,
            stdout="",
            stderr="",
        )
        assert result.errors == []
        assert result.warnings == []


class TestExportTex:
    """Tests for export_tex function."""

    def test_export_minimal_document(self):
        """Test exporting minimal document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path)

            assert result == output_path
            assert output_path.exists()
            content = output_path.read_text()
            assert r"\documentclass{article}" in content
            assert r"\begin{document}" in content
            assert r"\end{document}" in content

    def test_export_with_metadata(self):
        """Test exporting document with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {"title": "Test Title", "author": "Test Author"},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path)

            content = output_path.read_text()
            assert r"\title{Test Title}" in content
            assert r"\author{Test Author}" in content
            assert r"\maketitle" in content

    def test_export_with_heading(self):
        """Test exporting document with heading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [{"type": "heading", "level": 1, "text": "Introduction"}],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path)

            content = output_path.read_text()
            assert r"\section{Introduction}" in content

    def test_export_with_paragraphs(self):
        """Test exporting document with paragraphs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [
                    {"type": "paragraph", "text": "First paragraph."},
                    {"type": "paragraph", "text": "Second paragraph."},
                ],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path)

            content = output_path.read_text()
            assert "First paragraph." in content
            assert "Second paragraph." in content

    def test_export_with_references(self):
        """Test exporting document with references."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {},
                "references": [{"number": 1, "text": "Smith, 2020"}],
                "images": [],
            }
            result = export_tex(doc, output_path)

            content = output_path.read_text()
            assert r"\begin{thebibliography}" in content
            assert r"\bibitem{ref1}" in content
            assert r"\end{thebibliography}" in content

    def test_export_with_bibtex(self):
        """Test exporting document with bibtex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {},
                "references": [{"number": 1, "text": "Reference text"}],
                "images": [],
            }
            result = export_tex(doc, output_path, use_bibtex=True)

            content = output_path.read_text()
            assert r"\bibliography{test}" in content

            bib_path = output_path.with_suffix(".bib")
            assert bib_path.exists()
            bib_content = bib_path.read_text()
            assert "@misc" in bib_content

    def test_export_with_journal_preset(self):
        """Test exporting with journal preset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path, journal_preset="ieee")

            content = output_path.read_text()
            assert r"\documentclass[conference]{IEEEtran}" in content

    def test_export_with_class_options(self):
        """Test exporting with class options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path, class_options=["12pt", "twocolumn"])

            content = output_path.read_text()
            assert "12pt" in content
            assert "twocolumn" in content

    def test_export_with_additional_packages(self):
        """Test exporting with additional packages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path, packages=["booktabs", "siunitx"])

            content = output_path.read_text()
            assert r"\usepackage{booktabs}" in content
            assert r"\usepackage{siunitx}" in content

    def test_export_with_preamble(self):
        """Test exporting with additional preamble."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [],
                "metadata": {},
                "references": [],
                "images": [],
            }
            preamble = r"\newcommand{\mycommand}{test}"
            result = export_tex(doc, output_path, preamble=preamble)

            content = output_path.read_text()
            assert preamble in content

    def test_export_with_list_items(self):
        """Test exporting document with list items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [
                    {
                        "type": "list-item",
                        "text": "First item",
                        "list_type": "unordered",
                    },
                    {
                        "type": "list-item",
                        "text": "Second item",
                        "list_type": "unordered",
                    },
                    {"type": "paragraph", "text": "After list."},
                ],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path)

            content = output_path.read_text()
            assert r"\begin{itemize}" in content
            assert r"\item First item" in content
            assert r"\item Second item" in content
            assert r"\end{itemize}" in content

    def test_export_with_ordered_list(self):
        """Test exporting document with ordered list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            doc = {
                "blocks": [
                    {"type": "list-item", "text": "First", "list_type": "ordered"},
                    {"type": "list-item", "text": "Second", "list_type": "ordered"},
                ],
                "metadata": {},
                "references": [],
                "images": [],
            }
            result = export_tex(doc, output_path)

            content = output_path.read_text()
            assert r"\begin{enumerate}" in content
            assert r"\end{enumerate}" in content


class TestCompileTex:
    """Tests for compile_tex function."""

    def test_compile_nonexistent_file(self):
        """Test compiling nonexistent file."""
        result = compile_tex("/nonexistent/path/file.tex")
        assert result.success is False
        assert (
            "not found" in result.stderr.lower()
            or "not found" in str(result.errors).lower()
        )

    def test_compile_missing_compiler(self):
        """Test compile with missing compiler."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "test.tex"
            tex_path.write_text(
                r"\documentclass{article}\begin{document}Test\end{document}"
            )

            result = compile_tex(tex_path, compiler="nonexistent_compiler")
            assert result.success is False
            assert result.exit_code == 127

    @pytest.mark.skipif(
        shutil.which("pdflatex") is None,
        reason="pdflatex not installed",
    )
    def test_compile_simple_document(self):
        """Test compiling a simple document."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "test.tex"
            tex_content = r"""
\documentclass{article}
\begin{document}
Hello, World!
\end{document}
"""
            tex_path.write_text(tex_content)

            result = compile_tex(tex_path, clean=True)
            assert result.success is True
            assert result.pdf_path is not None
            assert result.pdf_path.exists()

    @pytest.mark.skipif(
        shutil.which("pdflatex") is None,
        reason="pdflatex not installed",
    )
    def test_compile_with_error(self):
        """Test compiling document with LaTeX error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "test.tex"
            tex_content = r"""
\documentclass{article}
\begin{document}
\badcommand
\end{document}
"""
            tex_path.write_text(tex_content)

            result = compile_tex(tex_path, clean=False)
            assert result.success is False
            assert len(result.errors) > 0

    def test_compile_timeout(self):
        """Test compile timeout handling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = Path(tmpdir) / "test.tex"
            tex_path.write_text(
                r"\documentclass{article}\begin{document}Test\end{document}"
            )

            # Mock subprocess.run to raise TimeoutExpired
            with patch("subprocess.run") as mock_run:
                import subprocess

                mock_run.side_effect = subprocess.TimeoutExpired(
                    cmd="pdflatex", timeout=1
                )

                # Also mock shutil.which to return a path
                with patch("shutil.which", return_value="/usr/bin/pdflatex"):
                    result = compile_tex(tex_path, timeout=1)
                    assert result.success is False
                    assert result.exit_code == 124
                    assert "timed out" in result.stderr.lower()


class TestWriteImagesToDir:
    """Tests for _write_images_to_dir helper function."""

    def test_write_single_image(self):
        """Test writing single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "figures"
            image_dir.mkdir()
            tex_parent = Path(tmpdir)

            images = [
                {"hash": "abc123", "extension": ".png", "data": b"\x89PNG\r\n\x1a\n"},
            ]

            result = _write_images_to_dir(images, image_dir, tex_parent)

            assert "abc123" in result
            assert "figures" in result["abc123"]

    def test_skip_duplicate_images(self):
        """Test skipping duplicate images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "figures"
            image_dir.mkdir()
            tex_parent = Path(tmpdir)

            images = [
                {"hash": "abc123", "extension": ".png", "data": b"data1"},
                {"hash": "abc123", "extension": ".png", "data": b"data2"},
            ]

            result = _write_images_to_dir(images, image_dir, tex_parent)

            # Should only have one entry
            assert len(result) == 1

    def test_skip_invalid_images(self):
        """Test skipping images without data or hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = Path(tmpdir) / "figures"
            image_dir.mkdir()
            tex_parent = Path(tmpdir)

            images = [
                {"hash": None, "extension": ".png", "data": b"data"},
                {"hash": "abc", "extension": ".png", "data": None},
            ]

            result = _write_images_to_dir(images, image_dir, tex_parent)

            assert len(result) == 0


class TestBuildLatexDocument:
    """Tests for _build_latex_document helper function."""

    def test_build_minimal_document(self):
        """Test building minimal document."""
        result = _build_latex_document(
            blocks=[],
            metadata={},
            references=[],
            document_class="article",
        )

        assert r"\documentclass{article}" in result
        assert r"\begin{document}" in result
        assert r"\end{document}" in result

    def test_build_with_options(self):
        """Test building with class options."""
        result = _build_latex_document(
            blocks=[],
            metadata={},
            references=[],
            document_class="article",
            class_options=["12pt", "a4paper"],
        )

        assert r"\documentclass[12pt,a4paper]{article}" in result

    def test_default_packages(self):
        """Test that default packages are included."""
        result = _build_latex_document(
            blocks=[],
            metadata={},
            references=[],
            document_class="article",
        )

        assert r"\usepackage[utf8]{inputenc}" in result
        assert r"\usepackage[T1]{fontenc}" in result
        assert r"\usepackage{amsmath}" in result
        assert r"\usepackage{graphicx}" in result
        assert r"\usepackage{hyperref}" in result


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])
