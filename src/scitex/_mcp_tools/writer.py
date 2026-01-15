#!/usr/bin/env python3
# Timestamp: 2026-01-15
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/writer.py
"""Writer module tools for FastMCP unified server."""

from __future__ import annotations

import json
from typing import List, Optional, Union


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def register_writer_tools(mcp) -> None:
    """Register writer tools with FastMCP server."""

    @mcp.tool()
    async def writer_clone_project(
        project_dir: str,
        git_strategy: str = "child",
        branch: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> str:
        """[writer] Create a new LaTeX manuscript project from template."""
        from scitex.writer._mcp.handlers import clone_project_handler

        result = await clone_project_handler(
            project_dir=project_dir,
            git_strategy=git_strategy,
            branch=branch,
            tag=tag,
        )
        return _json(result)

    @mcp.tool()
    async def writer_compile_manuscript(
        project_dir: str,
        timeout: int = 300,
        no_figs: bool = False,
        ppt2tif: bool = False,
        crop_tif: bool = False,
        quiet: bool = False,
        verbose: bool = False,
        force: bool = False,
    ) -> str:
        """[writer] Compile manuscript LaTeX document to PDF."""
        from scitex.writer._mcp.handlers import compile_manuscript_handler

        result = await compile_manuscript_handler(
            project_dir=project_dir,
            timeout=timeout,
            no_figs=no_figs,
            ppt2tif=ppt2tif,
            crop_tif=crop_tif,
            quiet=quiet,
            verbose=verbose,
            force=force,
        )
        return _json(result)

    @mcp.tool()
    async def writer_compile_supplementary(
        project_dir: str,
        timeout: int = 300,
        no_figs: bool = False,
        ppt2tif: bool = False,
        crop_tif: bool = False,
        quiet: bool = False,
    ) -> str:
        """[writer] Compile supplementary materials LaTeX document to PDF."""
        from scitex.writer._mcp.handlers import compile_supplementary_handler

        result = await compile_supplementary_handler(
            project_dir=project_dir,
            timeout=timeout,
            no_figs=no_figs,
            ppt2tif=ppt2tif,
            crop_tif=crop_tif,
            quiet=quiet,
        )
        return _json(result)

    @mcp.tool()
    async def writer_compile_revision(
        project_dir: str,
        track_changes: bool = False,
        timeout: int = 300,
    ) -> str:
        """[writer] Compile revision document to PDF with optional change tracking."""
        from scitex.writer._mcp.handlers import compile_revision_handler

        result = await compile_revision_handler(
            project_dir=project_dir,
            track_changes=track_changes,
            timeout=timeout,
        )
        return _json(result)

    @mcp.tool()
    async def writer_get_project_info(project_dir: str) -> str:
        """[writer] Get writer project structure and status information."""
        from scitex.writer._mcp.handlers import get_project_info_handler

        result = await get_project_info_handler(project_dir=project_dir)
        return _json(result)

    @mcp.tool()
    async def writer_get_pdf(
        project_dir: str,
        doc_type: str = "manuscript",
    ) -> str:
        """[writer] Get path to compiled PDF for a document type."""
        from scitex.writer._mcp.handlers import get_pdf_handler

        result = await get_pdf_handler(
            project_dir=project_dir,
            doc_type=doc_type,
        )
        return _json(result)

    @mcp.tool()
    async def writer_list_document_types() -> str:
        """[writer] List available document types in a writer project."""
        from scitex.writer._mcp.handlers import list_document_types_handler

        result = await list_document_types_handler()
        return _json(result)

    @mcp.tool()
    async def writer_csv_to_latex(
        csv_path: str,
        output_path: Optional[str] = None,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        longtable: bool = False,
    ) -> str:
        """[writer] Convert CSV file to LaTeX table format."""
        from scitex.writer._mcp.handlers import csv2latex_handler

        result = await csv2latex_handler(
            csv_path=csv_path,
            output_path=output_path,
            caption=caption,
            label=label,
            longtable=longtable,
        )
        return _json(result)

    @mcp.tool()
    async def writer_latex_to_csv(
        latex_path: str,
        output_path: Optional[str] = None,
        table_index: int = 0,
    ) -> str:
        """[writer] Convert LaTeX table to CSV format."""
        from scitex.writer._mcp.handlers import latex2csv_handler

        result = await latex2csv_handler(
            latex_path=latex_path,
            output_path=output_path,
            table_index=table_index,
        )
        return _json(result)

    @mcp.tool()
    async def writer_pdf_to_images(
        pdf_path: str,
        output_dir: Optional[str] = None,
        pages: Optional[Union[int, List[int]]] = None,
        dpi: int = 150,
        format: str = "png",
    ) -> str:
        """[writer] Render PDF pages as images."""
        from scitex.writer._mcp.handlers import pdf_to_images_handler

        result = await pdf_to_images_handler(
            pdf_path=pdf_path,
            output_dir=output_dir,
            pages=pages,
            dpi=dpi,
            format=format,
        )
        return _json(result)

    @mcp.tool()
    async def writer_list_figures(
        project_dir: str,
        extensions: Optional[List[str]] = None,
    ) -> str:
        """[writer] List all figures in a writer project directory."""
        from scitex.writer._mcp.handlers import list_figures_handler

        result = await list_figures_handler(
            project_dir=project_dir,
            extensions=extensions,
        )
        return _json(result)

    @mcp.tool()
    async def writer_convert_figure(
        input_path: str,
        output_path: str,
        dpi: int = 300,
        quality: int = 95,
    ) -> str:
        """[writer] Convert figure between formats (e.g., PDF to PNG, PNG to JPG)."""
        from scitex.writer._mcp.handlers import convert_figure_handler

        result = await convert_figure_handler(
            input_path=input_path,
            output_path=output_path,
            dpi=dpi,
            quality=quality,
        )
        return _json(result)


# EOF
