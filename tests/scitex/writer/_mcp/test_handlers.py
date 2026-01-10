# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_mcp/handlers.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-09
# # File: src/scitex/writer/_mcp.handlers.py
# # ----------------------------------------
# 
# """
# MCP Handler implementations for SciTeX Writer module.
# 
# Provides async handlers for LaTeX manuscript operations:
# - clone_project_handler: Create new writer project
# - compile_manuscript_handler: Compile manuscript PDF
# - compile_supplementary_handler: Compile supplementary PDF
# - compile_revision_handler: Compile revision PDF
# - get_project_info_handler: Get project information
# - get_pdf_handler: Get compiled PDF path
# - list_document_types_handler: List document types
# """
# 
# from __future__ import annotations
# 
# import asyncio
# from pathlib import Path
# from typing import List, Optional, Union
# 
# 
# async def clone_project_handler(
#     project_dir: str,
#     git_strategy: str = "child",
#     branch: Optional[str] = None,
#     tag: Optional[str] = None,
# ) -> dict:
#     """
#     Create a new writer project from template.
# 
#     Parameters
#     ----------
#     project_dir : str
#         Path to create project directory
#     git_strategy : str, optional
#         Git initialization strategy (child, parent, origin, none)
#     branch : str, optional
#         Specific branch to clone
#     tag : str, optional
#         Specific tag to clone
# 
#     Returns
#     -------
#     dict
#         Success status and project path
#     """
#     try:
#         from scitex.writer._clone_writer_project import clone_writer_project
# 
#         # Handle git_strategy='none'
#         git_strat = None if git_strategy == "none" else git_strategy
# 
#         # Run clone in executor (blocking operation)
#         loop = asyncio.get_event_loop()
#         success = await loop.run_in_executor(
#             None,
#             lambda: clone_writer_project(
#                 project_dir=project_dir,
#                 git_strategy=git_strat,
#                 branch=branch,
#                 tag=tag,
#             ),
#         )
# 
#         if success:
#             resolved_path = Path(project_dir)
#             if not resolved_path.is_absolute():
#                 resolved_path = Path.cwd() / resolved_path
# 
#             return {
#                 "success": True,
#                 "project_path": str(resolved_path),
#                 "git_strategy": git_strategy,
#                 "structure": {
#                     "00_shared": "Shared resources (figures, bibliography)",
#                     "01_manuscript": "Main manuscript",
#                     "02_supplementary": "Supplementary materials",
#                     "03_revision": "Revision documents",
#                     "scripts": "Compilation scripts",
#                 },
#                 "message": f"Successfully created writer project at {resolved_path}",
#             }
#         else:
#             return {
#                 "success": False,
#                 "error": "Failed to clone writer project",
#                 "project_dir": project_dir,
#             }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def compile_manuscript_handler(
#     project_dir: str,
#     timeout: int = 300,
#     no_figs: bool = False,
#     ppt2tif: bool = False,
#     crop_tif: bool = False,
#     quiet: bool = False,
#     verbose: bool = False,
#     force: bool = False,
# ) -> dict:
#     """
#     Compile manuscript to PDF.
# 
#     Parameters
#     ----------
#     project_dir : str
#         Path to writer project directory
#     timeout : int, optional
#         Maximum compilation time in seconds
#     no_figs : bool, optional
#         Exclude figures for quick compilation
#     ppt2tif : bool, optional
#         Convert PowerPoint files to TIF format (WSL only)
#     crop_tif : bool, optional
#         Crop TIF images to remove whitespace
#     quiet : bool, optional
#         Suppress detailed LaTeX logs
#     verbose : bool, optional
#         Show verbose LaTeX output
#     force : bool, optional
#         Force recompilation, ignore cache
# 
#     Returns
#     -------
#     dict
#         Success status, PDF path, and compilation details
#     """
#     try:
#         from scitex.writer._compile import compile_manuscript
# 
#         project_path = Path(project_dir)
#         if not project_path.is_absolute():
#             project_path = Path.cwd() / project_path
# 
#         # Run compilation in executor
#         loop = asyncio.get_event_loop()
# 
#         def do_compile():
#             return compile_manuscript(
#                 project_path,
#                 timeout=timeout,
#                 no_figs=no_figs,
#                 ppt2tif=ppt2tif,
#                 crop_tif=crop_tif,
#                 quiet=quiet,
#                 verbose=verbose,
#                 force=force,
#             )
# 
#         result = await loop.run_in_executor(None, do_compile)
# 
#         if result.success:
#             return {
#                 "success": True,
#                 "output_pdf": str(result.output_pdf) if result.output_pdf else None,
#                 "exit_code": result.exit_code,
#                 "warnings": result.warnings[:10] if result.warnings else [],
#                 "message": "Manuscript compiled successfully",
#             }
#         else:
#             return {
#                 "success": False,
#                 "exit_code": result.exit_code,
#                 "errors": result.errors[:10] if result.errors else [],
#                 "warnings": result.warnings[:10] if result.warnings else [],
#                 "error": f"Compilation failed with exit code {result.exit_code}",
#             }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def compile_supplementary_handler(
#     project_dir: str,
#     timeout: int = 300,
#     no_figs: bool = False,
#     ppt2tif: bool = False,
#     crop_tif: bool = False,
#     quiet: bool = False,
# ) -> dict:
#     """
#     Compile supplementary materials to PDF.
# 
#     Parameters
#     ----------
#     project_dir : str
#         Path to writer project directory
#     timeout : int, optional
#         Maximum compilation time in seconds
#     no_figs : bool, optional
#         Exclude figures for quick compilation
#     ppt2tif : bool, optional
#         Convert PowerPoint files to TIF format (WSL only)
#     crop_tif : bool, optional
#         Crop TIF images to remove whitespace
#     quiet : bool, optional
#         Suppress detailed LaTeX logs
# 
#     Returns
#     -------
#     dict
#         Success status, PDF path, and compilation details
#     """
#     try:
#         from scitex.writer._compile import compile_supplementary
# 
#         project_path = Path(project_dir)
#         if not project_path.is_absolute():
#             project_path = Path.cwd() / project_path
# 
#         loop = asyncio.get_event_loop()
# 
#         def do_compile():
#             return compile_supplementary(
#                 project_path,
#                 timeout=timeout,
#                 no_figs=no_figs,
#                 ppt2tif=ppt2tif,
#                 crop_tif=crop_tif,
#                 quiet=quiet,
#             )
# 
#         result = await loop.run_in_executor(None, do_compile)
# 
#         if result.success:
#             return {
#                 "success": True,
#                 "output_pdf": str(result.output_pdf) if result.output_pdf else None,
#                 "exit_code": result.exit_code,
#                 "warnings": result.warnings[:10] if result.warnings else [],
#                 "message": "Supplementary materials compiled successfully",
#             }
#         else:
#             return {
#                 "success": False,
#                 "exit_code": result.exit_code,
#                 "errors": result.errors[:10] if result.errors else [],
#                 "warnings": result.warnings[:10] if result.warnings else [],
#                 "error": f"Compilation failed with exit code {result.exit_code}",
#             }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def compile_revision_handler(
#     project_dir: str,
#     track_changes: bool = False,
#     timeout: int = 300,
# ) -> dict:
#     """
#     Compile revision document to PDF.
# 
#     Parameters
#     ----------
#     project_dir : str
#         Path to writer project directory
#     track_changes : bool, optional
#         Enable change tracking in output
#     timeout : int, optional
#         Maximum compilation time in seconds
# 
#     Returns
#     -------
#     dict
#         Success status, PDF path, and compilation details
#     """
#     try:
#         from scitex.writer._compile import compile_revision
# 
#         project_path = Path(project_dir)
#         if not project_path.is_absolute():
#             project_path = Path.cwd() / project_path
# 
#         loop = asyncio.get_event_loop()
# 
#         def do_compile():
#             return compile_revision(
#                 project_path,
#                 track_changes=track_changes,
#                 timeout=timeout,
#             )
# 
#         result = await loop.run_in_executor(None, do_compile)
# 
#         if result.success:
#             return {
#                 "success": True,
#                 "output_pdf": str(result.output_pdf) if result.output_pdf else None,
#                 "exit_code": result.exit_code,
#                 "track_changes": track_changes,
#                 "warnings": result.warnings[:10] if result.warnings else [],
#                 "message": "Revision compiled successfully",
#             }
#         else:
#             return {
#                 "success": False,
#                 "exit_code": result.exit_code,
#                 "errors": result.errors[:10] if result.errors else [],
#                 "warnings": result.warnings[:10] if result.warnings else [],
#                 "error": f"Compilation failed with exit code {result.exit_code}",
#             }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def get_project_info_handler(project_dir: str) -> dict:
#     """
#     Get writer project information.
# 
#     Parameters
#     ----------
#     project_dir : str
#         Path to writer project directory
# 
#     Returns
#     -------
#     dict
#         Project structure and status information
#     """
#     try:
#         from scitex.writer import Writer
# 
#         project_path = Path(project_dir)
#         if not project_path.is_absolute():
#             project_path = Path.cwd() / project_path
# 
#         loop = asyncio.get_event_loop()
# 
#         def get_info():
#             writer = Writer(project_path)
# 
#             # Check for compiled PDFs
#             pdfs = {}
#             for doc_type in ["manuscript", "supplementary", "revision"]:
#                 pdf = writer.get_pdf(doc_type)
#                 pdfs[doc_type] = str(pdf) if pdf else None
# 
#             return {
#                 "project_name": writer.project_name,
#                 "project_dir": str(writer.project_dir.absolute()),
#                 "git_root": str(writer.git_root) if writer.git_root else None,
#                 "documents": {
#                     "shared": str(writer.shared.root),
#                     "manuscript": str(writer.manuscript.root),
#                     "supplementary": str(writer.supplementary.root),
#                     "revision": str(writer.revision.root),
#                     "scripts": str(writer.scripts.root),
#                 },
#                 "compiled_pdfs": pdfs,
#             }
# 
#         info = await loop.run_in_executor(None, get_info)
# 
#         return {
#             "success": True,
#             **info,
#         }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def get_pdf_handler(
#     project_dir: str,
#     doc_type: str = "manuscript",
# ) -> dict:
#     """
#     Get path to compiled PDF.
# 
#     Parameters
#     ----------
#     project_dir : str
#         Path to writer project directory
#     doc_type : str, optional
#         Document type (manuscript, supplementary, revision)
# 
#     Returns
#     -------
#     dict
#         PDF path if exists
#     """
#     try:
#         from scitex.writer import Writer
# 
#         project_path = Path(project_dir)
#         if not project_path.is_absolute():
#             project_path = Path.cwd() / project_path
# 
#         loop = asyncio.get_event_loop()
# 
#         def get_pdf():
#             writer = Writer(project_path)
#             return writer.get_pdf(doc_type)
# 
#         pdf = await loop.run_in_executor(None, get_pdf)
# 
#         if pdf:
#             return {
#                 "success": True,
#                 "exists": True,
#                 "doc_type": doc_type,
#                 "pdf_path": str(pdf),
#             }
#         else:
#             return {
#                 "success": True,
#                 "exists": False,
#                 "doc_type": doc_type,
#                 "pdf_path": None,
#                 "message": f"No compiled PDF found for {doc_type}",
#             }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def list_document_types_handler() -> dict:
#     """
#     List available document types in a writer project.
# 
#     Returns
#     -------
#     dict
#         List of document types with descriptions
#     """
#     return {
#         "success": True,
#         "document_types": [
#             {
#                 "id": "manuscript",
#                 "name": "Manuscript",
#                 "description": "Main manuscript document",
#                 "directory": "01_manuscript",
#                 "compile_command": "compile_manuscript",
#             },
#             {
#                 "id": "supplementary",
#                 "name": "Supplementary Materials",
#                 "description": "Supplementary information, figures, and tables",
#                 "directory": "02_supplementary",
#                 "compile_command": "compile_supplementary",
#             },
#             {
#                 "id": "revision",
#                 "name": "Revision",
#                 "description": "Revision document with optional change tracking",
#                 "directory": "03_revision",
#                 "compile_command": "compile_revision",
#                 "supports_track_changes": True,
#             },
#         ],
#         "shared_directory": {
#             "id": "shared",
#             "name": "Shared Resources",
#             "description": "Figures, bibliography, and shared assets",
#             "directory": "00_shared",
#         },
#     }
# 
# 
# async def csv2latex_handler(
#     csv_path: str,
#     output_path: Optional[str] = None,
#     caption: Optional[str] = None,
#     label: Optional[str] = None,
#     longtable: bool = False,
# ) -> dict:
#     """
#     Convert CSV file to LaTeX table.
# 
#     Parameters
#     ----------
#     csv_path : str
#         Path to CSV file
#     output_path : str, optional
#         Output path for LaTeX file
#     caption : str, optional
#         Table caption
#     label : str, optional
#         Table label for referencing
#     longtable : bool, optional
#         Use longtable for multi-page tables
# 
#     Returns
#     -------
#     dict
#         LaTeX content and output path
#     """
#     try:
#         from scitex.writer.utils import csv2latex
# 
#         loop = asyncio.get_event_loop()
#         latex_content = await loop.run_in_executor(
#             None,
#             lambda: csv2latex(
#                 csv_path=csv_path,
#                 output_path=output_path,
#                 caption=caption,
#                 label=label,
#                 longtable=longtable,
#             ),
#         )
# 
#         return {
#             "success": True,
#             "latex_content": latex_content,
#             "output_path": output_path,
#             "message": f"Converted {csv_path} to LaTeX table",
#         }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def latex2csv_handler(
#     latex_path: str,
#     output_path: Optional[str] = None,
#     table_index: int = 0,
# ) -> dict:
#     """
#     Convert LaTeX table to CSV.
# 
#     Parameters
#     ----------
#     latex_path : str
#         Path to LaTeX file containing table
#     output_path : str, optional
#         Output path for CSV file
#     table_index : int, optional
#         Index of table to extract
# 
#     Returns
#     -------
#     dict
#         CSV content preview and output path
#     """
#     try:
#         from scitex.writer.utils import latex2csv
# 
#         loop = asyncio.get_event_loop()
#         df = await loop.run_in_executor(
#             None,
#             lambda: latex2csv(
#                 latex_path=latex_path,
#                 output_path=output_path,
#                 table_index=table_index,
#             ),
#         )
# 
#         return {
#             "success": True,
#             "rows": len(df),
#             "columns": list(df.columns),
#             "preview": df.head(5).to_dict(),
#             "output_path": output_path,
#             "message": f"Converted LaTeX table to CSV ({len(df)} rows)",
#         }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def pdf_to_images_handler(
#     pdf_path: str,
#     output_dir: Optional[str] = None,
#     pages: Optional[Union[int, List[int]]] = None,
#     dpi: int = 150,
#     format: str = "png",
# ) -> dict:
#     """
#     Render PDF pages as images.
# 
#     Parameters
#     ----------
#     pdf_path : str
#         Path to PDF file
#     output_dir : str, optional
#         Output directory for images
#     pages : int or list of int, optional
#         Page(s) to render (0-indexed). If None, renders all.
#     dpi : int, optional
#         Resolution in DPI
#     format : str, optional
#         Output format (png, jpg)
# 
#     Returns
#     -------
#     dict
#         List of rendered images with paths
#     """
#     try:
#         from scitex.writer.utils import pdf_to_images
# 
#         loop = asyncio.get_event_loop()
#         images = await loop.run_in_executor(
#             None,
#             lambda: pdf_to_images(
#                 pdf_path=pdf_path,
#                 output_dir=output_dir,
#                 pages=pages,
#                 dpi=dpi,
#                 format=format,
#             ),
#         )
# 
#         return {
#             "success": True,
#             "images": images,
#             "count": len(images),
#             "message": f"Rendered {len(images)} page(s) from {pdf_path}",
#         }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def list_figures_handler(
#     project_dir: str,
#     extensions: Optional[list] = None,
# ) -> dict:
#     """
#     List figures in writer project.
# 
#     Parameters
#     ----------
#     project_dir : str
#         Path to writer project directory
#     extensions : list, optional
#         File extensions to include
# 
#     Returns
#     -------
#     dict
#         List of figure info
#     """
#     try:
#         from scitex.writer.utils import list_figures
# 
#         loop = asyncio.get_event_loop()
#         figures = await loop.run_in_executor(
#             None,
#             lambda: list_figures(
#                 project_dir=project_dir,
#                 extensions=extensions,
#             ),
#         )
# 
#         return {
#             "success": True,
#             "figures": figures,
#             "count": len(figures),
#             "message": f"Found {len(figures)} figures in {project_dir}",
#         }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# async def convert_figure_handler(
#     input_path: str,
#     output_path: str,
#     dpi: int = 300,
#     quality: int = 95,
# ) -> dict:
#     """
#     Convert figure between formats.
# 
#     Parameters
#     ----------
#     input_path : str
#         Input figure path
#     output_path : str
#         Output figure path
#     dpi : int, optional
#         Resolution for PDF rasterization
#     quality : int, optional
#         JPEG quality (1-100)
# 
#     Returns
#     -------
#     dict
#         Conversion result with paths
#     """
#     try:
#         from scitex.writer.utils import convert_figure
# 
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None,
#             lambda: convert_figure(
#                 input_path=input_path,
#                 output_path=output_path,
#                 dpi=dpi,
#                 quality=quality,
#             ),
#         )
# 
#         return {
#             "success": True,
#             **result,
#             "message": f"Converted {input_path} to {output_path}",
#         }
# 
#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e),
#         }
# 
# 
# __all__ = [
#     "clone_project_handler",
#     "compile_manuscript_handler",
#     "compile_supplementary_handler",
#     "compile_revision_handler",
#     "get_project_info_handler",
#     "get_pdf_handler",
#     "list_document_types_handler",
#     "csv2latex_handler",
#     "latex2csv_handler",
#     "pdf_to_images_handler",
#     "list_figures_handler",
#     "convert_figure_handler",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_mcp/handlers.py
# --------------------------------------------------------------------------------
