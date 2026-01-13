#!/usr/bin/env python3
# Timestamp: 2026-01-09
# File: src/scitex/writer/utils/_converters.py
# ----------------------------------------

"""
Conversion utilities for writer module.

Provides:
- CSV <-> LaTeX table conversion
- PDF page to image rendering
- Figure format conversion
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scitex import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CSV <-> LaTeX Table Converters
# =============================================================================


def csv2latex(
    csv_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    escape: bool = True,
    longtable: bool = False,
    index: bool = False,
    column_format: Optional[str] = None,
    **kwargs,
) -> str:
    """
    Convert CSV file to LaTeX table.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file
    output_path : str or Path, optional
        If provided, save LaTeX to this file
    caption : str, optional
        Table caption
    label : str, optional
        Table label for referencing
    escape : bool, default True
        Escape special LaTeX characters
    longtable : bool, default False
        Use longtable environment for multi-page tables
    index : bool, default False
        Include DataFrame index in output
    column_format : str, optional
        LaTeX column format (e.g., 'lcr', 'l|cc|r')
    **kwargs
        Additional arguments passed to pandas.DataFrame.to_latex()

    Returns
    -------
    str
        LaTeX table string

    Examples
    --------
    >>> latex = csv2latex("data.csv", caption="Results", label="tab:results")
    >>> csv2latex("data.csv", "table.tex")  # Save to file
    """
    import pandas as pd

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Build to_latex arguments
    latex_kwargs = {
        "index": index,
        "escape": escape,
        "caption": caption,
        "label": label,
    }

    if longtable:
        latex_kwargs["longtable"] = True

    if column_format:
        latex_kwargs["column_format"] = column_format

    # Merge with user kwargs
    latex_kwargs.update(kwargs)

    # Convert to LaTeX
    latex_content = df.to_latex(**latex_kwargs)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(latex_content)
        logger.info(f"Saved LaTeX table to {output_path}")

    return latex_content


def latex2csv(
    latex_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    table_index: int = 0,
) -> pd.DataFrame:
    """
    Convert LaTeX table to CSV/DataFrame.

    Parameters
    ----------
    latex_path : str or Path
        Path to LaTeX file containing table
    output_path : str or Path, optional
        If provided, save CSV to this file
    table_index : int, default 0
        Which table to extract if multiple tables exist

    Returns
    -------
    pd.DataFrame
        Extracted table as DataFrame

    Examples
    --------
    >>> df = latex2csv("table.tex")
    >>> df = latex2csv("table.tex", "output.csv")
    """
    import pandas as pd

    latex_path = Path(latex_path)
    if not latex_path.exists():
        raise FileNotFoundError(f"LaTeX file not found: {latex_path}")

    with open(latex_path) as f:
        content = f.read()

    # Extract table content (between \begin{tabular} and \end{tabular})
    # Also handle longtable
    patterns = [
        r"\\begin\{tabular\}.*?\n(.*?)\\end\{tabular\}",
        r"\\begin\{longtable\}.*?\n(.*?)\\end\{longtable\}",
    ]

    tables = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        tables.extend(matches)

    if not tables:
        raise ValueError("No table found in LaTeX file")

    if table_index >= len(tables):
        raise IndexError(
            f"Table index {table_index} out of range. Found {len(tables)} tables."
        )

    table_content = tables[table_index]

    # Parse table rows
    rows = []
    for line in table_content.split("\n"):
        line = line.strip()
        if not line or line.startswith("\\"):
            continue
        if "&" in line:
            # Remove trailing \\ and split by &
            line = re.sub(r"\\\\.*$", "", line)
            cells = [cell.strip() for cell in line.split("&")]
            rows.append(cells)

    if not rows:
        raise ValueError("Could not parse table rows")

    # Create DataFrame (first row as header if it looks like headers)
    if len(rows) > 1:
        df = pd.DataFrame(rows[1:], columns=rows[0])
    else:
        df = pd.DataFrame(rows)

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved CSV to {output_path}")

    return df


# =============================================================================
# PDF to Image Rendering
# =============================================================================


def pdf_to_images(
    pdf_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    pages: Optional[Union[int, List[int]]] = None,
    dpi: int = 150,
    format: str = "png",
    prefix: str = "page",
) -> List[Dict[str, Any]]:
    """
    Render PDF pages as images.

    Parameters
    ----------
    pdf_path : str or Path
        Path to PDF file
    output_dir : str or Path, optional
        Directory to save images. If None, uses temp directory.
    pages : int or list of int, optional
        Page(s) to render (0-indexed). If None, renders all pages.
    dpi : int, default 150
        Resolution in DPI
    format : str, default 'png'
        Output format ('png', 'jpg', 'jpeg')
    prefix : str, default 'page'
        Filename prefix

    Returns
    -------
    list of dict
        List of dicts with image info:
        - page: Page number (0-indexed)
        - path: Path to saved image
        - width: Image width in pixels
        - height: Image height in pixels

    Examples
    --------
    >>> # Render first page as thumbnail
    >>> images = pdf_to_images("paper.pdf", pages=0, dpi=72)
    >>> print(images[0]['path'])

    >>> # Render all pages at high resolution
    >>> images = pdf_to_images("paper.pdf", "output/", dpi=300)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF required for PDF to image conversion. "
            "Install with: pip install PyMuPDF"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Setup output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="pdf_images_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize format
    format = format.lower()
    if format == "jpeg":
        format = "jpg"

    # Calculate zoom factor for DPI (default PDF DPI is 72)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    results = []
    doc = fitz.open(pdf_path)

    try:
        if pages is None:
            pages_to_render = range(len(doc))
        elif isinstance(pages, int):
            pages_to_render = [pages]
        else:
            pages_to_render = pages

        for page_num in pages_to_render:
            if page_num < 0 or page_num >= len(doc):
                logger.warning(f"Page {page_num} out of range, skipping")
                continue

            pdf_page = doc[page_num]
            pix = pdf_page.get_pixmap(matrix=matrix)

            # Generate filename
            filename = f"{prefix}_{page_num + 1:03d}.{format}"
            filepath = output_dir / filename

            # Save image
            if format == "png":
                pix.save(str(filepath))
            else:  # jpg
                # Convert to RGB if needed and save as JPEG
                try:
                    import io

                    from PIL import Image

                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    if img.mode in ("RGBA", "LA", "P"):
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "P":
                            img = img.convert("RGBA")
                        if img.mode == "RGBA":
                            background.paste(img, mask=img.split()[-1])
                        else:
                            background.paste(img)
                        img = background
                    elif img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(str(filepath), "JPEG", quality=95)
                except ImportError:
                    # Fallback to PNG if PIL not available
                    filepath = filepath.with_suffix(".png")
                    pix.save(str(filepath))
                    format = "png"

            results.append(
                {
                    "page": page_num,
                    "path": str(filepath),
                    "width": pix.width,
                    "height": pix.height,
                    "dpi": dpi,
                    "format": format,
                }
            )

            logger.debug(f"Rendered page {page_num + 1} to {filepath}")

    finally:
        doc.close()

    logger.info(f"Rendered {len(results)} pages from {pdf_path}")
    return results


def pdf_thumbnail(
    pdf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    page: int = 0,
    width: int = 200,
    format: str = "png",
) -> Dict[str, Any]:
    """
    Generate a thumbnail from a PDF page.

    Parameters
    ----------
    pdf_path : str or Path
        Path to PDF file
    output_path : str or Path, optional
        Path to save thumbnail. If None, auto-generates.
    page : int, default 0
        Page to use for thumbnail (0-indexed)
    width : int, default 200
        Thumbnail width in pixels (height auto-calculated)
    format : str, default 'png'
        Output format ('png', 'jpg')

    Returns
    -------
    dict
        Thumbnail info with path, width, height

    Examples
    --------
    >>> thumb = pdf_thumbnail("paper.pdf")
    >>> print(thumb['path'])
    """
    try:
        import fitz
    except ImportError:
        raise ImportError(
            "PyMuPDF required for PDF thumbnails. Install with: pip install PyMuPDF"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)

    try:
        if page < 0 or page >= len(doc):
            raise IndexError(f"Page {page} out of range. PDF has {len(doc)} pages.")

        pdf_page = doc[page]

        # Calculate zoom to achieve desired width
        page_rect = pdf_page.rect
        zoom = width / page_rect.width
        matrix = fitz.Matrix(zoom, zoom)

        pix = pdf_page.get_pixmap(matrix=matrix)

        # Determine output path
        if output_path is None:
            output_dir = Path(tempfile.mkdtemp(prefix="pdf_thumb_"))
            output_path = output_dir / f"{pdf_path.stem}_thumb.{format}"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save
        pix.save(str(output_path))

        return {
            "path": str(output_path),
            "width": pix.width,
            "height": pix.height,
            "source_page": page,
            "source_pdf": str(pdf_path),
            "format": format,
        }

    finally:
        doc.close()


# =============================================================================
# Figure Handlers
# =============================================================================


def list_figures(
    project_dir: Union[str, Path],
    extensions: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    List all figures in a writer project.

    Parameters
    ----------
    project_dir : str or Path
        Path to writer project directory
    extensions : list of str, optional
        Figure extensions to include. Default: common image formats.

    Returns
    -------
    list of dict
        List of figure info dicts with path, name, size, etc.

    Examples
    --------
    >>> figures = list_figures("my_paper")
    >>> for fig in figures:
    ...     print(fig['name'], fig['size_kb'])
    """
    project_dir = Path(project_dir)
    if not project_dir.exists():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    if extensions is None:
        extensions = [
            ".png",
            ".jpg",
            ".jpeg",
            ".pdf",
            ".eps",
            ".svg",
            ".tif",
            ".tiff",
            ".ppt",
            ".pptx",
        ]

    # Search in common figure locations
    figure_dirs = [
        project_dir / "00_shared" / "figures",
        project_dir / "00_shared" / "figs",
        project_dir / "01_manuscript" / "figures",
        project_dir / "01_manuscript" / "figs",
        project_dir / "02_supplementary" / "figures",
        project_dir / "02_supplementary" / "figs",
    ]

    figures = []
    for fig_dir in figure_dirs:
        if fig_dir.exists():
            for ext in extensions:
                for filepath in fig_dir.glob(f"*{ext}"):
                    stat = filepath.stat()
                    figures.append(
                        {
                            "path": str(filepath),
                            "name": filepath.name,
                            "stem": filepath.stem,
                            "extension": filepath.suffix,
                            "size_bytes": stat.st_size,
                            "size_kb": round(stat.st_size / 1024, 2),
                            "directory": str(fig_dir),
                            "relative_path": str(filepath.relative_to(project_dir)),
                        }
                    )

    # Sort by name
    figures.sort(key=lambda x: x["name"])

    logger.info(f"Found {len(figures)} figures in {project_dir}")
    return figures


def convert_figure(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    dpi: int = 300,
    quality: int = 95,
) -> Dict[str, Any]:
    """
    Convert figure between formats.

    Parameters
    ----------
    input_path : str or Path
        Input figure path
    output_path : str or Path
        Output figure path (format determined by extension)
    dpi : int, default 300
        Resolution for rasterization (PDF/SVG to raster)
    quality : int, default 95
        JPEG quality (1-100)

    Returns
    -------
    dict
        Conversion result with paths and sizes

    Examples
    --------
    >>> convert_figure("fig1.pdf", "fig1.png", dpi=300)
    >>> convert_figure("fig1.png", "fig1.jpg", quality=90)
    """
    from PIL import Image

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_ext = input_path.suffix.lower()
    output_ext = output_path.suffix.lower()

    # Handle PDF input
    if input_ext == ".pdf":
        try:
            import fitz

            doc = fitz.open(input_path)
            page = doc[0]
            zoom = dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)

            if output_ext in [".jpg", ".jpeg"]:
                # Save as PNG first, then convert
                import io

                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(str(output_path), "JPEG", quality=quality)
            else:
                pix.save(str(output_path))

            doc.close()
        except ImportError:
            raise ImportError("PyMuPDF required for PDF conversion")
    else:
        # Standard image conversion with PIL
        img = Image.open(input_path)

        # Handle format-specific conversions
        if output_ext in [".jpg", ".jpeg"]:
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.save(str(output_path), "JPEG", quality=quality)
        else:
            img.save(str(output_path))

    # Get output size
    output_stat = output_path.stat()

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_size_kb": round(input_path.stat().st_size / 1024, 2),
        "output_size_kb": round(output_stat.st_size / 1024, 2),
        "dpi": dpi,
        "quality": quality if output_ext in [".jpg", ".jpeg"] else None,
    }


__all__ = [
    "csv2latex",
    "latex2csv",
    "pdf_to_images",
    "pdf_thumbnail",
    "list_figures",
    "convert_figure",
]

# EOF
