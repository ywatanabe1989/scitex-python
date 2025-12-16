#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_pdf.py

"""PDF metadata embedding using PDF Info Dictionary."""


def embed_metadata_pdf(image_path: str, metadata_json: str, metadata: dict) -> None:
    """
    Embed metadata into a PDF file using PDF Info Dictionary.

    Args:
        image_path: Path to the PDF file.
        metadata_json: JSON string of metadata to embed.
        metadata: Original metadata dict for extracting title/author.

    Raises:
        ImportError: If pypdf is not installed.
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF metadata support. "
            "Install with: pip install pypdf"
        )

    # Read existing PDF
    reader = PdfReader(image_path)
    writer = PdfWriter()

    # Copy all pages
    for page in reader.pages:
        writer.add_page(page)

    # Prepare metadata for PDF Info Dictionary
    pdf_metadata = {
        "/Title": metadata.get("title", ""),
        "/Author": metadata.get("author", ""),
        "/Subject": metadata_json,  # Store full JSON in Subject field
        "/Creator": "SciTeX",
        "/Producer": "SciTeX",
    }

    # Add metadata
    writer.add_metadata(pdf_metadata)

    # Write back to file
    with open(image_path, "wb") as output_file:
        writer.write(output_file)


# EOF
