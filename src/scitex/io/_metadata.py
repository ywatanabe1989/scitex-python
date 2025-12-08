#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata.py
# ----------------------------------------
"""
Image and PDF metadata embedding and extraction for research reproducibility.

This module provides functions to embed and extract metadata from image and PDF files.
Metadata is stored using standard formats:
- PNG: tEXt chunks
- JPEG: EXIF ImageDescription field
- PDF: XMP metadata (industry standard)

The metadata is stored as JSON strings, allowing flexible dictionary structures.
"""

import json
import os
from typing import Any, Dict, Optional

from PIL import Image
from PIL.PngImagePlugin import PngInfo

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------


def embed_metadata(image_path: str, metadata: Dict[str, Any]) -> None:
    """
    Embed metadata into an existing image or PDF file.

    Args:
        image_path: Path to the image/PDF file (PNG, JPEG, or PDF)
        metadata: Dictionary containing metadata (must be JSON serializable)

    Raises:
        ValueError: If file format is not supported or metadata is not JSON serializable
        FileNotFoundError: If file doesn't exist

    Example:
        >>> metadata = {
        ...     'experiment': 'seizure_prediction_001',
        ...     'session': '2024-11-14',
        ...     'analysis': 'PAC'
        ... }
        >>> embed_metadata('result.png', metadata)
        >>> embed_metadata('result.pdf', metadata)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Serialize metadata to JSON
    try:
        metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Metadata must be JSON serializable: {e}")

    # Handle PNG format
    if image_path.lower().endswith(".png"):
        # Open the image
        img = Image.open(image_path)
        # Create new PNG info with metadata
        pnginfo = PngInfo()
        pnginfo.add_text("scitex_metadata", metadata_json)

        # Save with metadata
        img.save(image_path, "PNG", pnginfo=pnginfo)

    # Handle JPEG format
    elif image_path.lower().endswith((".jpg", ".jpeg")):
        # Open the image
        img = Image.open(image_path)

        try:
            import piexif
        except ImportError:
            raise ImportError(
                "piexif is required for JPEG metadata support. "
                "Install with: pip install piexif"
            )

        # Convert to RGB if necessary (JPEG doesn't support RGBA)
        if img.mode in ("RGBA", "LA", "P"):
            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            if img.mode in ("RGBA", "LA"):
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            img = rgb_img

        # Create EXIF dict with metadata in ImageDescription field
        exif_dict = {
            "0th": {piexif.ImageIFD.ImageDescription: metadata_json.encode("utf-8")},
            "Exif": {},
            "GPS": {},
            "1st": {},
        }

        # Try to preserve existing EXIF data
        try:
            existing_exif = piexif.load(img.info.get("exif", b""))
            # Merge with new metadata (prioritize new metadata)
            for ifd in ["Exif", "GPS", "1st"]:
                if ifd in existing_exif:
                    exif_dict[ifd].update(existing_exif[ifd])
        except:
            pass  # If existing EXIF is corrupted, just use new metadata

        exif_bytes = piexif.dump(exif_dict)

        # Save with EXIF metadata (quality=100 for maximum quality)
        img.save(
            image_path,
            "JPEG",
            quality=100,
            subsampling=0,
            optimize=False,
            exif=exif_bytes,
        )

    # Handle PDF format
    elif image_path.lower().endswith(".pdf"):
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

    else:
        raise ValueError(
            f"Unsupported file format: {image_path}. "
            "Only PNG, JPEG, and PDF formats are supported."
        )

    # Close image if it was opened (not for PDF)
    if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
        img.close()


def read_metadata(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata from an image or PDF file.

    Args:
        image_path: Path to the file (PNG, JPEG, or PDF)

    Returns:
        Dictionary containing metadata, or None if no metadata found

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported

    Example:
        >>> metadata = read_metadata('result.png')
        >>> print(metadata['experiment'])
        'seizure_prediction_001'
        >>> metadata = read_metadata('result.pdf')
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Don't open PDF files with PIL
    if not image_path.lower().endswith(".pdf"):
        img = Image.open(image_path)
    metadata = None

    try:
        # Handle PNG format
        if image_path.lower().endswith(".png"):
            # Check for scitex_metadata in PNG info
            if hasattr(img, "info") and "scitex_metadata" in img.info:
                metadata_json = img.info["scitex_metadata"]
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    # Metadata exists but is not valid JSON
                    metadata = {"raw": metadata_json}

        # Handle JPEG format
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            try:
                import piexif

                # Load EXIF data
                if "exif" in img.info:
                    exif_dict = piexif.load(img.info["exif"])

                    # Try to read ImageDescription field
                    if piexif.ImageIFD.ImageDescription in exif_dict.get("0th", {}):
                        description = exif_dict["0th"][piexif.ImageIFD.ImageDescription]

                        # Decode bytes to string
                        if isinstance(description, bytes):
                            description = description.decode("utf-8", errors="ignore")

                        # Try to parse as JSON
                        try:
                            metadata = json.loads(description)
                        except json.JSONDecodeError:
                            # If not JSON, return as raw text
                            metadata = {"raw": description}
            except ImportError:
                pass  # piexif not available, return None
            except Exception:
                pass  # EXIF data corrupted or not readable

        # Handle PDF format
        elif image_path.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader

                reader = PdfReader(image_path)

                # Try to read metadata from PDF Info Dictionary
                if reader.metadata:
                    # Check Subject field for JSON metadata
                    if "/Subject" in reader.metadata:
                        subject = reader.metadata["/Subject"]
                        try:
                            metadata = json.loads(subject)
                        except json.JSONDecodeError:
                            # If not JSON, create metadata dict from available fields
                            metadata = {
                                "title": reader.metadata.get("/Title", ""),
                                "author": reader.metadata.get("/Author", ""),
                                "subject": subject,
                                "creator": reader.metadata.get("/Creator", ""),
                            }
            except ImportError:
                pass  # pypdf not available, return None
            except Exception:
                pass  # PDF metadata corrupted or not readable
            finally:
                # No need to close anything for PDF
                pass

        else:
            raise ValueError(
                f"Unsupported file format: {image_path}. "
                "Only PNG, JPEG, and PDF formats are supported."
            )

    finally:
        # Only close if img was opened (not for PDF)
        if not image_path.lower().endswith(".pdf"):
            img.close()

    return metadata


def has_metadata(image_path: str) -> bool:
    """
    Check if an image file has embedded metadata.

    Args:
        image_path: Path to the image file

    Returns:
        True if metadata exists, False otherwise

    Example:
        >>> if has_metadata('result.png'):
        ...     print(read_metadata('result.png'))
    """
    try:
        metadata = read_metadata(image_path)
        return metadata is not None
    except:
        return False


# EOF
