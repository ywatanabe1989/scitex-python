#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata.py
# ----------------------------------------
"""
Image metadata embedding and extraction for research reproducibility.

This module provides functions to embed and extract metadata from image files
(PNG and JPEG formats). Metadata is stored using standard formats:
- PNG: tEXt chunks
- JPEG: EXIF ImageDescription field

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
    Embed metadata into an existing image file.

    Args:
        image_path: Path to the image file (PNG or JPEG)
        metadata: Dictionary containing metadata (must be JSON serializable)

    Raises:
        ValueError: If file format is not supported or metadata is not JSON serializable
        FileNotFoundError: If image file doesn't exist

    Example:
        >>> metadata = {
        ...     'experiment': 'seizure_prediction_001',
        ...     'session': '2024-11-14',
        ...     'analysis': 'PAC'
        ... }
        >>> embed_metadata('result.png', metadata)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Serialize metadata to JSON
    try:
        metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Metadata must be JSON serializable: {e}")

    # Open the image
    img = Image.open(image_path)

    # Handle PNG format
    if image_path.lower().endswith('.png'):
        # Create new PNG info with metadata
        pnginfo = PngInfo()
        pnginfo.add_text("scitex_metadata", metadata_json)

        # Save with metadata
        img.save(image_path, "PNG", pnginfo=pnginfo)

    # Handle JPEG format
    elif image_path.lower().endswith(('.jpg', '.jpeg')):
        try:
            import piexif
        except ImportError:
            raise ImportError(
                "piexif is required for JPEG metadata support. "
                "Install with: pip install piexif"
            )

        # Convert to RGB if necessary (JPEG doesn't support RGBA)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                rgb_img.paste(img, mask=img.split()[-1])
            else:
                rgb_img.paste(img)
            img = rgb_img

        # Create EXIF dict with metadata in ImageDescription field
        exif_dict = {
            "0th": {
                piexif.ImageIFD.ImageDescription: metadata_json.encode('utf-8')
            },
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

        # Save with EXIF metadata
        img.save(image_path, "JPEG", quality=95, exif=exif_bytes)

    else:
        raise ValueError(
            f"Unsupported image format: {image_path}. "
            "Only PNG and JPEG formats are supported."
        )

    img.close()


def read_metadata(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Read metadata from an image file.

    Args:
        image_path: Path to the image file (PNG or JPEG)

    Returns:
        Dictionary containing metadata, or None if no metadata found

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file format is not supported

    Example:
        >>> metadata = read_metadata('result.png')
        >>> print(metadata['experiment'])
        'seizure_prediction_001'
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img = Image.open(image_path)
    metadata = None

    try:
        # Handle PNG format
        if image_path.lower().endswith('.png'):
            # Check for scitex_metadata in PNG info
            if hasattr(img, 'info') and 'scitex_metadata' in img.info:
                metadata_json = img.info['scitex_metadata']
                try:
                    metadata = json.loads(metadata_json)
                except json.JSONDecodeError:
                    # Metadata exists but is not valid JSON
                    metadata = {'raw': metadata_json}

        # Handle JPEG format
        elif image_path.lower().endswith(('.jpg', '.jpeg')):
            try:
                import piexif

                # Load EXIF data
                if 'exif' in img.info:
                    exif_dict = piexif.load(img.info['exif'])

                    # Try to read ImageDescription field
                    if piexif.ImageIFD.ImageDescription in exif_dict.get("0th", {}):
                        description = exif_dict["0th"][piexif.ImageIFD.ImageDescription]

                        # Decode bytes to string
                        if isinstance(description, bytes):
                            description = description.decode('utf-8', errors='ignore')

                        # Try to parse as JSON
                        try:
                            metadata = json.loads(description)
                        except json.JSONDecodeError:
                            # If not JSON, return as raw text
                            metadata = {'raw': description}
            except ImportError:
                pass  # piexif not available, return None
            except Exception:
                pass  # EXIF data corrupted or not readable

        else:
            raise ValueError(
                f"Unsupported image format: {image_path}. "
                "Only PNG and JPEG formats are supported."
            )

    finally:
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
