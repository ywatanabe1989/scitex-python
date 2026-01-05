# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_crop.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2025-11-19 15:30:00 (ywatanabe)"
# # File: ./src/scitex/plt/utils/_crop.py
# 
# """
# Auto-crop figures to content area with optional margin.
# 
# This utility automatically detects the content area of saved figures
# and crops them, removing excess whitespace while preserving a small margin.
# """
# 
# import os
# import numpy as np
# from typing import Optional, Tuple
# from PIL import Image
# 
# 
# def find_content_area(image_path: str) -> Tuple[int, int, int, int]:
#     """
#     Find the bounding box of the content area in an image.
# 
#     Parameters
#     ----------
#     image_path : str
#         Path to the image file
# 
#     Returns
#     -------
#     tuple
#         (left, upper, right, lower) bounding box coordinates
# 
#     Raises
#     ------
#     FileNotFoundError
#         If the image cannot be read
#     """
#     # Read the image with PIL
#     img = Image.open(image_path)
# 
#     # Convert to numpy array for processing
#     img_array = np.array(img)
# 
#     # Check if image has alpha channel (RGBA)
#     if len(img_array.shape) == 3 and img_array.shape[2] == 4:
#         # Use alpha channel to find content (non-transparent pixels)
#         alpha = img_array[:, :, 3]
#         # Find non-transparent pixels
#         rows = np.any(alpha > 0, axis=1)
#         cols = np.any(alpha > 0, axis=0)
#     else:
#         # For RGB images, detect background color from corners and find non-background pixels
#         if len(img_array.shape) == 3:
#             # Sample background color from corners (more robust than assuming white)
#             h, w = img_array.shape[:2]
#             corners = [
#                 img_array[0, 0],  # top-left
#                 img_array[0, w - 1],  # top-right
#                 img_array[h - 1, 0],  # bottom-left
#                 img_array[h - 1, w - 1],  # bottom-right
#             ]
#             # Use median of corners as background color (robust to one corner having content)
#             bg_color = np.median(corners, axis=0).astype(np.uint8)
# 
#             # Find pixels that differ significantly from background (threshold: 10 per channel)
#             diff = np.abs(img_array.astype(np.int16) - bg_color.astype(np.int16))
#             is_content = np.any(diff > 10, axis=2)
#         else:
#             # Grayscale: detect background from corners
#             h, w = img_array.shape
#             corners = [
#                 img_array[0, 0],
#                 img_array[0, w - 1],
#                 img_array[h - 1, 0],
#                 img_array[h - 1, w - 1],
#             ]
#             bg_value = np.median(corners)
#             is_content = np.abs(img_array.astype(np.int16) - bg_value) > 10
# 
#         rows = np.any(is_content, axis=1)
#         cols = np.any(is_content, axis=0)
# 
#     # Find the bounding box
#     if np.any(rows) and np.any(cols):
#         y_min, y_max = np.where(rows)[0][[0, -1]]
#         x_min, x_max = np.where(cols)[0][[0, -1]]
#         return x_min, y_min, x_max + 1, y_max + 1
#     else:
#         # If no content found, return the whole image
#         return 0, 0, img.width, img.height
# 
# 
# def crop(
#     input_path: str,
#     output_path: Optional[str] = None,
#     margin: int = 12,
#     overwrite: bool = False,
#     verbose: bool = False,
#     return_offset: bool = False,
#     crop_box: Optional[Tuple[int, int, int, int]] = None,
# ) -> str:
#     """
#     Crop a figure image to its content area with a specified margin.
# 
#     This function is designed for publication-ready figures created with
#     large margins. It automatically detects the content and crops to it,
#     leaving a small margin around the content.
# 
#     Parameters
#     ----------
#     input_path : str
#         Path to the input image (PNG, TIFF, etc.)
#     output_path : str, optional
#         Path to save the cropped image. If None and overwrite=True,
#         overwrites the input. If None and overwrite=False, adds '_cropped' suffix.
#     margin : int, optional
#         Margin in pixels to add around the content area (default: 12, ~1mm at 300 DPI).
#         Only used when crop_box is None (auto-detection mode).
#     overwrite : bool, optional
#         Whether to overwrite the input file (default: False)
#     verbose : bool, optional
#         Whether to print detailed information (default: False)
#     return_offset : bool, optional
#         If True, also return the crop offset (left, upper) for metadata adjustment.
#         Default is False.
#     crop_box : tuple, optional
#         Explicit crop coordinates (left, upper, right, lower). If provided,
#         skips auto-detection and uses these exact coordinates for cropping.
#         This is useful for applying the same crop to multiple images (e.g., hitmap).
# 
#     Returns
#     -------
#     str or tuple
#         Path to the saved cropped image. If return_offset=True, returns
#         (path, offset_dict) where offset_dict has keys 'left', 'upper', 'right', 'lower'
#         representing the crop boundaries.
# 
#     Raises
#     ------
#     FileNotFoundError
#         If the input image cannot be read
# 
#     Examples
#     --------
#     >>> import scitex as stx
#     >>> fig, ax = stx.plt.subplots(**stx.plt.presets.SCITEX_STYLE)
#     >>> ax.plot([1, 2, 3], [1, 2, 3])
#     >>> stx.io.save(fig, "figure.png", dpi=300, transparent=True)
#     >>> stx.plt.crop("figure.png", "figure_cropped.png")  # 1mm margin (12px at 300 DPI)
# 
#     >>> # Or crop in place
#     >>> stx.plt.crop("figure.png", overwrite=True)
# 
#     >>> # Apply explicit crop coordinates (e.g., for hitmap)
#     >>> _, offset = stx.plt.crop("figure.png", return_offset=True)
#     >>> crop_box = (offset['left'], offset['upper'], offset['right'], offset['lower'])
#     >>> stx.plt.crop("hitmap.png", crop_box=crop_box)
#     """
#     # Determine output path
#     if output_path is None:
#         if overwrite:
#             output_path = input_path
#         else:
#             # Add '_cropped' suffix before extension
#             base, ext = os.path.splitext(input_path)
#             output_path = f"{base}_cropped{ext}"
# 
#     # Read the image with PIL (preserves alpha channel and DPI metadata)
#     img = Image.open(input_path)
# 
#     original_width, original_height = img.size
# 
#     if verbose:
#         print(f"Original image dimensions: {original_width}x{original_height}")
#         if "dpi" in img.info:
#             print(f"Original DPI: {img.info['dpi']}")
# 
#     # Use explicit crop_box if provided, otherwise auto-detect
#     if crop_box is not None:
#         # Use explicit crop coordinates (no margin adjustment - use as-is)
#         left, upper, right, lower = crop_box
#         if verbose:
#             print(f"Using explicit crop_box: left={left}, upper={upper}, right={right}, lower={lower}")
#     else:
#         # Find the content area (returns left, upper, right, lower)
#         left, upper, right, lower = find_content_area(input_path)
# 
#         if verbose:
#             print(
#                 f"Content area detected at: left={left}, upper={upper}, right={right}, lower={lower}"
#             )
# 
#         # Calculate the coordinates with margin, clamping to the image boundaries
#         left = max(left - margin, 0)
#         upper = max(upper - margin, 0)
#         right = min(right + margin, img.width)
#         lower = min(lower + margin, img.height)
# 
#     if verbose:
#         print(f"Cropping to: left={left}, upper={upper}, right={right}, lower={lower}")
#         print(f"New dimensions: {right - left}x{lower - upper}")
# 
#     # Crop the image using PIL (lossless operation)
#     cropped_img = img.crop((left, upper, right, lower))
# 
#     # Preserve DPI metadata and save with maximum quality
#     save_kwargs = {}
# 
#     # Preserve DPI if it exists
#     if "dpi" in img.info:
#         save_kwargs["dpi"] = img.info["dpi"]
# 
#     ext = os.path.splitext(output_path)[1].lower()
#     if ext in [".png"]:
#         # PNG: lossless compression, preserve all metadata including text chunks
#         save_kwargs["compress_level"] = (
#             0  # No compression for maximum quality and speed
#         )
#         save_kwargs["optimize"] = False
# 
#         # Preserve PNG text chunks (tEXt, zTXt, iTXt) where scitex metadata is stored
#         from PIL import PngImagePlugin
# 
#         pnginfo = PngImagePlugin.PngInfo()
# 
#         # Copy all text chunks from original image
#         for key, value in img.info.items():
#             if isinstance(value, (str, bytes)):
#                 try:
#                     pnginfo.add_text(key, value)
#                 except Exception:
#                     pass  # Skip if can't add this chunk
# 
#         save_kwargs["pnginfo"] = pnginfo
#     elif ext in [".jpg", ".jpeg"]:
#         # JPEG: maximum quality for daily use (quality=100, no compression artifacts)
#         save_kwargs["quality"] = 100
#         save_kwargs["subsampling"] = 0  # No chroma subsampling (best quality)
#         save_kwargs["optimize"] = False
# 
#         # Preserve EXIF metadata where scitex metadata is stored
#         try:
#             import piexif
# 
#             # Try to load existing EXIF data
#             if "exif" in img.info:
#                 # piexif.load() reads the EXIF bytes
#                 exif_dict = piexif.load(img.info["exif"])
#                 # Re-dump to bytes for saving
#                 exif_bytes = piexif.dump(exif_dict)
#                 save_kwargs["exif"] = exif_bytes
#         except ImportError:
#             # piexif not available, try PIL's built-in EXIF
#             if hasattr(img, "getexif"):
#                 exif = img.getexif()
#                 if exif:
#                     save_kwargs["exif"] = exif
#         except Exception:
#             # If EXIF reading fails, continue without it
#             pass
# 
#     # Save the cropped image
#     cropped_img.save(output_path, **save_kwargs)
# 
#     # Calculate space saved
#     final_width, final_height = cropped_img.size
#     area_reduction = 1 - (
#         (final_width * final_height) / (original_width * original_height)
#     )
#     area_reduction_pct = area_reduction * 100
# 
#     if verbose:
#         print(f"Image processed: {input_path}")
#         print(
#             f"Size changed from {original_width}x{original_height} to {final_width}x{final_height}"
#         )
#         print(f"Saved {area_reduction_pct:.1f}% of the original area")
#         if output_path != input_path:
#             print(f"Saved to: {output_path}")
# 
#     if return_offset:
#         offset = {
#             'left': left,
#             'upper': upper,
#             'right': right,
#             'lower': lower,
#             'original_width': original_width,
#             'original_height': original_height,
#             'new_width': final_width,
#             'new_height': final_height,
#         }
#         return output_path, offset
# 
#     return output_path
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/plt/utils/_crop.py
# --------------------------------------------------------------------------------
