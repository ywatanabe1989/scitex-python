# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_jpeg.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata/embed_metadata_jpeg.py
# 
# """JPEG metadata embedding using EXIF ImageDescription field."""
# 
# from PIL import Image
# 
# 
# def embed_metadata_jpeg(image_path: str, metadata_json: str) -> None:
#     """
#     Embed metadata into a JPEG file using EXIF ImageDescription field.
# 
#     Args:
#         image_path: Path to the JPEG file.
#         metadata_json: JSON string of metadata to embed.
# 
#     Raises:
#         ImportError: If piexif is not installed.
#     """
#     try:
#         import piexif
#     except ImportError:
#         raise ImportError(
#             "piexif is required for JPEG metadata support. "
#             "Install with: pip install piexif"
#         )
# 
#     img = Image.open(image_path)
# 
#     # Convert to RGB if necessary (JPEG doesn't support RGBA)
#     if img.mode in ("RGBA", "LA", "P"):
#         rgb_img = Image.new("RGB", img.size, (255, 255, 255))
#         if img.mode == "P":
#             img = img.convert("RGBA")
#         if img.mode in ("RGBA", "LA"):
#             rgb_img.paste(img, mask=img.split()[-1])
#         else:
#             rgb_img.paste(img)
#         img = rgb_img
# 
#     # Create EXIF dict with metadata in ImageDescription field
#     exif_dict = {
#         "0th": {piexif.ImageIFD.ImageDescription: metadata_json.encode("utf-8")},
#         "Exif": {},
#         "GPS": {},
#         "1st": {},
#     }
# 
#     # Try to preserve existing EXIF data
#     try:
#         existing_exif = piexif.load(img.info.get("exif", b""))
#         # Merge with new metadata (prioritize new metadata)
#         for ifd in ["Exif", "GPS", "1st"]:
#             if ifd in existing_exif:
#                 exif_dict[ifd].update(existing_exif[ifd])
#     except:
#         pass  # If existing EXIF is corrupted, just use new metadata
# 
#     exif_bytes = piexif.dump(exif_dict)
# 
#     # Save with EXIF metadata (quality=100 for maximum quality)
#     img.save(
#         image_path,
#         "JPEG",
#         quality=100,
#         subsampling=0,
#         optimize=False,
#         exif=exif_bytes,
#     )
#     img.close()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/embed_metadata_jpeg.py
# --------------------------------------------------------------------------------
