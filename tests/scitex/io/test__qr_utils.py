# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_qr_utils.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_qr_utils.py
# """
# QR code utilities for image metadata visualization.
# """
# 
# import json
# import os
# from pathlib import Path
# 
# from scitex import logging
# 
# __FILE__ = __file__
# __DIR__ = os.path.dirname(__FILE__)
# 
# logger = logging.getLogger(__name__)
# 
# 
# def add_qr_to_figure(fig, metadata, position="bottom-right", size=0.08):
#     """
#     Add a minimal QR code to a matplotlib figure.
# 
#     Args:
#         fig: matplotlib figure object
#         metadata: dictionary of metadata
#         position: 'bottom-right', 'bottom-left', 'top-right', 'top-left'
#         size: relative size of QR code (0-1)
# 
#     Returns:
#         Modified figure
#     """
#     try:
#         import qrcode
#         from PIL import Image as PILImage
#     except ImportError:
#         logger.warning(
#             "qrcode library not available. Install with: pip install qrcode[pil]"
#         )
#         return fig
# 
#     # Ensure metadata has URL
#     if "url" not in metadata:
#         metadata = dict(metadata)
#         metadata["url"] = "https://scitex.ai"
# 
#     # Generate minimal QR code
#     metadata_json = json.dumps(metadata, ensure_ascii=False)
#     qr = qrcode.QRCode(
#         version=1,  # Start minimal
#         error_correction=qrcode.constants.ERROR_CORRECT_L,  # Lowest redundancy
#         box_size=2,  # Smallest box
#         border=1,  # Minimal border
#     )
#     qr.add_data(metadata_json)
#     qr.make(fit=True)
#     qr_img = qr.make_image(fill_color="black", back_color="white")
# 
#     # Position mapping
#     positions = {
#         "bottom-right": (0.92, 0.02),
#         "bottom-left": (0.02, 0.02),
#         "top-right": (0.92, 0.88),
#         "top-left": (0.02, 0.88),
#     }
# 
#     if position not in positions:
#         position = "bottom-right"
# 
#     x, y = positions[position]
# 
#     # Add QR code to figure
#     ax_qr = fig.add_axes(
#         [x, y, size, size * (fig.get_figheight() / fig.get_figwidth())]
#     )
#     ax_qr.imshow(qr_img, cmap="gray")
#     ax_qr.axis("off")
# 
#     return fig
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_qr_utils.py
# --------------------------------------------------------------------------------
