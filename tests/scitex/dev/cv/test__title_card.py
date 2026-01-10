# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/cv/_title_card.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: 2026-01-08
# # File: src/scitex/dev/cv/_title_card.py
# """Title card generation using matplotlib.
# 
# Creates opening/closing title cards as images or short videos
# with consistent SciTeX branding.
# """
# 
# from __future__ import annotations
# 
# from datetime import datetime
# from pathlib import Path
# from typing import Optional
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# # SciTeX brand colors (using matplotlib-compatible RGBA tuples)
# SCITEX_COLORS = {
#     "primary": "#1a1a2e",
#     "secondary": "#16213e",
#     "accent": "#4da6ff",
#     "text": "#ffffff",
#     "text_muted": (1.0, 1.0, 1.0, 0.7),  # White with 70% opacity
#     "text_dim": (1.0, 1.0, 1.0, 0.5),  # White with 50% opacity
# }
# 
# 
# def _create_gradient_background(
#     width: int = 1920,
#     height: int = 1080,
#     color1: str = "#1a1a2e",
#     color2: str = "#16213e",
#     angle: float = 135,
# ) -> np.ndarray:
#     """Create a gradient background.
# 
#     Parameters
#     ----------
#     width : int
#         Image width in pixels.
#     height : int
#         Image height in pixels.
#     color1, color2 : str
#         Hex colors for gradient.
#     angle : float
#         Gradient angle in degrees.
# 
#     Returns
#     -------
#     np.ndarray
#         RGB image array (height, width, 3).
#     """
#     import matplotlib.colors as mcolors
# 
#     # Convert hex to RGB
#     c1 = np.array(mcolors.to_rgb(color1))
#     c2 = np.array(mcolors.to_rgb(color2))
# 
#     # Create gradient
#     angle_rad = np.radians(angle)
#     x = np.linspace(0, 1, width)
#     y = np.linspace(0, 1, height)
#     xx, yy = np.meshgrid(x, y)
# 
#     # Gradient along angle direction
#     t = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
#     t = (t - t.min()) / (t.max() - t.min())
# 
#     # Interpolate colors
#     img = np.zeros((height, width, 3))
#     for i in range(3):
#         img[:, :, i] = c1[i] + t * (c2[i] - c1[i])
# 
#     return img
# 
# 
# def create_title_card(
#     title: str,
#     subtitle: str = "",
#     timestamp: str = "",
#     output_path: Optional[str] = None,
#     width: int = 1920,
#     height: int = 1080,
#     dpi: int = 100,
#     background: str = "gradient",
#     title_fontsize: int = 72,
#     subtitle_fontsize: int = 36,
#     timestamp_fontsize: int = 18,
# ) -> Path:
#     """Create a title card image.
# 
#     Parameters
#     ----------
#     title : str
#         Main title text.
#     subtitle : str, optional
#         Subtitle text.
#     timestamp : str, optional
#         Timestamp text. If empty, uses current datetime.
#     output_path : str, optional
#         Output file path. Auto-generated if None.
#     width, height : int
#         Image dimensions in pixels.
#     dpi : int
#         Output DPI.
#     background : str
#         Background style: "gradient", "solid", or hex color.
#     title_fontsize : int
#         Title font size in points.
#     subtitle_fontsize : int
#         Subtitle font size in points.
#     timestamp_fontsize : int
#         Timestamp font size in points.
# 
#     Returns
#     -------
#     Path
#         Path to generated image.
#     """
#     # Set up figure
#     fig_width = width / dpi
#     fig_height = height / dpi
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
# 
#     # Background
#     if background == "gradient":
#         bg = _create_gradient_background(width, height)
#         ax.imshow(bg, extent=[0, 1, 0, 1], aspect="auto")
#     elif background == "solid":
#         ax.set_facecolor(SCITEX_COLORS["primary"])
#     else:
#         ax.set_facecolor(background)
# 
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.axis("off")
# 
#     # Title
#     ax.text(
#         0.5,
#         0.55,
#         title,
#         ha="center",
#         va="center",
#         fontsize=title_fontsize,
#         fontweight="bold",
#         color=SCITEX_COLORS["text"],
#         transform=ax.transAxes,
#     )
# 
#     # Subtitle
#     if subtitle:
#         ax.text(
#             0.5,
#             0.42,
#             subtitle,
#             ha="center",
#             va="center",
#             fontsize=subtitle_fontsize,
#             fontweight="normal",
#             color=SCITEX_COLORS["text_muted"],
#             transform=ax.transAxes,
#         )
# 
#     # Timestamp
#     ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M")
#     ax.text(
#         0.5,
#         0.30,
#         ts,
#         ha="center",
#         va="center",
#         fontsize=timestamp_fontsize,
#         fontfamily="monospace",
#         color=SCITEX_COLORS["text_dim"],
#         transform=ax.transAxes,
#     )
# 
#     # Save
#     if output_path is None:
#         output_path = f"/tmp/title_card_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
# 
#     # Save with exact dimensions (no bbox_inches='tight' to preserve size)
#     fig.savefig(
#         output_path,
#         facecolor=SCITEX_COLORS["primary"],
#         edgecolor="none",
#         dpi=dpi,
#     )
#     plt.close(fig)
# 
#     return output_path
# 
# 
# def create_opening(
#     title: str,
#     subtitle: str = "Part of SciTeX",
#     timestamp: str = "",
#     output_path: Optional[str] = None,
#     product: str = "SciTeX",
#     version: str = "",
#     **kwargs,
# ) -> Path:
#     """Create an opening title card with SciTeX branding.
# 
#     Parameters
#     ----------
#     title : str
#         Main title (e.g., "Demo: Feature X").
#     subtitle : str
#         Subtitle (default: "Part of SciTeX").
#     timestamp : str
#         Timestamp. Uses current datetime if empty.
#     output_path : str, optional
#         Output file path.
#     product : str
#         Product name (e.g., "FigRecipe", "SciTeX").
#     version : str
#         Version string (e.g., "v2.1.2").
#     **kwargs
#         Additional arguments passed to create_title_card.
# 
#     Returns
#     -------
#     Path
#         Path to generated image.
#     """
#     if version:
#         subtitle = f"{product} {version}"
#     elif subtitle == "Part of SciTeX" and product != "SciTeX":
#         subtitle = f"Part of SciTeX - {product}"
# 
#     if output_path is None:
#         output_path = f"/tmp/opening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
# 
#     return create_title_card(
#         title=title,
#         subtitle=subtitle,
#         timestamp=timestamp,
#         output_path=output_path,
#         **kwargs,
#     )
# 
# 
# def create_closing(
#     output_path: Optional[str] = None,
#     product: str = "SciTeX",
#     tagline: str = "Automated Science",
#     url: str = "https://scitex.ai",
#     width: int = 1920,
#     height: int = 1080,
#     dpi: int = 100,
# ) -> Path:
#     """Create a closing branding card.
# 
#     Parameters
#     ----------
#     output_path : str, optional
#         Output file path.
#     product : str
#         Main product name.
#     tagline : str
#         Tagline text.
#     url : str
#         Website URL.
#     width, height : int
#         Image dimensions.
#     dpi : int
#         Output DPI.
# 
#     Returns
#     -------
#     Path
#         Path to generated image.
#     """
#     fig_width = width / dpi
#     fig_height = height / dpi
#     fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
# 
#     # Gradient background
#     bg = _create_gradient_background(width, height)
#     ax.imshow(bg, extent=[0, 1, 0, 1], aspect="auto")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.axis("off")
# 
#     # Product name
#     ax.text(
#         0.5,
#         0.55,
#         product,
#         ha="center",
#         va="center",
#         fontsize=72,
#         fontweight="bold",
#         color=SCITEX_COLORS["text"],
#         transform=ax.transAxes,
#     )
# 
#     # Tagline
#     ax.text(
#         0.5,
#         0.42,
#         tagline,
#         ha="center",
#         va="center",
#         fontsize=28,
#         color=SCITEX_COLORS["text_muted"],
#         transform=ax.transAxes,
#     )
# 
#     # URL
#     ax.text(
#         0.5,
#         0.30,
#         url,
#         ha="center",
#         va="center",
#         fontsize=24,
#         color=SCITEX_COLORS["accent"],
#         transform=ax.transAxes,
#     )
# 
#     # Save with exact dimensions
#     if output_path is None:
#         output_path = f"/tmp/closing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
#     output_path = Path(output_path)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
# 
#     fig.savefig(
#         output_path,
#         facecolor=SCITEX_COLORS["primary"],
#         edgecolor="none",
#         dpi=dpi,
#     )
#     plt.close(fig)
# 
#     return output_path
# 
# 
# __all__ = [
#     "create_title_card",
#     "create_opening",
#     "create_closing",
# ]
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/dev/cv/_title_card.py
# --------------------------------------------------------------------------------
