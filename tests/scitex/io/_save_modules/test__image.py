#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-12 13:57:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/scitex_repo/tests/scitex/io/_save_modules/test__image.py
# ----------------------------------------
import os

__FILE__ = "./tests/scitex/io/_save_modules/test__image.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Test cases for image saving wrapper functionality
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

    from scitex.io._save_modules import save_image


class TestSaveImage:
    """Test suite for save_image wrapper function"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file_png = os.path.join(self.temp_dir, "test.png")
        self.test_file_jpg = os.path.join(self.temp_dir, "test.jpg")

    def teardown_method(self):
        """Clean up after tests"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        # Close any matplotlib figures
        if MATPLOTLIB_AVAILABLE:
            plt.close('all')

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL/Pillow not installed")
    def test_save_pil_image(self):
        """Test saving PIL Image object"""
        # Create a simple RGB image
        img = Image.new('RGB', (100, 100), color='red')
        save_image(img, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        
        # Verify saved image
        loaded = Image.open(self.test_file_png)
        assert loaded.size == (100, 100)
        assert loaded.mode == 'RGB'

    def test_save_numpy_array_rgb(self):
        """Test saving numpy array as image (RGB)"""
        # Create RGB image array (H, W, C)
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red channel
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        
        if PIL_AVAILABLE:
            loaded = Image.open(self.test_file_png)
            assert loaded.size == (100, 100)

    def test_save_numpy_array_grayscale(self):
        """Test saving grayscale numpy array"""
        # Create grayscale image
        arr = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    def test_save_numpy_array_float(self):
        """Test saving float numpy array (0-1 range)"""
        # Create float array
        arr = np.random.rand(100, 100, 3)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_save_matplotlib_figure(self):
        """Test saving matplotlib figure"""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title("Test Plot")
        
        save_image(fig, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        plt.close(fig)

    def test_save_different_formats(self):
        """Test saving in different image formats"""
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # PNG
        save_image(arr, self.test_file_png)
        assert os.path.exists(self.test_file_png)
        
        # JPEG
        save_image(arr, self.test_file_jpg)
        assert os.path.exists(self.test_file_jpg)
        
        # BMP
        bmp_file = os.path.join(self.temp_dir, "test.bmp")
        save_image(arr, bmp_file)
        assert os.path.exists(bmp_file)

    def test_save_with_quality(self):
        """Test saving JPEG with quality setting"""
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # High quality
        high_quality = os.path.join(self.temp_dir, "high_quality.jpg")
        save_image(arr, high_quality, quality=95)
        
        # Low quality
        low_quality = os.path.join(self.temp_dir, "low_quality.jpg")
        save_image(arr, low_quality, quality=10)
        
        # High quality file should be larger
        assert os.path.getsize(high_quality) > os.path.getsize(low_quality)

    def test_save_rgba_image(self):
        """Test saving image with alpha channel"""
        # Create RGBA image
        arr = np.zeros((100, 100, 4), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red
        arr[:, :, 3] = 128  # Semi-transparent
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)
        
        if PIL_AVAILABLE:
            loaded = Image.open(self.test_file_png)
            assert loaded.mode in ['RGBA', 'LA']

    def test_save_large_image(self):
        """Test saving large image"""
        # Create large image
        arr = np.random.randint(0, 256, (2000, 2000, 3), dtype=np.uint8)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    def test_save_binary_image(self):
        """Test saving binary (black and white) image"""
        # Create binary image
        arr = np.random.choice([0, 255], size=(100, 100), p=[0.5, 0.5]).astype(np.uint8)
        
        save_image(arr, self.test_file_png)
        
        assert os.path.exists(self.test_file_png)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="matplotlib not installed")
    def test_save_matplotlib_with_dpi(self):
        """Test saving matplotlib figure with custom DPI"""
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([1, 2, 3], [1, 2, 3])
        
        # Save with high DPI
        save_image(fig, self.test_file_png, dpi=300)
        
        assert os.path.exists(self.test_file_png)
        plt.close(fig)

    def test_save_from_file_path(self):
        """Test copying existing image file"""
        # First create an image
        arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        temp_source = os.path.join(self.temp_dir, "source.png")
        
        if PIL_AVAILABLE:
            Image.fromarray(arr).save(temp_source)
            
            # Now use save_image to copy it
            save_image(temp_source, self.test_file_png)
            
            assert os.path.exists(self.test_file_png)

    def test_error_invalid_input(self):
        """Test error handling for invalid input"""
        with pytest.raises(ValueError):
            save_image("not an image", self.test_file_png)
        
        with pytest.raises(ValueError):
            save_image(123, self.test_file_png)

    def test_save_with_metadata(self):
        """Test saving image with metadata (if supported)"""
        if PIL_AVAILABLE:
            img = Image.new('RGB', (100, 100), color='blue')
            
            # PIL images can have info dict
            img.info['description'] = 'Test image'
            img.info['software'] = 'scitex'
            
            save_image(img, self.test_file_png)
            
            loaded = Image.open(self.test_file_png)
            # Note: Not all formats preserve metadata

    def test_save_palette_image(self):
        """Test saving palette/indexed color image"""
        if PIL_AVAILABLE:
            # Create palette image
            img = Image.new('P', (100, 100))
            palette = []
            for i in range(256):
                palette.extend([i, 0, 0])  # Red gradient
            img.putpalette(palette)
            
            save_image(img, self.test_file_png)
            
            assert os.path.exists(self.test_file_png)
            loaded = Image.open(self.test_file_png)
            assert loaded.mode == 'P'


# EOF

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_image.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-12-01 12:23:32 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_image.py
# 
# import os
# 
# __FILE__ = __file__
# 
# import io as _io
# import logging
# 
# import plotly
# from PIL import Image
# 
# logger = logging.getLogger(__name__)
# 
# 
# def save_image(
#     obj,
#     spath,
#     metadata=None,
#     add_qr=False,
#     qr_position="bottom-right",
#     verbose=False,
#     save_stats=True,
#     **kwargs,
# ):
#     # Auto-save stats BEFORE saving (obj may be deleted during save)
#     if save_stats:
#         _save_stats_from_figure(obj, spath, verbose=verbose)
# 
#     # Add URL to metadata if not present
#     if metadata is not None:
#         if verbose:
#             logger.info(f"📝 Saving figure with metadata to: {spath}")
# 
#         if "url" not in metadata:
#             metadata = dict(metadata)
#             metadata["url"] = "https://scitex.ai"
#             if verbose:
#                 logger.info("  • Auto-added URL: https://scitex.ai")
# 
#         # Add QR code to figure if requested
#         if add_qr:
#             if verbose:
#                 logger.info(f"  • Adding QR code at position: {qr_position}")
#             try:
#                 from .._qr_utils import add_qr_to_figure
# 
#                 # Only add QR for matplotlib figures
#                 if hasattr(obj, "savefig") or (
#                     hasattr(obj, "figure") and hasattr(obj.figure, "savefig")
#                 ):
#                     fig = obj if hasattr(obj, "savefig") else obj.figure
#                     obj = add_qr_to_figure(fig, metadata, position=qr_position)
#             except Exception as e:
#                 import warnings
# 
#                 warnings.warn(f"Failed to add QR code: {e}")
# 
#     # png
#     if spath.endswith(".png"):
#         # plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(file=spath, format="png")
#         # PIL image
#         elif isinstance(obj, Image.Image):
#             obj.save(spath)
#         # matplotlib
#         else:
#             try:
#                 obj.savefig(spath, **kwargs)
#             except:
#                 obj.figure.savefig(spath, **kwargs)
#         del obj
# 
#     # tiff
#     elif spath.endswith(".tiff") or spath.endswith(".tif"):
#         # PIL image
#         if isinstance(obj, Image.Image):
#             obj.save(spath)
#         # matplotlib
#         else:
#             # Use kwargs dpi if provided, otherwise default to 300
#             save_kwargs = {"format": "tiff", "dpi": kwargs.get("dpi", 300)}
#             save_kwargs.update(kwargs)
#             try:
#                 obj.savefig(spath, **save_kwargs)
#             except:
#                 obj.figure.savefig(spath, **save_kwargs)
# 
#         del obj
# 
#     # jpeg
#     elif spath.endswith(".jpeg") or spath.endswith(".jpg"):
#         buf = _io.BytesIO()
# 
#         # plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(buf, format="png")
#             buf.seek(0)
#             img = Image.open(buf)
#             img.convert("RGB").save(
#                 spath, "JPEG", quality=100, subsampling=0, optimize=False
#             )
#             buf.close()
# 
#         # PIL image
#         elif isinstance(obj, Image.Image):
#             # Save with maximum quality for JPEG (quality=100 for daily use)
#             obj.save(spath, quality=100, subsampling=0, optimize=False)
# 
#         # matplotlib
#         else:
#             save_kwargs = {"format": "png"}
#             save_kwargs.update(kwargs)
#             try:
#                 obj.savefig(buf, **save_kwargs)
#             except:
#                 obj.figure.savefig(buf, **save_kwargs)
# 
#             buf.seek(0)
#             img = Image.open(buf)
#             # Save JPEG with very high quality settings for daily use (quality=98 is near-lossless)
#             img.convert("RGB").save(
#                 spath, "JPEG", quality=100, subsampling=0, optimize=False
#             )
#             buf.close()
#         del obj
# 
#     # GIF
#     elif spath.endswith(".gif"):
#         # PIL image
#         if isinstance(obj, Image.Image):
#             obj.save(spath, save_all=True)
#         # plotly - convert via PNG first
#         elif isinstance(obj, plotly.graph_objs.Figure):
#             buf = _io.BytesIO()
#             obj.write_image(buf, format="png")
#             buf.seek(0)
#             img = Image.open(buf)
#             img.save(spath, "GIF")
#             buf.close()
#         # matplotlib
#         else:
#             buf = _io.BytesIO()
#             save_kwargs = {"format": "png"}
#             save_kwargs.update(kwargs)
#             try:
#                 obj.savefig(buf, **save_kwargs)
#             except:
#                 obj.figure.savefig(buf, **save_kwargs)
#             buf.seek(0)
#             img = Image.open(buf)
#             img.save(spath, "GIF")
#             buf.close()
#         del obj
# 
#     # SVG
#     elif spath.endswith(".svg"):
#         # Plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(file=spath, format="svg")
#         # Matplotlib
#         else:
#             save_kwargs = {"format": "svg"}
#             save_kwargs.update(kwargs)
#             try:
#                 obj.savefig(spath, **save_kwargs)
#             except AttributeError:
#                 obj.figure.savefig(spath, **save_kwargs)
#         del obj
# 
#     # PDF
#     elif spath.endswith(".pdf"):
#         # Plotly
#         if isinstance(obj, plotly.graph_objs.Figure):
#             obj.write_image(file=spath, format="pdf")
#         # PIL Image - convert to PDF
#         elif isinstance(obj, Image.Image):
#             # Convert RGBA to RGB if needed
#             if obj.mode == "RGBA":
#                 rgb_img = Image.new("RGB", obj.size, (255, 255, 255))
#                 rgb_img.paste(obj, mask=obj.split()[3])
#                 rgb_img.save(spath, "PDF")
#             else:
#                 obj.save(spath, "PDF")
#         # Matplotlib
#         else:
#             save_kwargs = {"format": "pdf"}
#             save_kwargs.update(kwargs)
#             try:
#                 obj.savefig(spath, **save_kwargs)
#             except AttributeError:
#                 obj.figure.savefig(spath, **save_kwargs)
#         del obj
# 
#     # Embed metadata if provided
#     if metadata is not None:
#         from .._metadata import embed_metadata
# 
#         try:
#             embed_metadata(spath, metadata)
#             if verbose:
#                 logger.debug(f"  • Embedded metadata: {metadata}")
#         except Exception as e:
#             import warnings
# 
#             warnings.warn(f"Failed to embed metadata: {e}")
# 
# def _save_stats_from_figure(obj, spath, verbose=False):
#     """
#     Extract and save statistical annotations from a figure.
# 
#     Saves to {basename}_stats.csv if stats are found.
#     """
#     try:
#         from scitex.bridge import extract_stats_from_axes
#     except ImportError:
#         return  # Bridge not available
# 
#     # Get the matplotlib figure
#     fig = None
#     if hasattr(obj, "savefig"):
#         fig = obj
#     elif hasattr(obj, "figure") and hasattr(obj.figure, "savefig"):
#         fig = obj.figure
#     elif hasattr(obj, "_fig_mpl"):
#         fig = obj._fig_mpl
# 
#     if fig is None:
#         return
# 
#     # Extract stats from all axes
#     all_stats = []
#     try:
#         for ax in fig.axes:
#             stats = extract_stats_from_axes(ax)
#             all_stats.extend(stats)
#     except Exception:
#         return  # Silently fail if extraction fails
# 
#     if not all_stats:
#         return  # No stats to save
# 
#     # Build stats dataframe
#     try:
#         import pandas as pd
# 
#         stats_data = []
#         for stat in all_stats:
#             row = {
#                 "test_type": stat.test_type,
#                 "statistic_name": stat.statistic.get("name", ""),
#                 "statistic_value": stat.statistic.get("value", ""),
#                 "p_value": stat.p_value,
#                 "stars": stat.stars,
#             }
#             # Add effect size if available
#             if stat.effect_size:
#                 row["effect_size_name"] = stat.effect_size.get("name", "")
#                 row["effect_size_value"] = stat.effect_size.get("value", "")
#             # Add CI if available
#             if stat.ci_95:
#                 row["ci_95_lower"] = stat.ci_95[0]
#                 row["ci_95_upper"] = stat.ci_95[1]
#             # Add sample/group info if available (for consistent naming with plot CSV)
#             if stat.samples:
#                 for group_name, group_info in stat.samples.items():
#                     if isinstance(group_info, dict):
#                         row[f"n_{group_name}"] = group_info.get("n")
#                         row[f"mean_{group_name}"] = group_info.get("mean")
#                         row[f"std_{group_name}"] = group_info.get("std")
#             stats_data.append(row)
# 
#         stats_df = pd.DataFrame(stats_data)
# 
#         # Save to {basename}_stats.csv
#         import os
#         base, ext = os.path.splitext(spath)
#         stats_path = f"{base}_stats.csv"
#         stats_df.to_csv(stats_path, index=False)
# 
#         if verbose:
#             logger.info(f"  • Auto-saved stats to: {stats_path}")
# 
#     except Exception as e:
#         import warnings
#         warnings.warn(f"Failed to auto-save stats: {e}")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/io/_save_modules/_image.py
# --------------------------------------------------------------------------------
