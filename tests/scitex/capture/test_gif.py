#!/usr/bin/env python3
"""Tests for scitex.capture.gif module.

Tests GIF creation functionality:
- GifCreator class methods
- create_gif_from_files()
- create_gif_from_session()
- create_gif_from_pattern()
- Session detection
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestGifCreatorInit:
    """Test GifCreator initialization."""

    def test_initialization(self):
        """Test GifCreator initializes correctly."""
        from scitex.capture.gif import GifCreator

        creator = GifCreator()
        assert creator is not None


class TestCreateGifFromFiles:
    """Test create_gif_from_files functionality."""

    def test_create_gif_with_valid_images(self):
        """Test GIF creation with valid image files."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            image_paths = []
            for i in range(3):
                img_path = os.path.join(tmpdir, f"frame_{i}.jpg")
                img = Image.new("RGB", (100, 100), color=(255 - i * 50, i * 50, 0))
                img.save(img_path, "JPEG")
                image_paths.append(img_path)

            output_path = os.path.join(tmpdir, "output.gif")
            creator = GifCreator()
            result = creator.create_gif_from_files(
                image_paths=image_paths, output_path=output_path
            )

            assert result is not None
            assert os.path.exists(result)
            assert result.endswith(".gif")

    def test_create_gif_empty_paths(self):
        """Test GIF creation with empty paths list."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.gif")
            creator = GifCreator()
            result = creator.create_gif_from_files(
                image_paths=[], output_path=output_path
            )

            assert result is None

    def test_create_gif_nonexistent_paths(self):
        """Test GIF creation with nonexistent image paths."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.gif")
            creator = GifCreator()
            result = creator.create_gif_from_files(
                image_paths=["/nonexistent/image1.jpg", "/nonexistent/image2.jpg"],
                output_path=output_path,
            )

            assert result is None

    def test_create_gif_with_duration(self):
        """Test GIF creation with custom duration."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            image_paths = []
            for i in range(2):
                img_path = os.path.join(tmpdir, f"frame_{i}.jpg")
                img = Image.new("RGB", (50, 50), color="blue")
                img.save(img_path, "JPEG")
                image_paths.append(img_path)

            output_path = os.path.join(tmpdir, "output.gif")
            creator = GifCreator()
            result = creator.create_gif_from_files(
                image_paths=image_paths, output_path=output_path, duration=1.0
            )

            assert result is not None

    def test_create_gif_with_different_sizes(self):
        """Test GIF creation with images of different sizes."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create images with different sizes
            img1_path = os.path.join(tmpdir, "frame_1.jpg")
            img2_path = os.path.join(tmpdir, "frame_2.jpg")

            Image.new("RGB", (100, 100), "red").save(img1_path)
            Image.new("RGB", (200, 200), "blue").save(img2_path)

            output_path = os.path.join(tmpdir, "output.gif")
            creator = GifCreator()
            result = creator.create_gif_from_files(
                image_paths=[img1_path, img2_path], output_path=output_path
            )

            assert result is not None

    def test_create_gif_creates_output_directory(self):
        """Test GIF creation creates parent directory."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "frame.jpg")
            Image.new("RGB", (50, 50), "green").save(img_path)

            # Output to nested nonexistent directory
            output_path = os.path.join(tmpdir, "nested", "deep", "output.gif")
            creator = GifCreator()
            result = creator.create_gif_from_files(
                image_paths=[img_path], output_path=output_path
            )

            assert result is not None
            assert os.path.exists(os.path.dirname(output_path))


class TestCreateGifFromSession:
    """Test create_gif_from_session functionality."""

    def test_create_gif_from_session_no_files(self):
        """Test GIF creation from session with no matching files."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            creator = GifCreator()
            result = creator.create_gif_from_session(
                session_id="20250104_120000", screenshot_dir=tmpdir
            )

            assert result is None

    def test_create_gif_from_session_with_files(self):
        """Test GIF creation from session with matching files."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "20250104_120000"

            # Create session files
            for i in range(3):
                img_path = os.path.join(tmpdir, f"{session_id}_{i:04d}_120000000.jpg")
                Image.new("RGB", (100, 100), "red").save(img_path)

            creator = GifCreator()
            result = creator.create_gif_from_session(
                session_id=session_id, screenshot_dir=tmpdir
            )

            assert result is not None
            assert session_id in result

    def test_create_gif_from_session_png_fallback(self):
        """Test GIF creation falls back to PNG files."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "20250104_130000"

            # Create PNG files (no JPG)
            for i in range(2):
                img_path = os.path.join(tmpdir, f"{session_id}_{i:04d}_130000000.png")
                Image.new("RGB", (100, 100), "blue").save(img_path)

            creator = GifCreator()
            result = creator.create_gif_from_session(
                session_id=session_id, screenshot_dir=tmpdir
            )

            assert result is not None

    def test_create_gif_from_session_max_frames(self):
        """Test GIF creation with max_frames limit."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "20250104_140000"

            # Create many session files
            for i in range(10):
                img_path = os.path.join(tmpdir, f"{session_id}_{i:04d}_140000000.jpg")
                Image.new("RGB", (50, 50), "green").save(img_path)

            creator = GifCreator()
            result = creator.create_gif_from_session(
                session_id=session_id, screenshot_dir=tmpdir, max_frames=3
            )

            assert result is not None


class TestCreateGifFromPattern:
    """Test create_gif_from_pattern functionality."""

    def test_create_gif_from_pattern_no_matches(self):
        """Test GIF creation with no matching files."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "nonexistent_*.jpg")
            creator = GifCreator()
            result = creator.create_gif_from_pattern(pattern=pattern)

            assert result is None

    def test_create_gif_from_pattern_with_matches(self):
        """Test GIF creation with matching files."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create matching files
            for i in range(3):
                img_path = os.path.join(tmpdir, f"screenshot_{i}.jpg")
                Image.new("RGB", (100, 100), "purple").save(img_path)

            pattern = os.path.join(tmpdir, "screenshot_*.jpg")
            output_path = os.path.join(tmpdir, "result.gif")
            creator = GifCreator()
            result = creator.create_gif_from_pattern(
                pattern=pattern, output_path=output_path
            )

            assert result is not None
            assert os.path.exists(result)

    def test_create_gif_from_pattern_auto_output_path(self):
        """Test GIF creation generates output path automatically."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create matching files
            img_path = os.path.join(tmpdir, "test_image.jpg")
            Image.new("RGB", (50, 50), "yellow").save(img_path)

            pattern = os.path.join(tmpdir, "test_*.jpg")
            creator = GifCreator()
            result = creator.create_gif_from_pattern(pattern=pattern)

            assert result is not None
            assert "gif_summary_" in result

    def test_create_gif_from_pattern_max_frames(self):
        """Test GIF creation from pattern with max_frames."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create many files
            for i in range(10):
                img_path = os.path.join(tmpdir, f"img_{i:02d}.jpg")
                Image.new("RGB", (50, 50), "orange").save(img_path)

            pattern = os.path.join(tmpdir, "img_*.jpg")
            output_path = os.path.join(tmpdir, "limited.gif")
            creator = GifCreator()
            result = creator.create_gif_from_pattern(
                pattern=pattern, output_path=output_path, max_frames=3
            )

            assert result is not None


class TestGetRecentSessions:
    """Test get_recent_sessions functionality."""

    def test_get_recent_sessions_empty_dir(self):
        """Test getting sessions from empty directory."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            creator = GifCreator()
            result = creator.get_recent_sessions(screenshot_dir=tmpdir)

            assert result == []

    def test_get_recent_sessions_nonexistent_dir(self):
        """Test getting sessions from nonexistent directory."""
        from scitex.capture.gif import GifCreator

        creator = GifCreator()
        result = creator.get_recent_sessions(screenshot_dir="/nonexistent/path")

        assert result == []

    def test_get_recent_sessions_with_files(self):
        """Test getting sessions with session files present."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create session files with proper naming
            session1 = "20250101_100000"
            session2 = "20250102_110000"

            Path(os.path.join(tmpdir, f"{session1}_0001_100000000.jpg")).touch()
            Path(os.path.join(tmpdir, f"{session1}_0002_100001000.jpg")).touch()
            Path(os.path.join(tmpdir, f"{session2}_0001_110000000.jpg")).touch()

            creator = GifCreator()
            result = creator.get_recent_sessions(screenshot_dir=tmpdir)

            assert len(result) == 2
            assert session2 in result  # Newer session
            assert session1 in result

    def test_get_recent_sessions_sorted_newest_first(self):
        """Test sessions are sorted newest first."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            sessions = ["20250101_100000", "20250103_100000", "20250102_100000"]

            for sess in sessions:
                Path(os.path.join(tmpdir, f"{sess}_0001_100000000.jpg")).touch()

            creator = GifCreator()
            result = creator.get_recent_sessions(screenshot_dir=tmpdir)

            assert result[0] == "20250103_100000"  # Newest
            assert result[-1] == "20250101_100000"  # Oldest


class TestCreateGifFromRecentSession:
    """Test create_gif_from_recent_session functionality."""

    def test_create_gif_from_recent_session_no_sessions(self):
        """Test GIF creation when no sessions exist."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            creator = GifCreator()
            result = creator.create_gif_from_recent_session(screenshot_dir=tmpdir)

            assert result is None

    def test_create_gif_from_recent_session_with_session(self):
        """Test GIF creation from most recent session."""
        from scitex.capture.gif import GifCreator

        try:
            from PIL import Image
        except ImportError:
            pytest.skip("PIL not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            session_id = "20250104_150000"

            # Create session files
            for i in range(3):
                img_path = os.path.join(tmpdir, f"{session_id}_{i:04d}_150000000.jpg")
                Image.new("RGB", (100, 100), "cyan").save(img_path)

            creator = GifCreator()
            result = creator.create_gif_from_recent_session(screenshot_dir=tmpdir)

            assert result is not None


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_create_gif_from_session_function(self):
        """Test create_gif_from_session convenience function."""
        from scitex.capture.gif import create_gif_from_session

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_gif_from_session(
                session_id="nonexistent", screenshot_dir=tmpdir
            )
            assert result is None

    def test_create_gif_from_files_function(self):
        """Test create_gif_from_files convenience function."""
        from scitex.capture.gif import create_gif_from_files

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.gif")
            result = create_gif_from_files(image_paths=[], output_path=output_path)
            assert result is None

    def test_create_gif_from_pattern_function(self):
        """Test create_gif_from_pattern convenience function."""
        from scitex.capture.gif import create_gif_from_pattern

        with tempfile.TemporaryDirectory() as tmpdir:
            pattern = os.path.join(tmpdir, "*.jpg")
            result = create_gif_from_pattern(pattern=pattern)
            assert result is None

    def test_create_gif_from_latest_session_function(self):
        """Test create_gif_from_latest_session convenience function."""
        from scitex.capture.gif import create_gif_from_latest_session

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_gif_from_latest_session(screenshot_dir=tmpdir)
            assert result is None


class TestModuleExports:
    """Test module exports."""

    def test_gifcreator_importable(self):
        """Test GifCreator class can be imported."""
        from scitex.capture.gif import GifCreator

        assert GifCreator is not None

    def test_all_functions_importable(self):
        """Test all convenience functions can be imported."""
        from scitex.capture.gif import (
            create_gif_from_files,
            create_gif_from_latest_session,
            create_gif_from_pattern,
            create_gif_from_session,
        )

        assert callable(create_gif_from_session)
        assert callable(create_gif_from_files)
        assert callable(create_gif_from_pattern)
        assert callable(create_gif_from_latest_session)

    def test_functions_from_package_init(self):
        """Test GIF functions accessible from package init."""
        from scitex.capture import (
            create_gif_from_files,
            create_gif_from_latest_session,
            create_gif_from_pattern,
            create_gif_from_session,
        )

        assert callable(create_gif_from_session)


class TestPILNotAvailable:
    """Test behavior when PIL is not available."""

    def test_create_gif_without_pil(self):
        """Test GIF creation handles missing PIL gracefully."""
        from scitex.capture.gif import GifCreator

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.gif")

            # Mock PIL import to fail
            with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
                creator = GifCreator()
                # Pass valid paths but PIL import should fail
                result = creator.create_gif_from_files(
                    image_paths=["/some/path.jpg"], output_path=output_path
                )

                # Should return None when PIL is not available
                assert result is None

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/capture/gif.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-10-18 09:55:56 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/gif.py
# # ----------------------------------------
# from __future__ import annotations
# import os
# 
# __FILE__ = "./src/scitex/capture/gif.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# """
# GIF creation functionality for CAM.
# Create animated GIFs from screenshot sequences for visual summaries.
# """
# 
# import glob
# import re
# from datetime import datetime
# from pathlib import Path
# from typing import List, Optional
# 
# 
# class GifCreator:
#     """
#     Creates animated GIFs from screenshot sequences.
#     Useful for creating visual summaries of monitoring sessions or workflows.
#     """
# 
#     def __init__(self):
#         """Initialize GIF creator."""
#         pass
# 
#     def create_gif_from_session(
#         self,
#         session_id: str,
#         output_path: Optional[str] = None,
#         screenshot_dir: str = "~/.scitex/capture",
#         duration: float = 0.5,
#         optimize: bool = True,
#         max_frames: Optional[int] = None,
#     ) -> Optional[str]:
#         """
#         Create a GIF from a monitoring session's screenshots.
# 
#         Args:
#             session_id: Session ID from monitoring (e.g., "20250823_104523")
#             output_path: Output GIF path (auto-generated if None)
#             screenshot_dir: Directory containing screenshots
#             duration: Duration per frame in seconds (default: 0.5)
#             optimize: Optimize GIF for smaller file size (default: True)
#             max_frames: Maximum number of frames to include (None = all)
# 
#         Returns:
#             Path to created GIF file, or None if failed
#         """
#         try:
#             screenshot_dir = Path(screenshot_dir).expanduser()
# 
#             # Find all screenshots for this session
#             pattern = f"{session_id}_*.jpg"
#             jpg_files = list(screenshot_dir.glob(pattern))
# 
#             # Also try PNG if no JPG files found
#             if not jpg_files:
#                 pattern = f"{session_id}_*.png"
#                 jpg_files = list(screenshot_dir.glob(pattern))
# 
#             if not jpg_files:
#                 print(f"No screenshots found for session {session_id}")
#                 return None
# 
#             # Sort by filename (which includes timestamp)
#             jpg_files.sort()
# 
#             # Limit frames if specified
#             if max_frames and len(jpg_files) > max_frames:
#                 # Take evenly spaced frames
#                 step = len(jpg_files) // max_frames
#                 jpg_files = jpg_files[::step][:max_frames]
# 
#             if output_path is None:
#                 output_path = screenshot_dir / f"{session_id}_summary.gif"
#             else:
#                 output_path = Path(output_path)
# 
#             return self.create_gif_from_files(
#                 image_paths=[str(f) for f in jpg_files],
#                 output_path=str(output_path),
#                 duration=duration,
#                 optimize=optimize,
#             )
# 
#         except Exception as e:
#             print(f"Error creating GIF from session: {e}")
#             return None
# 
#     def create_gif_from_files(
#         self,
#         image_paths: List[str],
#         output_path: str,
#         duration: float = 0.5,
#         optimize: bool = True,
#         loop: int = 0,
#     ) -> Optional[str]:
#         """
#         Create a GIF from a list of image files.
# 
#         Args:
#             image_paths: List of image file paths
#             output_path: Output GIF path
#             duration: Duration per frame in seconds (default: 0.5)
#             optimize: Optimize GIF for smaller file size (default: True)
#             loop: Number of loops (0 = infinite, default: 0)
# 
#         Returns:
#             Path to created GIF file, or None if failed
#         """
#         try:
#             from PIL import Image
# 
#             if not image_paths:
#                 print("No image paths provided")
#                 return None
# 
#             # Load all images
#             images = []
#             for path in image_paths:
#                 if not os.path.exists(path):
#                     print(f"Image not found: {path}")
#                     continue
# 
#                 try:
#                     img = Image.open(path)
#                     # Convert to RGB if necessary (for consistency)
#                     if img.mode != "RGB":
#                         img = img.convert("RGB")
#                     images.append(img)
#                 except Exception as e:
#                     print(f"Error loading image {path}: {e}")
#                     continue
# 
#             if not images:
#                 print("No valid images found")
#                 return None
# 
#             # Ensure all images have the same size (resize to first image size)
#             target_size = images[0].size
#             for i in range(1, len(images)):
#                 if images[i].size != target_size:
#                     images[i] = images[i].resize(target_size, Image.Resampling.LANCZOS)
# 
#             # Create output directory if it doesn't exist
#             output_path = Path(output_path)
#             output_path.parent.mkdir(parents=True, exist_ok=True)
# 
#             # Save as GIF
#             duration_ms = int(duration * 1000)  # Convert to milliseconds
# 
#             images[0].save(
#                 str(output_path),
#                 format="GIF",
#                 save_all=True,
#                 append_images=images[1:],
#                 duration=duration_ms,
#                 loop=loop,
#                 optimize=optimize,
#             )
# 
#             if output_path.exists():
#                 file_size = output_path.stat().st_size / 1024  # KB
#                 print(
#                     f"📹 GIF created: {output_path} ({len(images)} frames, {file_size:.1f}KB)"
#                 )
#                 return str(output_path)
#             else:
#                 return None
# 
#         except ImportError:
#             print(
#                 "PIL (Pillow) is required for GIF creation. Install with: pip install Pillow"
#             )
#             return None
#         except Exception as e:
#             print(f"Error creating GIF: {e}")
#             return None
# 
#     def create_gif_from_pattern(
#         self,
#         pattern: str,
#         output_path: Optional[str] = None,
#         duration: float = 0.5,
#         optimize: bool = True,
#         max_frames: Optional[int] = None,
#     ) -> Optional[str]:
#         """
#         Create a GIF from files matching a glob pattern.
# 
#         Args:
#             pattern: Glob pattern for image files (e.g., "/path/screenshots/*.jpg")
#             output_path: Output GIF path (auto-generated if None)
#             duration: Duration per frame in seconds (default: 0.5)
#             optimize: Optimize GIF for smaller file size (default: True)
#             max_frames: Maximum number of frames to include (None = all)
# 
#         Returns:
#             Path to created GIF file, or None if failed
#         """
#         try:
#             # Find matching files
#             files = glob.glob(pattern)
#             files.sort()  # Sort alphabetically
# 
#             if not files:
#                 print(f"No files found matching pattern: {pattern}")
#                 return None
# 
#             # Limit frames if specified
#             if max_frames and len(files) > max_frames:
#                 step = len(files) // max_frames
#                 files = files[::step][:max_frames]
# 
#             if output_path is None:
#                 # Generate output path based on pattern
#                 pattern_dir = Path(pattern).parent
#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 output_path = pattern_dir / f"gif_summary_{timestamp}.gif"
# 
#             return self.create_gif_from_files(
#                 image_paths=files,
#                 output_path=str(output_path),
#                 duration=duration,
#                 optimize=optimize,
#             )
# 
#         except Exception as e:
#             print(f"Error creating GIF from pattern: {e}")
#             return None
# 
#     def get_recent_sessions(
#         self, screenshot_dir: str = "~/.scitex/capture"
#     ) -> List[str]:
#         """
#         Get list of recent monitoring session IDs.
# 
#         Args:
#             screenshot_dir: Directory containing screenshots
# 
#         Returns:
#             List of session IDs sorted by recency (newest first)
#         """
#         try:
#             screenshot_dir = Path(screenshot_dir).expanduser()
# 
#             if not screenshot_dir.exists():
#                 return []
# 
#             # Find all monitoring session files (format: SESSIONID_NNNN_timestamp.ext)
#             session_pattern = re.compile(r"^(\d{8}_\d{6})_\d{4}_.*\.(jpg|png)$")
# 
#             sessions = set()
#             for file in screenshot_dir.iterdir():
#                 if file.is_file():
#                     match = session_pattern.match(file.name)
#                     if match:
#                         sessions.add(match.group(1))
# 
#             # Sort by session ID (which includes timestamp)
#             return sorted(sessions, reverse=True)
# 
#         except Exception as e:
#             print(f"Error getting recent sessions: {e}")
#             return []
# 
#     def create_gif_from_recent_session(
#         self,
#         screenshot_dir: str = "~/.scitex/capture",
#         duration: float = 0.5,
#         optimize: bool = True,
#         max_frames: Optional[int] = None,
#     ) -> Optional[str]:
#         """
#         Create a GIF from the most recent monitoring session.
# 
#         Args:
#             screenshot_dir: Directory containing screenshots
#             duration: Duration per frame in seconds (default: 0.5)
#             optimize: Optimize GIF for smaller file size (default: True)
#             max_frames: Maximum number of frames to include (None = all)
# 
#         Returns:
#             Path to created GIF file, or None if failed
#         """
#         sessions = self.get_recent_sessions(screenshot_dir)
# 
#         if not sessions:
#             print("No monitoring sessions found")
#             return None
# 
#         latest_session = sessions[0]
#         print(f"Creating GIF from latest session: {latest_session}")
# 
#         return self.create_gif_from_session(
#             session_id=latest_session,
#             screenshot_dir=screenshot_dir,
#             duration=duration,
#             optimize=optimize,
#             max_frames=max_frames,
#         )
# 
# 
# # Convenience functions for easy usage
# def create_gif_from_session(session_id: str, **kwargs) -> Optional[str]:
#     """Create GIF from monitoring session screenshots."""
#     creator = GifCreator()
#     return creator.create_gif_from_session(session_id, **kwargs)
# 
# 
# def create_gif_from_files(
#     image_paths: List[str], output_path: str, **kwargs
# ) -> Optional[str]:
#     """Create GIF from list of image files."""
#     creator = GifCreator()
#     return creator.create_gif_from_files(image_paths, output_path, **kwargs)
# 
# 
# def create_gif_from_pattern(pattern: str, **kwargs) -> Optional[str]:
#     """Create GIF from files matching glob pattern."""
#     creator = GifCreator()
#     return creator.create_gif_from_pattern(pattern, **kwargs)
# 
# 
# def create_gif_from_latest_session(**kwargs) -> Optional[str]:
#     """Create GIF from the most recent monitoring session."""
#     creator = GifCreator()
#     return creator.create_gif_from_recent_session(**kwargs)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/capture/gif.py
# --------------------------------------------------------------------------------
