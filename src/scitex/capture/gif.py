#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-18 09:55:56 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/gif.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/capture/gif.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
GIF creation functionality for CAM.
Create animated GIFs from screenshot sequences for visual summaries.
"""

import glob
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class GifCreator:
    """
    Creates animated GIFs from screenshot sequences.
    Useful for creating visual summaries of monitoring sessions or workflows.
    """

    def __init__(self):
        """Initialize GIF creator."""
        pass

    def create_gif_from_session(
        self,
        session_id: str,
        output_path: Optional[str] = None,
        screenshot_dir: str = "~/.scitex/capture",
        duration: float = 0.5,
        optimize: bool = True,
        max_frames: Optional[int] = None,
    ) -> Optional[str]:
        """
        Create a GIF from a monitoring session's screenshots.

        Args:
            session_id: Session ID from monitoring (e.g., "20250823_104523")
            output_path: Output GIF path (auto-generated if None)
            screenshot_dir: Directory containing screenshots
            duration: Duration per frame in seconds (default: 0.5)
            optimize: Optimize GIF for smaller file size (default: True)
            max_frames: Maximum number of frames to include (None = all)

        Returns:
            Path to created GIF file, or None if failed
        """
        try:
            screenshot_dir = Path(screenshot_dir).expanduser()

            # Find all screenshots for this session
            pattern = f"{session_id}_*.jpg"
            jpg_files = list(screenshot_dir.glob(pattern))

            # Also try PNG if no JPG files found
            if not jpg_files:
                pattern = f"{session_id}_*.png"
                jpg_files = list(screenshot_dir.glob(pattern))

            if not jpg_files:
                print(f"No screenshots found for session {session_id}")
                return None

            # Sort by filename (which includes timestamp)
            jpg_files.sort()

            # Limit frames if specified
            if max_frames and len(jpg_files) > max_frames:
                # Take evenly spaced frames
                step = len(jpg_files) // max_frames
                jpg_files = jpg_files[::step][:max_frames]

            if output_path is None:
                output_path = screenshot_dir / f"{session_id}_summary.gif"
            else:
                output_path = Path(output_path)

            return self.create_gif_from_files(
                image_paths=[str(f) for f in jpg_files],
                output_path=str(output_path),
                duration=duration,
                optimize=optimize,
            )

        except Exception as e:
            print(f"Error creating GIF from session: {e}")
            return None

    def create_gif_from_files(
        self,
        image_paths: List[str],
        output_path: str,
        duration: float = 0.5,
        optimize: bool = True,
        loop: int = 0,
    ) -> Optional[str]:
        """
        Create a GIF from a list of image files.

        Args:
            image_paths: List of image file paths
            output_path: Output GIF path
            duration: Duration per frame in seconds (default: 0.5)
            optimize: Optimize GIF for smaller file size (default: True)
            loop: Number of loops (0 = infinite, default: 0)

        Returns:
            Path to created GIF file, or None if failed
        """
        try:
            from PIL import Image

            if not image_paths:
                print("No image paths provided")
                return None

            # Load all images
            images = []
            for path in image_paths:
                if not os.path.exists(path):
                    print(f"Image not found: {path}")
                    continue

                try:
                    img = Image.open(path)
                    # Convert to RGB if necessary (for consistency)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"Error loading image {path}: {e}")
                    continue

            if not images:
                print("No valid images found")
                return None

            # Ensure all images have the same size (resize to first image size)
            target_size = images[0].size
            for i in range(1, len(images)):
                if images[i].size != target_size:
                    images[i] = images[i].resize(target_size, Image.Resampling.LANCZOS)

            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as GIF
            duration_ms = int(duration * 1000)  # Convert to milliseconds

            images[0].save(
                str(output_path),
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=duration_ms,
                loop=loop,
                optimize=optimize,
            )

            if output_path.exists():
                file_size = output_path.stat().st_size / 1024  # KB
                print(
                    f"📹 GIF created: {output_path} ({len(images)} frames, {file_size:.1f}KB)"
                )
                return str(output_path)
            else:
                return None

        except ImportError:
            print(
                "PIL (Pillow) is required for GIF creation. Install with: pip install Pillow"
            )
            return None
        except Exception as e:
            print(f"Error creating GIF: {e}")
            return None

    def create_gif_from_pattern(
        self,
        pattern: str,
        output_path: Optional[str] = None,
        duration: float = 0.5,
        optimize: bool = True,
        max_frames: Optional[int] = None,
    ) -> Optional[str]:
        """
        Create a GIF from files matching a glob pattern.

        Args:
            pattern: Glob pattern for image files (e.g., "/path/screenshots/*.jpg")
            output_path: Output GIF path (auto-generated if None)
            duration: Duration per frame in seconds (default: 0.5)
            optimize: Optimize GIF for smaller file size (default: True)
            max_frames: Maximum number of frames to include (None = all)

        Returns:
            Path to created GIF file, or None if failed
        """
        try:
            # Find matching files
            files = glob.glob(pattern)
            files.sort()  # Sort alphabetically

            if not files:
                print(f"No files found matching pattern: {pattern}")
                return None

            # Limit frames if specified
            if max_frames and len(files) > max_frames:
                step = len(files) // max_frames
                files = files[::step][:max_frames]

            if output_path is None:
                # Generate output path based on pattern
                pattern_dir = Path(pattern).parent
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = pattern_dir / f"gif_summary_{timestamp}.gif"

            return self.create_gif_from_files(
                image_paths=files,
                output_path=str(output_path),
                duration=duration,
                optimize=optimize,
            )

        except Exception as e:
            print(f"Error creating GIF from pattern: {e}")
            return None

    def get_recent_sessions(
        self, screenshot_dir: str = "~/.scitex/capture"
    ) -> List[str]:
        """
        Get list of recent monitoring session IDs.

        Args:
            screenshot_dir: Directory containing screenshots

        Returns:
            List of session IDs sorted by recency (newest first)
        """
        try:
            screenshot_dir = Path(screenshot_dir).expanduser()

            if not screenshot_dir.exists():
                return []

            # Find all monitoring session files (format: SESSIONID_NNNN_timestamp.ext)
            session_pattern = re.compile(r"^(\d{8}_\d{6})_\d{4}_.*\.(jpg|png)$")

            sessions = set()
            for file in screenshot_dir.iterdir():
                if file.is_file():
                    match = session_pattern.match(file.name)
                    if match:
                        sessions.add(match.group(1))

            # Sort by session ID (which includes timestamp)
            return sorted(sessions, reverse=True)

        except Exception as e:
            print(f"Error getting recent sessions: {e}")
            return []

    def create_gif_from_recent_session(
        self,
        screenshot_dir: str = "~/.scitex/capture",
        duration: float = 0.5,
        optimize: bool = True,
        max_frames: Optional[int] = None,
    ) -> Optional[str]:
        """
        Create a GIF from the most recent monitoring session.

        Args:
            screenshot_dir: Directory containing screenshots
            duration: Duration per frame in seconds (default: 0.5)
            optimize: Optimize GIF for smaller file size (default: True)
            max_frames: Maximum number of frames to include (None = all)

        Returns:
            Path to created GIF file, or None if failed
        """
        sessions = self.get_recent_sessions(screenshot_dir)

        if not sessions:
            print("No monitoring sessions found")
            return None

        latest_session = sessions[0]
        print(f"Creating GIF from latest session: {latest_session}")

        return self.create_gif_from_session(
            session_id=latest_session,
            screenshot_dir=screenshot_dir,
            duration=duration,
            optimize=optimize,
            max_frames=max_frames,
        )


# Convenience functions for easy usage
def create_gif_from_session(session_id: str, **kwargs) -> Optional[str]:
    """Create GIF from monitoring session screenshots."""
    creator = GifCreator()
    return creator.create_gif_from_session(session_id, **kwargs)


def create_gif_from_files(
    image_paths: List[str], output_path: str, **kwargs
) -> Optional[str]:
    """Create GIF from list of image files."""
    creator = GifCreator()
    return creator.create_gif_from_files(image_paths, output_path, **kwargs)


def create_gif_from_pattern(pattern: str, **kwargs) -> Optional[str]:
    """Create GIF from files matching glob pattern."""
    creator = GifCreator()
    return creator.create_gif_from_pattern(pattern, **kwargs)


def create_gif_from_latest_session(**kwargs) -> Optional[str]:
    """Create GIF from the most recent monitoring session."""
    creator = GifCreator()
    return creator.create_gif_from_recent_session(**kwargs)


# EOF
