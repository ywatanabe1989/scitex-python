#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: tests/scitex/dev/cv/test_compose.py
"""Tests for scitex.dev.cv composition functions."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from scitex.dev.cv._compose import (
    _check_ffmpeg,
    compose,
    concatenate_videos,
    image_to_video,
)


def ffmpeg_available() -> bool:
    """Check if ffmpeg is available for integration tests."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


class TestCheckFfmpeg:
    """Tests for _check_ffmpeg helper."""

    def test_check_ffmpeg_when_available(self) -> None:
        """Test ffmpeg check when ffmpeg is available."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert _check_ffmpeg() is True

    def test_check_ffmpeg_when_not_available(self) -> None:
        """Test ffmpeg check when ffmpeg is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_ffmpeg() is False

    def test_check_ffmpeg_when_fails(self) -> None:
        """Test ffmpeg check when command fails."""
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")
        ):
            assert _check_ffmpeg() is False


class TestImageToVideo:
    """Tests for image_to_video function."""

    def test_image_to_video_no_ffmpeg_raises(self, tmp_path: Path) -> None:
        """Test that missing ffmpeg raises RuntimeError."""
        # Create dummy image
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        with patch("scitex.dev.cv._compose._check_ffmpeg", return_value=False):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                image_to_video(img_path, tmp_path / "out.mp4")

    @pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
    def test_image_to_video_basic(self, tmp_path: Path) -> None:
        """Test basic image to video conversion."""
        import cv2

        # Create test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[200:280, 280:360] = [0, 255, 0]  # Green square
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        output = tmp_path / "output.mp4"
        result = image_to_video(img_path, output, duration=1.0, fps=30)

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    @pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg not available")
    def test_image_to_video_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that parent directories are created."""
        import cv2

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img_path = tmp_path / "test.png"
        cv2.imwrite(str(img_path), img)

        output = tmp_path / "nested" / "dir" / "output.mp4"
        result = image_to_video(img_path, output, duration=0.5)

        assert output.exists()


class TestConcatenateVideos:
    """Tests for concatenate_videos function."""

    def test_concatenate_no_ffmpeg_raises(self, tmp_path: Path) -> None:
        """Test that missing ffmpeg raises RuntimeError."""
        with patch("scitex.dev.cv._compose._check_ffmpeg", return_value=False):
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                concatenate_videos([tmp_path / "a.mp4"], tmp_path / "out.mp4")

    def test_concatenate_empty_list_raises(self, tmp_path: Path) -> None:
        """Test that empty video list raises ValueError."""
        with patch("scitex.dev.cv._compose._check_ffmpeg", return_value=True):
            with pytest.raises(ValueError, match="No videos"):
                concatenate_videos([], tmp_path / "out.mp4")


class TestCompose:
    """Tests for compose function."""

    def test_compose_validates_ffmpeg(self, tmp_path: Path) -> None:
        """Test that compose checks for ffmpeg."""
        content = tmp_path / "content.mp4"
        content.touch()

        with patch("scitex.dev.cv._compose._check_ffmpeg", return_value=False):
            with patch("scitex.dev.cv._compose.image_to_video") as mock_i2v:
                mock_i2v.side_effect = RuntimeError("ffmpeg not found")
                opening = tmp_path / "opening.png"
                opening.touch()

                with pytest.raises(RuntimeError):
                    compose(
                        content=content,
                        output=tmp_path / "out.mp4",
                        opening=opening,
                    )


# EOF
