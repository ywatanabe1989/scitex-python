#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/dev/cv/_compose.py
"""Video composition utilities using ffmpeg.

Provides tools for:
- Converting images to video clips
- Concatenating videos
- Composing opening + content + closing
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union


def _check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def image_to_video(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    duration: float = 3.0,
    fps: int = 30,
    fade_in: float = 0.5,
    fade_out: float = 0.5,
    resolution: Optional[str] = None,
) -> Path:
    """Convert a static image to a video clip.

    Parameters
    ----------
    image_path : str or Path
        Input image path.
    output_path : str or Path
        Output video path.
    duration : float
        Duration in seconds.
    fps : int
        Frames per second.
    fade_in : float
        Fade-in duration in seconds.
    fade_out : float
        Fade-out duration in seconds.
    resolution : str, optional
        Output resolution (e.g., "1920x1080").

    Returns
    -------
    Path
        Path to output video.
    """
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found. Install with: apt install ffmpeg")

    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build filter chain
    filters = []

    # Duration loop
    filters.append("loop=loop=-1:size=1")

    # Scale if resolution specified
    if resolution:
        filters.append(f"scale={resolution}")

    # Fade effects
    if fade_in > 0:
        filters.append(f"fade=t=in:st=0:d={fade_in}")
    if fade_out > 0:
        fade_out_start = duration - fade_out
        filters.append(f"fade=t=out:st={fade_out_start}:d={fade_out}")

    # Trim to duration
    filters.append(f"trim=duration={duration}")

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(image_path),
        "-vf",
        filter_str,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(fps),
        "-preset",
        "fast",
        "-crf",
        "23",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    return output_path


def concatenate_videos(
    video_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    transition: str = "none",
    transition_duration: float = 0.5,
) -> Path:
    """Concatenate multiple videos.

    Parameters
    ----------
    video_paths : list
        List of input video paths.
    output_path : str or Path
        Output video path.
    transition : str
        Transition type: "none", "fade", "dissolve".
    transition_duration : float
        Transition duration in seconds.

    Returns
    -------
    Path
        Path to output video.
    """
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg not found. Install with: apt install ffmpeg")

    if not video_paths:
        raise ValueError("No videos to concatenate")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create concat file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for vp in video_paths:
            f.write(f"file '{Path(vp).absolute()}'\n")
        concat_file = f.name

    try:
        if transition == "none":
            # Simple concatenation
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "fast",
                "-crf",
                "23",
                str(output_path),
            ]
        else:
            # Use filter_complex for transitions
            # Build input list and filter
            inputs = []
            filter_parts = []

            for i, vp in enumerate(video_paths):
                inputs.extend(["-i", str(vp)])

            # For fade/dissolve, use xfade filter
            if len(video_paths) == 2:
                filter_str = f"[0:v][1:v]xfade=transition={transition}:duration={transition_duration}:offset=0[outv]"
                cmd = [
                    "ffmpeg",
                    "-y",
                    *inputs,
                    "-filter_complex",
                    filter_str,
                    "-map",
                    "[outv]",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "fast",
                    str(output_path),
                ]
            else:
                # Fall back to simple concat for >2 videos
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_file,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    str(output_path),
                ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    finally:
        Path(concat_file).unlink(missing_ok=True)

    return output_path


def compose(
    content: Union[str, Path],
    output: Union[str, Path],
    opening: Optional[Union[str, Path]] = None,
    closing: Optional[Union[str, Path]] = None,
    opening_duration: float = 3.0,
    closing_duration: float = 3.0,
    transition: str = "fade",
    transition_duration: float = 0.5,
) -> Path:
    """Compose a full video with opening, content, and closing.

    Parameters
    ----------
    content : str or Path
        Main content video path.
    output : str or Path
        Output video path.
    opening : str or Path, optional
        Opening image or video. If image, converts to video.
    closing : str or Path, optional
        Closing image or video. If image, converts to video.
    opening_duration : float
        Duration for opening if it's an image.
    closing_duration : float
        Duration for closing if it's an image.
    transition : str
        Transition type: "none", "fade", "dissolve".
    transition_duration : float
        Transition duration in seconds.

    Returns
    -------
    Path
        Path to output video.
    """
    videos_to_concat = []
    temp_files = []

    try:
        # Process opening
        if opening:
            opening = Path(opening)
            if opening.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                # Convert image to video
                temp_opening = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(temp_opening)
                image_to_video(opening, temp_opening, duration=opening_duration)
                videos_to_concat.append(temp_opening)
            else:
                videos_to_concat.append(opening)

        # Add main content
        videos_to_concat.append(Path(content))

        # Process closing
        if closing:
            closing = Path(closing)
            if closing.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                # Convert image to video
                temp_closing = Path(tempfile.mktemp(suffix=".mp4"))
                temp_files.append(temp_closing)
                image_to_video(closing, temp_closing, duration=closing_duration)
                videos_to_concat.append(temp_closing)
            else:
                videos_to_concat.append(closing)

        # Concatenate all
        return concatenate_videos(
            videos_to_concat,
            output,
            transition=transition,
            transition_duration=transition_duration,
        )

    finally:
        # Cleanup temp files
        for tf in temp_files:
            tf.unlink(missing_ok=True)


__all__ = [
    "image_to_video",
    "concatenate_videos",
    "compose",
]

# EOF
