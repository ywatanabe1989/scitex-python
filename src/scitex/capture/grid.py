#!/usr/bin/env python3
# Timestamp: "2026-01-01 19:50:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/grid.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = "./src/scitex/capture/grid.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Grid overlay functionality for screenshot coordinate mapping.

Provides utilities to draw coordinate grids on screenshots,
helping AI agents understand screen positions for precise targeting.
"""

from pathlib import Path
from typing import Tuple


def draw_grid_overlay(
    filepath: str,
    grid_spacing: int = 100,
    output_path: str = None,
    grid_color: Tuple[int, int, int] = (255, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 0),
    line_width: int = 1,
    show_coordinates: bool = True,
) -> str:
    """
    Draw a coordinate grid overlay on a screenshot image.

    This helps AI agents and users understand screen coordinates for
    precise click targeting and UI element identification.

    Parameters
    ----------
    filepath : str
        Path to the input image
    grid_spacing : int
        Pixels between grid lines (default: 100)
    output_path : str, optional
        Output path (default: adds '_grid' suffix to input)
    grid_color : tuple
        RGB color for grid lines (default: red)
    text_color : tuple
        RGB color for coordinate labels (default: yellow)
    line_width : int
        Width of grid lines in pixels (default: 1)
    show_coordinates : bool
        Whether to show coordinate labels (default: True)

    Returns
    -------
    str
        Path to the output image with grid overlay

    Examples
    --------
    >>> from scitex.capture.grid import draw_grid_overlay
    >>> draw_grid_overlay("/path/to/screenshot.jpg")
    '/path/to/screenshot_grid.jpg'
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError(
            "PIL (Pillow) is required for grid overlay. "
            "Install with: pip install Pillow"
        )

    # Load image
    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to load a font for coordinate labels
    font = None
    if show_coordinates:
        font = _get_font(size=12)

    # Draw vertical lines
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=line_width)
        if show_coordinates and font:
            draw.text((x + 2, 10), str(x), fill=text_color, font=font)

    # Draw horizontal lines
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=grid_color, width=line_width)
        if show_coordinates and font:
            draw.text((10, y + 2), str(y), fill=text_color, font=font)

    # Generate output path
    if output_path is None:
        path_obj = Path(filepath)
        output_path = str(path_obj.parent / f"{path_obj.stem}_grid{path_obj.suffix}")

    # Save
    if filepath.lower().endswith((".jpg", ".jpeg")):
        img.save(output_path, "JPEG", quality=85)
    else:
        img.save(output_path)

    return output_path


def _get_font(size: int = 12):
    """Get a monospace font for coordinate labels."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    # Try common system fonts
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue

    # Fallback to default
    try:
        return ImageFont.load_default()
    except:
        return None


def add_monitor_info_overlay(
    filepath: str,
    monitor_info: dict,
    output_path: str = None,
) -> str:
    """
    Add monitor boundary and info overlay to a multi-monitor screenshot.

    Parameters
    ----------
    filepath : str
        Path to the input image
    monitor_info : dict
        Dictionary with monitor information from get_info()
    output_path : str, optional
        Output path (default: adds '_monitors' suffix)

    Returns
    -------
    str
        Path to the output image with monitor overlay
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("PIL (Pillow) is required")

    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    font = _get_font(size=14)

    # Draw monitor boundaries and labels
    monitors = monitor_info.get("Monitors", {}).get("Details", [])
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    for i, mon in enumerate(monitors):
        color = colors[i % len(colors)]
        x = mon.get("X", 0)
        y = mon.get("Y", 0)
        w = mon.get("Width", 0)
        h = mon.get("Height", 0)

        # Offset for combined image (Y might be negative)
        # Find minimum Y to offset
        min_y = min(m.get("Y", 0) for m in monitors) if monitors else 0
        offset_y = -min_y if min_y < 0 else 0

        # Draw rectangle border
        rect_y = y + offset_y
        draw.rectangle(
            [(x, rect_y), (x + w - 1, rect_y + h - 1)],
            outline=color,
            width=3,
        )

        # Draw label
        label = f"Monitor {i}: {w}x{h} @ ({x},{y})"
        if mon.get("Primary"):
            label += " [PRIMARY]"

        if font:
            draw.text((x + 10, rect_y + 10), label, fill=color, font=font)

    # Generate output path
    if output_path is None:
        path_obj = Path(filepath)
        output_path = str(
            path_obj.parent / f"{path_obj.stem}_monitors{path_obj.suffix}"
        )

    # Save
    if filepath.lower().endswith((".jpg", ".jpeg")):
        img.save(output_path, "JPEG", quality=85)
    else:
        img.save(output_path)

    return output_path


def draw_cursor_overlay(
    filepath: str,
    cursor_pos: Tuple[int, int] = None,
    output_path: str = None,
    marker_color: Tuple[int, int, int] = (0, 255, 0),
    marker_size: int = 20,
    show_coords: bool = True,
    capture_mode: str = "all",  # "all" for all monitors, or monitor index "0", "1", etc.
) -> str:
    """
    Draw cursor position marker on a screenshot.

    Helps verify cursor coordinates for UI automation debugging.

    Parameters
    ----------
    filepath : str
        Path to the input image
    cursor_pos : tuple, optional
        System coordinates (x, y). If None, gets current cursor position.
    output_path : str, optional
        Output path (default: adds '_cursor' suffix)
    marker_color : tuple
        RGB color for cursor marker (default: green)
    marker_size : int
        Size of the crosshair marker (default: 20)
    show_coords : bool
        Whether to show coordinate text (default: True)
    capture_mode : str
        "all" for all-monitor capture, or "0", "1" etc. for specific monitor

    Returns
    -------
    str
        Path to the output image with cursor overlay
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("PIL (Pillow) is required")

    # Get cursor position if not provided
    if cursor_pos is None:
        cursor_pos = _get_cursor_position()
        if cursor_pos is None:
            raise RuntimeError("Could not get cursor position")

    sys_x, sys_y = cursor_pos

    # Get display info to calculate proper offsets
    display_info = get_display_info()
    monitors = display_info.get("monitors", [])

    # Calculate image coordinate offset based on capture mode
    if capture_mode == "all" and monitors:
        # For all-monitor capture: offset by the minimum X and Y
        min_x = min(m.get("X", 0) for m in monitors)
        min_y = min(m.get("Y", 0) for m in monitors)
        img_x = sys_x - min_x
        img_y = sys_y - min_y
    elif monitors and capture_mode.isdigit():
        # For single monitor capture: offset by that monitor's position
        mon_idx = int(capture_mode)
        if mon_idx < len(monitors):
            mon = monitors[mon_idx]
            img_x = sys_x - mon.get("X", 0)
            img_y = sys_y - mon.get("Y", 0)
        else:
            img_x, img_y = sys_x, sys_y
    else:
        # Fallback: no offset
        img_x, img_y = sys_x, sys_y

    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Find which monitor the cursor is on
    cursor_monitor = "?"
    for i, mon in enumerate(monitors):
        mx, my = mon.get("X", 0), mon.get("Y", 0)
        mw, mh = mon.get("Width", 0), mon.get("Height", 0)
        if mx <= sys_x < mx + mw and my <= sys_y < my + mh:
            cursor_monitor = str(i)
            break

    # Check if cursor is within image bounds
    if 0 <= img_x < width and 0 <= img_y < height:
        # Draw crosshair
        half = marker_size // 2
        # Horizontal line
        draw.line(
            [(img_x - half, img_y), (img_x + half, img_y)],
            fill=marker_color,
            width=2,
        )
        # Vertical line
        draw.line(
            [(img_x, img_y - half), (img_x, img_y + half)],
            fill=marker_color,
            width=2,
        )
        # Center dot
        draw.ellipse(
            [(img_x - 3, img_y - 3), (img_x + 3, img_y + 3)],
            fill=marker_color,
        )

        # Draw coordinate text with monitor info
        if show_coords:
            font = _get_font(size=12)
            text = f"Mon:{cursor_monitor} Sys:({sys_x},{sys_y}) Img:({img_x},{img_y})"
            if font:
                draw.text((img_x + 10, img_y + 10), text, fill=marker_color, font=font)
    else:
        # Cursor outside image - show info at corner
        font = _get_font(size=12)
        text = f"Outside image: Mon:{cursor_monitor} Sys:({sys_x},{sys_y}) Img:({img_x},{img_y})"
        if font:
            draw.text((10, height - 30), text, fill=(255, 0, 0), font=font)

    # Generate output path
    if output_path is None:
        path_obj = Path(filepath)
        output_path = str(path_obj.parent / f"{path_obj.stem}_cursor{path_obj.suffix}")

    # Save
    if filepath.lower().endswith((".jpg", ".jpeg")):
        img.save(output_path, "JPEG", quality=85)
    else:
        img.save(output_path)

    return output_path


def _get_cursor_position() -> Tuple[int, int]:
    """Get current cursor position from Windows via PowerShell."""
    import subprocess

    ps_script = """
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class CursorPos {
    [DllImport("user32.dll")]
    public static extern bool GetCursorPos(out POINT lpPoint);
    [StructLayout(LayoutKind.Sequential)]
    public struct POINT { public int X; public int Y; }
}
"@
$pos = New-Object CursorPos+POINT
[CursorPos]::GetCursorPos([ref]$pos) | Out-Null
Write-Output "$($pos.X),$($pos.Y)"
"""
    try:
        result = subprocess.run(
            ["powershell.exe", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return None


def _get_dpi_scale() -> float:
    """Get Windows DPI scaling factor via PowerShell."""
    import subprocess

    ps_script = """
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class DpiInfo {
    [DllImport("user32.dll")]
    public static extern IntPtr GetDC(IntPtr hwnd);
    [DllImport("gdi32.dll")]
    public static extern int GetDeviceCaps(IntPtr hdc, int nIndex);
    [DllImport("user32.dll")]
    public static extern int ReleaseDC(IntPtr hwnd, IntPtr hdc);
    public const int LOGPIXELSX = 88;
}
"@
$hdc = [DpiInfo]::GetDC([IntPtr]::Zero)
$dpi = [DpiInfo]::GetDeviceCaps($hdc, 88)
[DpiInfo]::ReleaseDC([IntPtr]::Zero, $hdc) | Out-Null
$scale = $dpi / 96.0
Write-Output $scale
"""
    try:
        result = subprocess.run(
            ["powershell.exe", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return 1.0


def get_display_info() -> dict:
    """Get comprehensive display info including DPI, resolution, monitors."""
    import subprocess

    ps_script = """
Add-Type -AssemblyName System.Windows.Forms
$screens = [System.Windows.Forms.Screen]::AllScreens
$info = @()
foreach ($s in $screens) {
    $info += @{
        Name = $s.DeviceName
        Primary = $s.Primary
        X = $s.Bounds.X
        Y = $s.Bounds.Y
        Width = $s.Bounds.Width
        Height = $s.Bounds.Height
    }
}
$info | ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(
            ["powershell.exe", "-Command", ps_script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import json

            monitors = json.loads(result.stdout.strip())
            if not isinstance(monitors, list):
                monitors = [monitors]
            dpi_scale = _get_dpi_scale()
            return {
                "monitors": monitors,
                "dpi_scale": dpi_scale,
                "dpi_percent": int(dpi_scale * 100),
            }
    except Exception:
        pass
    return {"monitors": [], "dpi_scale": 1.0, "dpi_percent": 100}


__all__ = [
    "draw_grid_overlay",
    "add_monitor_info_overlay",
    "draw_cursor_overlay",
    "get_display_info",
]

# EOF
