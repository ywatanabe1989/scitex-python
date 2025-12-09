#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-18 09:55:52 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/utils.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/capture/utils.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys

"""
Utility functions for easy screen capture.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from .capture import CaptureManager, ScreenshotWorker

# Global manager instance
_manager = CaptureManager()


def _manage_cache_size(cache_dir: Path, max_size_gb: float = 1.0):
    """
    Manage cache directory size by removing old files if size exceeds limit.

    Parameters
    ----------
    cache_dir : Path
        Directory to manage
    max_size_gb : float
        Maximum size in GB (default: 1.0)
    """
    if not cache_dir.exists():
        return

    max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes

    # Get all files with their sizes and modification times
    files = []
    total_size = 0

    for file_path in cache_dir.glob("*.jpg"):
        if file_path.is_file():
            size = file_path.stat().st_size
            mtime = file_path.stat().st_mtime
            files.append((file_path, size, mtime))
            total_size += size

    # Also check PNG files
    for file_path in cache_dir.glob("*.png"):
        if file_path.is_file():
            size = file_path.stat().st_size
            mtime = file_path.stat().st_mtime
            files.append((file_path, size, mtime))
            total_size += size

    # If under limit, nothing to do
    if total_size <= max_size_bytes:
        return

    # Sort by modification time (oldest first)
    files.sort(key=lambda x: x[2])

    # Remove oldest files until under limit
    for file_path, size, _ in files:
        if total_size <= max_size_bytes:
            break
        try:
            file_path.unlink()
            total_size -= size
        except:
            pass  # File might be in use


def capture(
    message: str = None,
    path: str = None,
    quality: int = 85,
    auto_categorize: bool = True,
    verbose: bool = True,
    monitor_id: int = 0,
    capture_all: bool = False,
    all: bool = False,
    app: str = None,
    url: str = None,
    url_wait: int = 3,
    url_width: int = 1920,
    url_height: int = 1080,
    max_cache_gb: float = 1.0,
) -> str:
    """
    Take a screenshot - monitor, window, browser, or everything.

    Parameters
    ----------
    message : str, optional
        Message to include in filename
    path : str, optional
        Output path (default: ~/.scitex/capture/)
    quality : int
        JPEG quality (1-100)
    all : bool
        Capture all monitors
    app : str, optional
        App name to capture (e.g., "chrome", "code")
    url : str, optional
        URL to capture via browser (e.g., "http://127.0.0.1:8000/")
    url_wait : int
        Seconds to wait for page load (default: 3)
    url_width : int
        Browser window width for URL capture (default: 1920)
    url_height : int
        Browser window height for URL capture (default: 1080)
    monitor_id : int
        Monitor to capture (0-based, default: 0)

    Returns
    -------
    str
        Path to saved screenshot

    Examples
    --------
    >>> from scitex import capture
    >>>
    >>> capture.snap()                           # Current monitor
    >>> capture.snap(all=True)                   # All monitors
    >>> capture.snap(app="chrome")               # Chrome window
    >>> capture.snap(url="http://localhost:8000") # Browser page
    """
    # Handle URL capture
    if url:
        # Auto-add http:// if no protocol specified
        if not url.startswith(("http://", "https://", "file://")):
            url = f"http://{url}"

        # Try Playwright first (headless, non-interfering)
        try:
            from playwright.sync_api import sync_playwright

            if path is None:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                url_slug = (
                    url.replace("://", "_").replace("/", "_").replace(":", "_")[:30]
                )
                path = f"~/.scitex/capture/{timestamp_str}-url-{url_slug}.jpg"

            path = os.path.expanduser(path)

            if verbose:
                print(f"📸 Capturing URL: {url}")

            # Check if DISPLAY is set (WSL with X11 forward causes visible browser)
            import os as _os

            original_display = _os.environ.get("DISPLAY")

            # Force headless by unsetting DISPLAY temporarily
            if original_display:
                _os.environ.pop("DISPLAY", None)

            try:
                with sync_playwright() as p:
                    # Use stealth args from scitex.browser
                    stealth_args = [
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                        "--window-size=1920,1080",
                    ]
                    browser = p.chromium.launch(headless=True, args=stealth_args)
                    context = browser.new_context(
                        viewport={"width": url_width, "height": url_height}
                    )
                    page = context.new_page()
                    # Use domcontentloaded for faster capture, with longer timeout
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    # Wait additional time for rendering
                    page.wait_for_timeout(url_wait * 1000)
                    page.screenshot(
                        path=path,
                        type="jpeg",
                        quality=quality,
                        full_page=False,
                    )
                    browser.close()
            finally:
                # Restore DISPLAY
                if original_display:
                    _os.environ["DISPLAY"] = original_display

            if Path(path).exists():
                if verbose:
                    print(f"📸 URL: {path}")
                return path

        except ImportError:
            if verbose:
                print(
                    "⚠️  Playwright not installed: pip install 'scitex[capture-browser]'"
                )
            pass  # Try PowerShell fallback
        except Exception as e:
            if verbose:
                print(f"⚠️  Playwright failed: {e}")
            pass  # Try PowerShell fallback

        # For WSL: Fallback to Windows-side browser
        if sys.platform == "linux" and "microsoft" in os.uname().release.lower():
            try:
                import base64
                import json
                import subprocess

                # Generate output path
                if path is None:
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    url_slug = (
                        url.replace("://", "_").replace("/", "_").replace(":", "_")[:30]
                    )
                    path = f"~/.scitex/capture/{timestamp_str}-url-{url_slug}.jpg"

                path = os.path.expanduser(path)

                if verbose:
                    print(f"📸 Capturing URL on Windows: {url}")

                # Use PowerShell script on Windows host
                script_dir = Path(__file__).parent / "powershell"
                script_path = script_dir / "capture_url.ps1"

                if script_path.exists():
                    # Find PowerShell
                    ps_paths = [
                        "powershell.exe",
                        "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                    ]
                    ps_exe = None
                    for p in ps_paths:
                        try:
                            result = subprocess.run(
                                [p, "-Command", "echo test"],
                                capture_output=True,
                                timeout=1,
                            )
                            if result.returncode == 0:
                                ps_exe = p
                                break
                        except:
                            continue

                    if ps_exe:
                        cmd = [
                            ps_exe,
                            "-NoProfile",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-File",
                            str(script_path),
                            "-Url",
                            url,
                            "-WaitSeconds",
                            str(url_wait),
                            "-WindowWidth",
                            str(url_width),
                            "-WindowHeight",
                            str(url_height),
                        ]

                        result = subprocess.run(
                            cmd, capture_output=True, text=True, timeout=30
                        )

                        if result.returncode == 0 and result.stdout.strip():
                            # Parse JSON
                            lines = result.stdout.strip().split("\n")
                            for line in lines:
                                if line.strip().startswith("{"):
                                    data = json.loads(line)
                                    if data.get("Success"):
                                        img_data = base64.b64decode(
                                            data.get("Base64Data", "")
                                        )

                                        # Save as JPEG
                                        try:
                                            import io

                                            from PIL import Image

                                            img = Image.open(io.BytesIO(img_data))
                                            if img.mode == "RGBA":
                                                rgb_img = Image.new(
                                                    "RGB",
                                                    img.size,
                                                    (255, 255, 255),
                                                )
                                                rgb_img.paste(img, mask=img.split()[3])
                                                img = rgb_img
                                            img.save(
                                                path,
                                                "JPEG",
                                                quality=quality,
                                                optimize=True,
                                            )

                                            if verbose:
                                                print(f"📸 URL: {path}")
                                            return path
                                        except ImportError:
                                            # Save as PNG fallback
                                            with open(
                                                path.replace(".jpg", ".png"),
                                                "wb",
                                            ) as f:
                                                f.write(img_data)
                                            return path.replace(".jpg", ".png")
                                    break

            except Exception as e:
                if verbose:
                    print(f"⚠️  PowerShell URL capture failed: {e}")

        # If all methods failed
        if verbose:
            print(
                "❌ URL capture failed - Playwright not available and PowerShell failed"
            )
        return None

    # Handle app-specific capture
    if app:
        info = _manager.get_info()
        windows = info.get("Windows", {}).get("Details", [])

        # Search for matching window
        app_lower = app.lower()
        matching_window = None

        for win in windows:
            process_name = win.get("ProcessName", "").lower()
            title = win.get("Title", "").lower()

            if app_lower in process_name or app_lower in title:
                matching_window = win
                break

        if matching_window:
            handle = matching_window.get("Handle")
            result_path = _manager.capture_window(handle, path)

            if result_path and verbose:
                print(f"📸 {matching_window.get('ProcessName')}: {result_path}")

            return result_path
        else:
            if verbose:
                print(f"❌ App '{app}' not found in visible windows")
            return None

    # Handle 'all' shorthand
    if all:
        capture_all = True

    # Take screenshot first to analyze it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    temp_dir = "/tmp/scitex_capture_temp"
    Path(temp_dir).mkdir(exist_ok=True)

    # Take screenshot to temp location
    use_jpeg = (
        True if path is None or path.lower().endswith((".jpg", ".jpeg")) else False
    )
    worker = ScreenshotWorker(
        output_dir=temp_dir,
        use_jpeg=use_jpeg,
        jpeg_quality=quality,
        verbose=verbose,  # Use the verbose parameter passed by user
    )

    worker.session_id = "capture"
    worker.screenshot_count = 0
    worker.monitor = monitor_id
    worker.capture_all = capture_all
    temp_path = worker._take_screenshot()

    if not temp_path:
        return None

    # Detect category if auto_categorize enabled
    category = "stdout"
    if auto_categorize:
        # First check if we're in an exception context
        if _is_in_exception_context():
            category = "stderr"
            # Add exception info to message
            import traceback

            exc_info = traceback.format_exc(limit=3)
            if message:
                message = f"{message}\n{exc_info}"
            else:
                message = exc_info
        else:
            # Try visual detection
            category = _detect_category(temp_path)

    # Build monitor/scope tag for filename
    scope_tag = ""
    if capture_all:
        scope_tag = "-all-monitors"
    elif monitor_id > 0:
        scope_tag = f"-monitor{monitor_id}"
    # monitor_id=0 (primary) gets no tag for cleaner default names

    # Normalize message for filename
    normalized_msg = ""
    if message:
        # Remove special chars, keep only alphanumeric and spaces
        import re

        normalized = re.sub(r"[^\w\s-]", "", message.split("\n")[0])  # First line only
        normalized = re.sub(r"[-\s]+", "-", normalized).strip("-")
        normalized_msg = f"-{normalized[:50]}" if normalized else ""  # Limit length

    # Add category suffix
    category_suffix = f"-{category}"

    # Handle path with category and message
    if path is None:
        # Include monitor/scope info in filename
        path = f"~/.scitex/capture/<timestamp><scope><message><category_suffix>.jpg"

    # Expand user home
    path = os.path.expanduser(path)

    # Replace placeholders
    if "<timestamp>" in path:
        path = path.replace("<timestamp>", timestamp)
    if "<scope>" in path:
        path = path.replace("<scope>", scope_tag)
    if "<message>" in path:
        path = path.replace("<message>", normalized_msg)
    if "<category_suffix>" in path:
        path = path.replace("<category_suffix>", category_suffix)

    # Ensure directory exists
    output_dir = Path(path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Move to final location
    final_path = Path(path)
    Path(temp_path).rename(final_path)

    # Add message with category as metadata
    if message or category != "stdout":
        metadata = (
            f"[{category.upper()}] {message}" if message else f"[{category.upper()}]"
        )
        _add_message_metadata(str(final_path), metadata)

    # Manage cache size (remove old files if needed)
    cache_dir = Path(os.path.expanduser("~/.scitex/capture"))
    if cache_dir.exists():
        _manage_cache_size(cache_dir, max_cache_gb)

    # Print path for user feedback (useful in interactive sessions)
    final_path_str = str(final_path)
    if verbose:
        try:
            if category == "stderr":
                print(f"📸 stderr: {final_path_str}")
            else:
                print(f"📸 stdout: {final_path_str}")
        except:
            # In case print fails in some environments
            pass

    return final_path_str


def take_screenshot(
    output_path: str = None, jpeg: bool = True, quality: int = 85
) -> Optional[str]:
    """
    Take a single screenshot (simple interface).

    Parameters
    ----------
    output_path : str, optional
        Where to save the screenshot
    jpeg : bool
        Use JPEG format (True) or PNG (False)
    quality : int
        JPEG quality (1-100)

    Returns
    -------
    str or None
        Path to saved screenshot
    """
    return _manager.take_single_screenshot(output_path, jpeg, quality)


def start_monitor(
    output_dir: str = "~/.scitex/capture/",
    interval: float = 1.0,
    jpeg: bool = True,
    quality: int = 60,
    on_capture=None,
    on_error=None,
    verbose: bool = True,
    monitor_id: int = 0,
    capture_all: bool = False,
) -> ScreenshotWorker:
    """
    Start continuous screenshot monitoring.

    Parameters
    ----------
    output_dir : str
        Directory for screenshots (default: ~/.scitex/capture/)
    interval : float
        Seconds between captures
    jpeg : bool
        Use JPEG compression
    quality : int
        JPEG quality (1-100)
    on_capture : callable, optional
        Function called with filepath after each capture
    on_error : callable, optional
        Function called with exception on errors
    verbose : bool
        Print status messages
    monitor_id : int
        Monitor number to capture (0-based index, default: 0 for primary monitor)
    capture_all : bool
        If True, capture all monitors combined (default: False)

    Returns
    -------
    ScreenshotWorker
        The worker instance

    Examples
    --------
    >>> # Simple monitoring
    >>> capture.start()

    >>> # With event hooks
    >>> capture.start(
    ...     on_capture=lambda path: print(f"Saved: {path}"),
    ...     on_error=lambda e: logging.error(e)
    ... )

    >>> # Detect specific screen content
    >>> def check_error_dialog(path):
    ...     if "error" in analyze_image(path):
    ...         send_alert(f"Error detected: {path}")
    >>> capture.start(on_capture=check_error_dialog)
    """
    # Expand user home directory
    output_dir = os.path.expanduser(output_dir)

    return _manager.start_capture(
        output_dir=output_dir,
        interval=interval,
        jpeg=jpeg,
        quality=quality,
        on_capture=on_capture,
        on_error=on_error,
        verbose=verbose,
        monitor_id=monitor_id,
        capture_all=capture_all,
    )


def stop_monitor():
    """Stop continuous screenshot monitoring."""
    _manager.stop_capture()


def _is_in_exception_context() -> bool:
    """
    Check if we're currently in an exception handler.
    """
    import sys

    # Check if there's an active exception
    exc_info = sys.exc_info()
    return exc_info[0] is not None


def _detect_category(filepath: str) -> str:
    """
    Detect screenshot category based on content.
    Simple heuristic based on common error indicators.
    """
    try:
        # Try OCR-based detection if available
        from PIL import Image

        img = Image.open(filepath)

        # Simple color-based heuristics
        # Red dominant = likely error
        # Yellow/orange dominant = likely warning
        pixels = img.convert("RGB").getdata()
        red_count = sum(1 for r, g, b in pixels if r > 200 and g < 100 and b < 100)
        yellow_count = sum(1 for r, g, b in pixels if r > 200 and g > 150 and b < 100)

        total_pixels = len(pixels)
        red_ratio = red_count / total_pixels if total_pixels > 0 else 0
        yellow_ratio = yellow_count / total_pixels if total_pixels > 0 else 0

        # Thresholds for detection
        if red_ratio > 0.05:  # More than 5% red pixels
            return "error"
        elif yellow_ratio > 0.05:  # More than 5% yellow pixels
            return "warning"

    except:
        pass

    # Check filename for common error keywords
    filename_lower = str(filepath).lower()
    if any(word in filename_lower for word in ["error", "fail", "exception", "crash"]):
        return "stderr"
    elif any(word in filename_lower for word in ["warn", "alert", "caution"]):
        return "stderr"  # Warnings also go to stderr

    return "stdout"


def _add_message_metadata(filepath: str, message: str):
    """Add message as metadata to image file."""
    try:
        # Try to add EXIF comment using PIL
        from PIL import Image

        img = Image.open(filepath)

        # Add comment to image metadata
        exif = img.getexif()
        exif[0x9286] = message  # UserComment EXIF tag

        # Save with metadata
        img.save(filepath, exif=exif)
    except:
        # If PIL not available, create companion text file
        text_path = Path(filepath).with_suffix(".txt")
        text_path.write_text(f"{datetime.now().isoformat()}: {message}\n")


# Convenience exports
__all__ = [
    "capture",
    "take_screenshot",
    "start_monitor",
    "stop_monitor",
]

# EOF
