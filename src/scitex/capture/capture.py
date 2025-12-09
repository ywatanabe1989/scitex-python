#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-18 09:55:59 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/capture/capture.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/capture/capture.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Core screenshot capture functionality.
Optimized for WSL to Windows host screen capture.
"""

import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class ScreenshotWorker:
    """
    Independent worker thread for continuous screenshot capture.
    Takes screenshots at configurable intervals with compression options.
    """

    def __init__(
        self,
        output_dir: str = "/tmp/scitex_capture_screenshots",
        interval_sec: float = 1.0,
        verbose: bool = False,
        use_jpeg: bool = True,
        jpeg_quality: int = 60,
        on_capture=None,
        on_error=None,
    ):
        """
        Initialize screenshot worker.

        Parameters
        ----------
        output_dir : str
            Directory for saving screenshots
        interval_sec : float
            Seconds between screenshots (default: 1.0)
        verbose : bool
            Print screenshot paths in runtime log
        use_jpeg : bool
            Use JPEG compression for smaller files (default: True)
        jpeg_quality : int
            JPEG quality 1-100, lower = smaller files (default: 60)
        on_capture : callable, optional
            Callback function called with filepath after each capture
        on_error : callable, optional
            Callback function called with exception on errors
        """
        self.output_dir = Path(output_dir)
        self.interval_sec = interval_sec
        self.verbose = verbose
        self.use_jpeg = use_jpeg
        self.jpeg_quality = jpeg_quality
        self.on_capture = on_capture
        self.on_error = on_error

        # Worker state
        self.running = False
        self.worker_thread = None
        self.screenshot_count = 0
        self.session_id = None

        # Monitor capture settings
        self.monitor = 0  # Default to primary monitor (0-based indexing)
        self.capture_all = False  # Default to single monitor

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start(self, session_id: str = None):
        """Start the screenshot worker thread."""
        if self.running:
            if self.verbose:
                print("⚠️ Worker already running")
            return

        self.running = True
        self.screenshot_count = 0
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Start worker thread
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="ScreenshotWorker"
        )
        self.worker_thread.start()

        if self.verbose:
            ext = "jpg" if self.use_jpeg else "png"
            print(
                f"📸 Started: {self.output_dir}/{self.session_id}_NNNN_*.{ext} (interval: {self.interval_sec}s)"
            )

    def stop(self):
        """Stop the screenshot worker thread."""
        if not self.running:
            return

        self.running = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)

        if self.verbose:
            print(
                f"📸 Stopped: {self.screenshot_count} screenshots in {self.output_dir}"
            )

    def _worker_loop(self):
        """Main worker loop that takes screenshots."""

        next_capture_time = time.time()

        while self.running:
            current_time = time.time()

            # Check if it's time for next capture
            if current_time >= next_capture_time:
                try:
                    screenshot_path = self._take_screenshot()

                    if screenshot_path:
                        if self.verbose:
                            # Simple one-line output
                            print(f"📸 {screenshot_path}")

                        # Call on_capture callback if provided
                        if self.on_capture:
                            try:
                                self.on_capture(screenshot_path)
                            except Exception as cb_error:
                                if self.verbose:
                                    print(f"⚠️ Callback error: {cb_error}")

                except Exception as e:
                    if self.verbose:
                        print(f"❌ Error: {e}")

                    # Call on_error callback if provided
                    if self.on_error:
                        try:
                            self.on_error(e)
                        except Exception as cb_error:
                            if self.verbose:
                                print(f"⚠️ Error callback failed: {cb_error}")

                # Schedule next capture
                next_capture_time = current_time + self.interval_sec

            # Short sleep to avoid busy waiting, but allow responsive stopping
            time.sleep(0.01)

    def _take_screenshot(self) -> Optional[str]:
        """Take a single screenshot."""
        try:
            # Generate filename with timestamp
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            ext = "jpg" if self.use_jpeg else "png"
            filename = (
                f"{self.session_id}_{self.screenshot_count:04d}_{timestamp}.{ext}"
            )
            filepath = self.output_dir / filename

            # Try Windows PowerShell method for WSL
            if self._is_wsl():
                if self._capture_windows_screen(
                    filepath,
                    monitor=self.monitor,
                    capture_all=self.capture_all,
                ):
                    self.screenshot_count += 1
                    return str(filepath)

            # Fallback to native screenshot tools
            if self._capture_native_screen(filepath):
                self.screenshot_count += 1
                return str(filepath)

            return None

        except Exception as e:
            if self.verbose:
                print(f"❌ Screenshot failed: {e}")
            return None

    def _is_wsl(self) -> bool:
        """Check if running in WSL."""
        return sys.platform == "linux" and "microsoft" in os.uname().release.lower()

    def _capture_windows_screen(
        self, filepath: Path, monitor: int = 1, capture_all: bool = False
    ) -> bool:
        """Capture Windows host screen from WSL with DPI awareness using external PowerShell scripts.

        Args:
            filepath: Path to save the screenshot
            monitor: Monitor number to capture (1-based index)
            capture_all: If True, capture all monitors combined
        """
        try:
            # Try using external PowerShell script first
            script_dir = Path(__file__).parent / "powershell"
            if capture_all:
                script_path = script_dir / "capture_all_monitors.ps1"
            else:
                script_path = script_dir / "capture_single_monitor.ps1"

            # Check if script exists
            if script_path.exists():
                # Find PowerShell executable
                ps_paths = [
                    "powershell.exe",
                    "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                    "/mnt/c/Windows/SysWOW64/WindowsPowerShell/v1.0/powershell.exe",
                ]

                ps_exe = None
                for path in ps_paths:
                    try:
                        test_result = subprocess.run(
                            [path, "-Command", "echo test"],
                            capture_output=True,
                            timeout=1,
                        )
                        if test_result.returncode == 0:
                            ps_exe = path
                            break
                    except:
                        continue

                if ps_exe:
                    # Build PowerShell command
                    if capture_all:
                        cmd = [
                            ps_exe,
                            "-NoProfile",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-File",
                            str(script_path),
                            "-OutputFormat",
                            "base64",
                        ]
                    else:
                        # Pass 0-based monitor index directly to PowerShell
                        cmd = [
                            ps_exe,
                            "-NoProfile",
                            "-ExecutionPolicy",
                            "Bypass",
                            "-File",
                            str(script_path),
                            "-MonitorNumber",
                            str(monitor),
                            "-OutputFormat",
                            "base64",
                        ]

                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=5
                    )

                    if result.returncode == 0 and result.stdout.strip():
                        # Decode base64 PNG data
                        import base64

                        png_data = base64.b64decode(result.stdout.strip())

                        # Save directly as JPEG if requested, otherwise as PNG
                        if self.use_jpeg:
                            try:
                                import io

                                from PIL import Image

                                img = Image.open(io.BytesIO(png_data))
                                # Convert RGBA to RGB for JPEG
                                if img.mode == "RGBA":
                                    rgb_img = Image.new(
                                        "RGB", img.size, (255, 255, 255)
                                    )
                                    rgb_img.paste(
                                        img, mask=img.split()[3]
                                    )  # Use alpha channel as mask
                                    img = rgb_img
                                img.save(
                                    str(filepath),
                                    "JPEG",
                                    quality=self.jpeg_quality,
                                    optimize=True,
                                )
                            except ImportError:
                                # PIL not available, save as PNG
                                with open(str(filepath), "wb") as f:
                                    f.write(png_data)
                        else:
                            with open(str(filepath), "wb") as f:
                                f.write(png_data)

                        return filepath.exists()

            # Fallback to inline script
            return self._capture_windows_screen_inline(filepath)

        except Exception as e:
            if self.verbose:
                print(f"❌ Windows screen capture error: {e}")
                import traceback

                traceback.print_exc()
        return False

    def _capture_windows_screen_inline(self, filepath: Path) -> bool:
        """Fallback inline PowerShell capture (when .ps1 files not available)."""
        try:
            if self.verbose:
                print("🔍 Attempting inline PowerShell capture...")
            # Use base64 encoding to avoid path issues (most reliable for WSL)
            # Now with DPI awareness for proper high-resolution capture
            ps_script = """
            Add-Type -AssemblyName System.Windows.Forms
            Add-Type -AssemblyName System.Drawing

            # Enable DPI awareness for proper high-resolution capture
            Add-Type @'
            using System;
            using System.Runtime.InteropServices;
            public class User32 {
                [DllImport("user32.dll")]
                public static extern bool SetProcessDPIAware();
            }
'@
            $null = [User32]::SetProcessDPIAware()

            $screen = [System.Windows.Forms.Screen]::PrimaryScreen
            $bitmap = New-Object System.Drawing.Bitmap $screen.Bounds.Width, $screen.Bounds.Height
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)

            # Set high quality rendering
            $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
            $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
            $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality

            $graphics.CopyFromScreen($screen.Bounds.X, $screen.Bounds.Y, 0, 0, $bitmap.Size)

            $stream = New-Object System.IO.MemoryStream
            $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
            $bytes = $stream.ToArray()
            [Convert]::ToBase64String($bytes)

            $graphics.Dispose()
            $bitmap.Dispose()
            $stream.Dispose()
            """

            # Find PowerShell executable
            ps_paths = [
                # Check PATH first (might be in .win-bin or similar)
                "powershell.exe",
                # Standard WSL path
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
                # Alternative locations
                "/mnt/c/Windows/SysWOW64/WindowsPowerShell/v1.0/powershell.exe",
            ]

            ps_exe = None
            for path in ps_paths:
                try:
                    # Just check if the file exists and is executable
                    test_path = (
                        Path(path) if not path.startswith("/mnt/") else Path(path)
                    )
                    if path == "powershell.exe":
                        # In PATH - use it directly
                        ps_exe = path
                        if self.verbose:
                            print(f"✓ Found PowerShell in PATH")
                        break
                    elif test_path.exists() or Path(path).exists():
                        ps_exe = path
                        if self.verbose:
                            print(f"✓ Found PowerShell at {path}")
                        break
                except:
                    continue

            if not ps_exe:
                if self.verbose:
                    print("❌ PowerShell executable not found")
                return False

            if self.verbose:
                print(f"✓ Using PowerShell: {ps_exe}")

            # Execute PowerShell
            cmd = [ps_exe, "-NoProfile", "-Command", ps_script]

            if self.verbose:
                print("🔄 Executing PowerShell script...")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

                if self.verbose:
                    print(f"✓ PowerShell return code: {result.returncode}")
                    if result.stderr:
                        print(f"PowerShell stderr: {result.stderr[:500]}")
                    if result.stdout:
                        print(f"✓ PowerShell stdout length: {len(result.stdout)} chars")
            except subprocess.TimeoutExpired as e:
                if self.verbose:
                    print(f"❌ PowerShell timeout after 10s")
                return False

            if result.returncode == 0 and result.stdout.strip():
                # Decode base64 PNG data
                import base64

                png_data = base64.b64decode(result.stdout.strip())

                # Save directly as JPEG if requested, otherwise as PNG
                if self.use_jpeg:
                    try:
                        import io

                        from PIL import Image

                        # Load PNG from memory
                        img = Image.open(io.BytesIO(png_data))
                        # Convert RGBA to RGB for JPEG
                        if img.mode == "RGBA":
                            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[3])
                            img = rgb_img
                        # Save as JPEG with quality
                        img.save(
                            str(filepath),
                            "JPEG",
                            quality=self.jpeg_quality,
                            optimize=True,
                        )
                    except ImportError:
                        # PIL not available, save as PNG
                        with open(str(filepath), "wb") as f:
                            f.write(png_data)
                else:
                    # Save as PNG
                    with open(str(filepath), "wb") as f:
                        f.write(png_data)

                return filepath.exists()
        except Exception:
            pass
        return False

    def _capture_native_screen(self, filepath: Path) -> bool:
        """Capture screen using native tools."""
        try:
            # Try mss first
            try:
                import mss

                with mss.mss() as sct:
                    # Capture primary monitor
                    monitor = (
                        sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                    )
                    screenshot = sct.grab(monitor)

                    if self.use_jpeg:
                        # Convert to PIL for JPEG saving
                        from PIL import Image

                        img = Image.frombytes(
                            "RGB",
                            screenshot.size,
                            screenshot.bgra,
                            "raw",
                            "BGRX",
                        )
                        img.save(str(filepath), "JPEG", quality=self.jpeg_quality)
                    else:
                        mss.tools.to_png(
                            screenshot.rgb,
                            screenshot.size,
                            output=str(filepath),
                        )

                    return filepath.exists()
            except ImportError:
                pass

            # Try scrot
            if self.use_jpeg:
                cmd = [
                    "scrot",
                    "-z",
                    "-q",
                    str(self.jpeg_quality),
                    str(filepath),
                ]
            else:
                cmd = ["scrot", "-z", str(filepath)]

            result = subprocess.run(cmd, capture_output=True, timeout=2)
            return result.returncode == 0 and filepath.exists()

        except Exception as e:
            if self.verbose:
                print(f"❌ Native screen capture failed: {e}")
        return False

    def get_status(self) -> dict:
        """Get current worker status."""
        return {
            "running": self.running,
            "session_id": self.session_id,
            "screenshot_count": self.screenshot_count,
            "output_dir": str(self.output_dir),
            "interval_sec": self.interval_sec,
            "use_jpeg": self.use_jpeg,
            "jpeg_quality": self.jpeg_quality,
        }


class CaptureManager:
    """High-level interface for managing screen capture."""

    def __init__(self):
        self.worker = None

    def start_capture(
        self,
        output_dir: str = "/tmp/scitex_capture_screenshots",
        interval: float = 1.0,
        jpeg: bool = True,
        quality: int = 60,
        on_capture=None,
        on_error=None,
        verbose: bool = False,
        monitor_id: int = 0,
        capture_all: bool = False,
    ) -> ScreenshotWorker:
        """Start continuous screen capture."""
        if self.worker and self.worker.running:
            print("Capture already running")
            return self.worker

        self.worker = ScreenshotWorker(
            output_dir=output_dir,
            interval_sec=interval,
            use_jpeg=jpeg,
            jpeg_quality=quality,
            on_capture=on_capture,
            on_error=on_error,
            verbose=verbose,
        )
        # Set monitor parameters
        self.worker.monitor = monitor_id
        self.worker.capture_all = capture_all
        self.worker.start()
        return self.worker

    def stop_capture(self):
        """Stop screen capture."""
        if self.worker:
            self.worker.stop()
            self.worker = None

    def take_single_screenshot(
        self,
        output_path: str = None,
        jpeg: bool = True,
        quality: int = 85,
        monitor_id: int = 0,
        capture_all_monitors: bool = False,
    ) -> Optional[str]:
        """
        Take a single screenshot.

        Args:
            output_path: Path to save screenshot
            jpeg: Use JPEG compression
            quality: JPEG quality (1-100)
            monitor_id: Monitor index to capture (0-based)
            capture_all_monitors: Capture all monitors combined

        Returns:
            Path to saved screenshot
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = "jpg" if jpeg else "png"
            output_path = f"/tmp/screenshot_{timestamp}.{ext}"

        worker = ScreenshotWorker(
            output_dir=str(Path(output_path).parent),
            use_jpeg=jpeg,
            jpeg_quality=quality,
            verbose=False,
        )

        # Set monitor parameters
        worker.monitor = monitor_id
        worker.capture_all = capture_all_monitors

        # Take single screenshot
        worker.session_id = "single"
        worker.screenshot_count = 0
        screenshot_path = worker._take_screenshot()

        if screenshot_path:
            # Rename to desired path
            Path(screenshot_path).rename(output_path)
            return output_path

        return None

    def get_info(self) -> dict:
        """
        Enumerate all available monitors and virtual desktops.

        Returns:
            Dictionary with monitor information
        """
        try:
            script_dir = Path(__file__).parent / "powershell"
            script_path = script_dir / "detect_monitors_and_desktops.ps1"

            if not script_path.exists():
                return {"error": "Detection script not found"}

            # Find PowerShell
            ps_paths = [
                "powershell.exe",
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            ]

            ps_exe = None
            for path in ps_paths:
                try:
                    result = subprocess.run(
                        [path, "-Command", "echo test"],
                        capture_output=True,
                        timeout=1,
                    )
                    if result.returncode == 0:
                        ps_exe = path
                        break
                except:
                    continue

            if not ps_exe:
                return {"error": "PowerShell not found"}

            # Execute detection script
            cmd = [
                ps_exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                # Parse JSON from output (skip non-JSON lines)
                import json

                lines = result.stdout.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("{"):
                        return json.loads(line)

                return {"error": "No JSON in output"}
            else:
                return {
                    "error": (result.stderr if result.stderr else "Detection failed")
                }

        except Exception as e:
            return {"error": str(e)}

    def capture_window(
        self,
        window_handle: int,
        output_path: str = None,
        jpeg: bool = True,
        quality: int = 85,
    ) -> Optional[str]:
        """
        Capture a specific window by its handle.

        Args:
            window_handle: Window handle (from get_info)
            output_path: Path to save screenshot
            jpeg: Use JPEG compression
            quality: JPEG quality (1-100)

        Returns:
            Path to saved screenshot or None
        """
        try:
            script_dir = Path(__file__).parent / "powershell"
            script_path = script_dir / "capture_window_by_handle.ps1"

            if not script_path.exists():
                return None

            # Find PowerShell
            ps_paths = [
                "powershell.exe",
                "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            ]

            ps_exe = None
            for path in ps_paths:
                try:
                    result = subprocess.run(
                        [path, "-Command", "echo test"],
                        capture_output=True,
                        timeout=1,
                    )
                    if result.returncode == 0:
                        ps_exe = path
                        break
                except:
                    continue

            if not ps_exe:
                return None

            # Execute window capture script
            cmd = [
                ps_exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-WindowHandle",
                str(window_handle),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                # Parse JSON from output
                import base64
                import json

                lines = result.stdout.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line.startswith("{"):
                        data = json.loads(line)
                        break
                else:
                    return None

                if not data.get("Success"):
                    return None

                # Generate output path if not provided
                if output_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = "jpg" if jpeg else "png"
                    output_path = f"/tmp/window_{window_handle}_{timestamp}.{ext}"

                # Decode base64 image
                img_data = base64.b64decode(data.get("Base64Data", ""))

                # Save as JPEG or PNG
                if jpeg:
                    try:
                        import io

                        from PIL import Image

                        img = Image.open(io.BytesIO(img_data))
                        if img.mode == "RGBA":
                            rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[3])
                            img = rgb_img
                        img.save(output_path, "JPEG", quality=quality, optimize=True)
                    except ImportError:
                        output_path = output_path.replace(".jpg", ".png").replace(
                            ".jpeg", ".png"
                        )
                        with open(output_path, "wb") as f:
                            f.write(img_data)
                else:
                    with open(output_path, "wb") as f:
                        f.write(img_data)

                return output_path if Path(output_path).exists() else None

        except Exception as e:
            return None


# EOF
