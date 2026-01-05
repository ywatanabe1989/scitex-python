# Add your tests here

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/audio/_cross_process_lock.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # Timestamp: "2025-12-27 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex-code/src/scitex/audio/_cross_process_lock.py
# # ----------------------------------------
# 
# """
# Cross-process FIFO lock for audio playback.
# 
# Ensures only one MCP server instance can play audio at a time,
# providing true FIFO ordering across all Claude Code sessions.
# """
# 
# from __future__ import annotations
# 
# import fcntl
# import os
# import time
# from contextlib import contextmanager
# from pathlib import Path
# 
# __all__ = ["AudioPlaybackLock", "acquire_audio_lock"]
# 
# # Lock file location
# SCITEX_BASE_DIR = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
# LOCK_FILE = SCITEX_BASE_DIR / "audio" / ".audio_playback.lock"
# 
# 
# class AudioPlaybackLock:
#     """Cross-process lock for sequential audio playback.
# 
#     Uses fcntl.flock() for POSIX file locking to ensure
#     only one process can play audio at a time.
#     """
# 
#     def __init__(self, lock_file: Path | None = None):
#         self.lock_file = lock_file or LOCK_FILE
#         self._fd: int | None = None
#         self._acquired = False
# 
#     def _ensure_lock_dir(self):
#         """Ensure the lock file directory exists."""
#         self.lock_file.parent.mkdir(parents=True, exist_ok=True)
# 
#     def acquire(self, timeout: float | None = None) -> bool:
#         """Acquire the audio playback lock.
# 
#         Args:
#             timeout: Maximum time to wait in seconds.
#                      None means wait indefinitely.
# 
#         Returns:
#             True if lock acquired, False if timeout.
#         """
#         self._ensure_lock_dir()
# 
#         # Open or create the lock file
#         self._fd = os.open(
#             str(self.lock_file),
#             os.O_RDWR | os.O_CREAT,
#             0o644,
#         )
# 
#         start_time = time.time()
# 
#         while True:
#             try:
#                 # Try to acquire exclusive lock (non-blocking)
#                 fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
#                 self._acquired = True
# 
#                 # Write PID to lock file for debugging
#                 os.ftruncate(self._fd, 0)
#                 os.lseek(self._fd, 0, os.SEEK_SET)
#                 os.write(self._fd, f"{os.getpid()}\n".encode())
# 
#                 return True
# 
#             except OSError:
#                 # Lock is held by another process
#                 if timeout is not None:
#                     elapsed = time.time() - start_time
#                     if elapsed >= timeout:
#                         self._cleanup()
#                         return False
# 
#                 # Wait a bit before retrying
#                 time.sleep(0.1)
# 
#     def release(self):
#         """Release the audio playback lock."""
#         if self._fd is not None and self._acquired:
#             try:
#                 fcntl.flock(self._fd, fcntl.LOCK_UN)
#             except OSError:
#                 pass
#             self._acquired = False
#         self._cleanup()
# 
#     def _cleanup(self):
#         """Clean up file descriptor."""
#         if self._fd is not None:
#             try:
#                 os.close(self._fd)
#             except OSError:
#                 pass
#             self._fd = None
# 
#     def __enter__(self):
#         self.acquire()
#         return self
# 
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.release()
#         return False
# 
# 
# @contextmanager
# def acquire_audio_lock(timeout: float | None = 60.0):
#     """Context manager for acquiring the audio playback lock.
# 
#     Args:
#         timeout: Maximum time to wait in seconds (default: 60s).
# 
#     Yields:
#         True if lock was acquired.
# 
#     Raises:
#         TimeoutError: If lock could not be acquired within timeout.
#     """
#     lock = AudioPlaybackLock()
#     try:
#         if not lock.acquire(timeout=timeout):
#             raise TimeoutError(f"Could not acquire audio lock within {timeout}s")
#         yield True
#     finally:
#         lock.release()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/audio/_cross_process_lock.py
# --------------------------------------------------------------------------------
