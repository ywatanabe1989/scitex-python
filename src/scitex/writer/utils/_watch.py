#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/watch.py

"""
Watch mode for auto-recompilation.

Monitors file changes and triggers automatic recompilation.
"""

import subprocess
from pathlib import Path
from typing import Optional, Callable

from scitex.logging import getLogger

logger = getLogger(__name__)


def watch_manuscript(
    project_dir: Path,
    interval: int = 2,
    on_compile: Optional[Callable] = None,
    timeout: Optional[int] = None,
) -> None:
    """
    Watch and auto-recompile manuscript on file changes.

    Args:
        project_dir: Path to writer project directory
        interval: Check interval in seconds
        on_compile: Callback function called after each compilation
        timeout: Optional timeout in seconds (None = infinite)

    Examples:
        >>> from pathlib import Path
        >>> def on_change():
        ...     print("Recompiled!")
        >>> watch_manuscript(Path("/path/to/project"), on_compile=on_change)
    """
    # Get compile script from project directory
    compile_script = project_dir / "compile"

    if not compile_script.exists():
        logger.error(f"compile script not found: {compile_script}")
        return

    # Build watch command
    cmd = [str(compile_script), "-m", "-w"]

    logger.info(f"Starting watch mode for {project_dir}")
    logger.info("Press Ctrl+C to stop")

    try:
        # Run watch script
        process = subprocess.Popen(
            cmd,
            cwd=project_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line-buffered
        )

        # Stream output
        for line in iter(process.stdout.readline, ""):
            if line:
                print(line.rstrip())

                # Call callback on compilation events
                if on_compile and "Compilation" in line:
                    try:
                        on_compile()
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

        process.wait(timeout=timeout)

    except KeyboardInterrupt:
        logger.info("\nWatch mode stopped by user")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    except Exception as e:
        logger.error(f"Watch mode error: {e}")
        if process:
            process.terminate()


__all__ = ["watch_manuscript"]

# EOF
