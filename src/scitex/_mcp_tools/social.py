#!/usr/bin/env python3
# Timestamp: 2026-01-22
# File: /home/ywatanabe/proj/scitex-code/src/scitex/_mcp_tools/social.py

"""Social media module tools for FastMCP unified server.

All MCP tools delegate to socialia CLI for reproducibility.
Each tool returns the CLI command used, enabling human reproduction.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Optional


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, default=str)


def _run_socialia_cli(*args: str) -> dict:
    """Run socialia CLI and return structured result."""
    cmd = [sys.executable, "-m", "socialia", "--json", *args]
    cli_command = f"socialia {' '.join(args)}"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                data["cli_command"] = cli_command
                return data
            except json.JSONDecodeError:
                return {
                    "success": True,
                    "output": result.stdout,
                    "cli_command": cli_command,
                }
        else:
            return {
                "success": False,
                "error": result.stderr or result.stdout,
                "cli_command": cli_command,
            }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Command timed out",
            "cli_command": cli_command,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "cli_command": cli_command,
        }


def register_social_tools(mcp) -> None:
    """Register social media tools with FastMCP server."""

    @mcp.tool()
    async def social_post(
        platform: str,
        text: str,
        reply_to: Optional[str] = None,
        quote: Optional[str] = None,
        subreddit: Optional[str] = None,
        title: Optional[str] = None,
        dry_run: bool = False,
    ) -> str:
        """[social] Post content to a social media platform (twitter, linkedin, reddit, youtube)."""
        args = ["post", platform, text]

        if reply_to:
            args.extend(["--reply-to", reply_to])
        if quote:
            args.extend(["--quote", quote])
        if subreddit and platform == "reddit":
            args.extend(["--subreddit", subreddit])
        if title:
            args.extend(["--title", title])
        if dry_run:
            args.append("--dry-run")

        result = _run_socialia_cli(*args)
        return _json(result)

    @mcp.tool()
    async def social_delete(
        platform: str,
        post_id: str,
    ) -> str:
        """[social] Delete a post from a platform (twitter, linkedin, reddit)."""
        result = _run_socialia_cli("delete", platform, post_id)
        return _json(result)

    @mcp.tool()
    async def social_status() -> str:
        """[social] Check social media configuration and authentication status."""
        result = _run_socialia_cli("status")
        return _json(result)

    @mcp.tool()
    async def social_analytics(
        platform: str,
        days: int = 7,
    ) -> str:
        """[social] Get analytics for a platform (twitter, youtube, ga)."""
        result = _run_socialia_cli("analytics", platform, "--days", str(days))
        return _json(result)

    @mcp.tool()
    async def social_thread(
        platform: str,
        posts: list[str],
        delay: int = 2,
        dry_run: bool = False,
    ) -> str:
        """[social] Post a thread of connected posts. Posts are list of strings."""
        import tempfile
        from pathlib import Path

        # Write posts to temp file (socialia expects file input)
        thread_content = "\n---\n".join(posts)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(thread_content)
            temp_path = f.name

        try:
            args = ["thread", platform, "--file", temp_path, "--delay", str(delay)]
            if dry_run:
                args.append("--dry-run")
            result = _run_socialia_cli(*args)
            result["thread_posts"] = posts
            return _json(result)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    @mcp.tool()
    async def social_check_availability() -> str:
        """[social] Check if socialia is installed and list available platforms."""
        try:
            from scitex.social import (
                SOCIALIA_AVAILABLE,
                __socialia_version__,
            )

            if SOCIALIA_AVAILABLE:
                return _json(
                    {
                        "available": True,
                        "version": __socialia_version__,
                        "platforms": ["twitter", "linkedin", "reddit", "youtube"],
                        "analytics": ["twitter", "youtube", "ga"],
                    }
                )
            else:
                return _json(
                    {
                        "available": False,
                        "error": "socialia not installed",
                        "install_command": "pip install socialia",
                    }
                )
        except Exception as e:
            return _json(
                {
                    "available": False,
                    "error": str(e),
                }
            )


# EOF
