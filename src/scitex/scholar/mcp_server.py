#!/usr/bin/env python3
# Timestamp: 2026-01-08
# File: src/scitex/scholar/mcp_server.py
# ----------------------------------------

"""
MCP Server for SciTeX Scholar - Scientific Literature Management

Provides tools for:
- Searching papers across multiple databases
- Resolving DOIs from paper titles
- Enriching BibTeX with metadata
- Downloading PDFs with institutional access
- Managing paper libraries
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

# Graceful MCP dependency handling
try:
    import mcp.types as types
    from mcp.server import NotificationOptions, Server
    from mcp.server.models import InitializationOptions
    from mcp.server.stdio import stdio_server

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    types = None  # type: ignore
    Server = None  # type: ignore
    NotificationOptions = None  # type: ignore
    InitializationOptions = None  # type: ignore
    stdio_server = None  # type: ignore

__all__ = ["ScholarServer", "main", "MCP_AVAILABLE"]

# Directory configuration
SCITEX_BASE_DIR = Path(os.getenv("SCITEX_DIR", Path.home() / ".scitex"))
SCITEX_SCHOLAR_DIR = SCITEX_BASE_DIR / "scholar"


def get_scholar_dir() -> Path:
    """Get the scholar data directory."""
    SCITEX_SCHOLAR_DIR.mkdir(parents=True, exist_ok=True)
    return SCITEX_SCHOLAR_DIR


class ScholarServer:
    """MCP Server for Scientific Literature Management."""

    def __init__(self):
        self.server = Server("scitex-scholar")
        self._scholar_instance = None
        self.setup_handlers()

    @property
    def scholar(self):
        """Lazy-load Scholar instance."""
        if self._scholar_instance is None:
            try:
                from scitex.scholar import Scholar

                self._scholar_instance = Scholar()
            except ImportError as e:
                raise RuntimeError(f"Scholar module not available: {e}")
        return self._scholar_instance

    def setup_handlers(self):
        """Set up MCP server handlers."""
        from ._mcp_handlers import (
            authenticate_handler,
            download_pdf_handler,
            download_pdfs_batch_handler,
            enrich_bibtex_handler,
            export_papers_handler,
            get_library_status_handler,
            parse_bibtex_handler,
            resolve_dois_handler,
            resolve_openurls_handler,
            search_papers_handler,
            validate_pdfs_handler,
        )
        from ._mcp_tool_schemas import get_tool_schemas

        @self.server.list_tools()
        async def handle_list_tools():
            return get_tool_schemas()

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            # Search tools
            if name == "search_papers":
                return await self._wrap_result(search_papers_handler(**arguments))

            # DOI Resolution
            elif name == "resolve_dois":
                return await self._wrap_result(resolve_dois_handler(**arguments))

            # BibTeX Enrichment
            elif name == "enrich_bibtex":
                return await self._wrap_result(enrich_bibtex_handler(**arguments))

            # PDF Download
            elif name == "download_pdf":
                return await self._wrap_result(download_pdf_handler(**arguments))

            elif name == "download_pdfs_batch":
                return await self._wrap_result(download_pdfs_batch_handler(**arguments))

            # Library Status
            elif name == "get_library_status":
                return await self._wrap_result(get_library_status_handler(**arguments))

            # Parse BibTeX
            elif name == "parse_bibtex":
                return await self._wrap_result(parse_bibtex_handler(**arguments))

            # Validate PDFs
            elif name == "validate_pdfs":
                return await self._wrap_result(validate_pdfs_handler(**arguments))

            # OpenURL Resolution
            elif name == "resolve_openurls":
                return await self._wrap_result(resolve_openurls_handler(**arguments))

            # Authentication
            elif name == "authenticate":
                return await self._wrap_result(authenticate_handler(**arguments))

            # Export
            elif name == "export_papers":
                return await self._wrap_result(export_papers_handler(**arguments))

            else:
                raise ValueError(f"Unknown tool: {name}")

        @self.server.list_resources()
        async def handle_list_resources():
            """List available library resources."""
            scholar_dir = get_scholar_dir()
            library_dir = scholar_dir / "library"

            if not library_dir.exists():
                return []

            resources = []

            # List projects as resources
            for project_dir in library_dir.iterdir():
                if project_dir.is_dir() and not project_dir.name.startswith("."):
                    pdf_count = len(list(project_dir.rglob("*.pdf")))
                    resources.append(
                        types.Resource(
                            uri=f"scholar://library/{project_dir.name}",
                            name=f"Library: {project_dir.name}",
                            description=f"Paper library with {pdf_count} PDFs",
                            mimeType="application/json",
                        )
                    )

            # List recent BibTeX files
            for bib_file in sorted(
                scholar_dir.rglob("*.bib"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:10]:
                mtime = datetime.fromtimestamp(bib_file.stat().st_mtime)
                resources.append(
                    types.Resource(
                        uri=f"scholar://bibtex/{bib_file.name}",
                        name=bib_file.name,
                        description=f"BibTeX from {mtime.strftime('%Y-%m-%d %H:%M')}",
                        mimeType="application/x-bibtex",
                    )
                )

            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str):
            """Read a library resource."""
            import json

            if uri.startswith("scholar://library/"):
                project_name = uri.replace("scholar://library/", "")
                library_dir = get_scholar_dir() / "library" / project_name

                if not library_dir.exists():
                    raise ValueError(f"Project not found: {project_name}")

                # Gather project info
                metadata_files = list(library_dir.rglob("metadata.json"))
                papers = []

                for meta_file in metadata_files[:100]:
                    try:
                        with open(meta_file) as f:
                            meta = json.load(f)
                        pdf_exists = any(
                            (meta_file.parent / f).exists()
                            for f in meta_file.parent.glob("*.pdf")
                        )
                        papers.append(
                            {
                                "id": meta_file.parent.name,
                                "title": meta.get("title"),
                                "doi": meta.get("doi"),
                                "has_pdf": pdf_exists,
                            }
                        )
                    except Exception:
                        pass

                content = json.dumps(
                    {
                        "project": project_name,
                        "paper_count": len(papers),
                        "papers": papers,
                    },
                    indent=2,
                )

                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text=content,
                )

            elif uri.startswith("scholar://bibtex/"):
                filename = uri.replace("scholar://bibtex/", "")
                bib_files = list(get_scholar_dir().rglob(filename))

                if not bib_files:
                    raise ValueError(f"BibTeX file not found: {filename}")

                with open(bib_files[0]) as f:
                    content = f.read()

                return types.TextResourceContents(
                    uri=uri,
                    mimeType="application/x-bibtex",
                    text=content,
                )

            else:
                raise ValueError(f"Unknown resource URI: {uri}")

    async def _wrap_result(self, coro):
        """Wrap handler result as MCP TextContent."""
        import json

        try:
            result = await coro
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(result, indent=2, default=str),
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({"success": False, "error": str(e)}, indent=2),
                )
            ]


async def _run_server():
    """Run the MCP server (internal)."""
    server = ScholarServer()
    async with stdio_server() as (read_stream, write_stream):
        await server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="scitex-scholar",
                server_version="0.1.0",
                capabilities=server.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


def main():
    """Main entry point for the MCP server."""
    if not MCP_AVAILABLE:
        import sys

        print("=" * 60)
        print("MCP Server 'scitex-scholar' requires the 'mcp' package.")
        print()
        print("Install with:")
        print("  pip install mcp")
        print()
        print("Or install scitex with MCP support:")
        print("  pip install scitex[mcp]")
        print("=" * 60)
        sys.exit(1)

    asyncio.run(_run_server())


if __name__ == "__main__":
    main()


# EOF
