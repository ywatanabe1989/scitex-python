#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI interface for Zotero integration.

Usage:
    python -m scitex.scholar.zotero import [options]
    python -m scitex.scholar.zotero export [options]
    python -m scitex.scholar.zotero sync [options]
"""

import argparse
import sys
import os

from scitex import logging
from .importer import ZoteroImporter
from .exporter import ZoteroExporter
from .linker import ZoteroLinker

logger = logging.getLogger(__name__)


def cmd_import(args):
    """Import from Zotero."""
    importer = ZoteroImporter(
        library_id=args.library_id or os.getenv("ZOTERO_LIBRARY_ID"),
        library_type=args.library_type,
        api_key=args.api_key or os.getenv("ZOTERO_API_KEY"),
        project=args.project,
    )

    if args.collection:
        papers = importer.import_collection(
            collection_name=args.collection,
            limit=args.limit,
            include_pdfs=args.include_pdfs,
            include_annotations=args.include_annotations,
        )
    elif args.tags:
        tags = args.tags.split(",")
        papers = importer.import_by_tags(
            tags=tags,
            match_all=args.match_all,
            limit=args.limit,
            include_pdfs=args.include_pdfs,
            include_annotations=args.include_annotations,
        )
    else:
        papers = importer.import_all(
            limit=args.limit,
            include_pdfs=args.include_pdfs,
            include_annotations=args.include_annotations,
        )

    if args.save_to_library:
        importer.import_to_library(papers)

    logger.success(f"Imported {len(papers)} papers")


def cmd_export(args):
    """Export to Zotero."""
    from scitex.scholar.core import Papers

    exporter = ZoteroExporter(
        library_id=args.library_id or os.getenv("ZOTERO_LIBRARY_ID"),
        library_type=args.library_type,
        api_key=args.api_key or os.getenv("ZOTERO_API_KEY"),
        project=args.project,
    )

    papers = Papers.from_project(args.project)

    if args.format == "zotero":
        results = exporter.export_papers(
            papers,
            collection_name=args.collection,
            create_collection=args.create_collection,
            update_existing=args.update_existing,
        )
        logger.success(f"Exported {len(results)} papers to Zotero")

    elif args.format == "bibtex":
        output_path = exporter.export_as_bibtex(papers, args.output)
        logger.success(f"Exported BibTeX: {output_path}")

    elif args.format == "ris":
        output_path = exporter.export_as_ris(papers, args.output)
        logger.success(f"Exported RIS: {output_path}")


def cmd_sync(args):
    """Live synchronization."""
    linker = ZoteroLinker(
        library_id=args.library_id or os.getenv("ZOTERO_LIBRARY_ID"),
        library_type=args.library_type,
        api_key=args.api_key or os.getenv("ZOTERO_API_KEY"),
        project=args.project,
        sync_interval=args.interval,
    )

    if args.once:
        stats = linker.sync_once(
            bidirectional=args.bidirectional,
            auto_import=args.auto_import,
            auto_export=args.auto_export,
        )
        logger.success(f"Sync stats: {stats}")
    else:
        logger.info("Starting continuous sync (Ctrl+C to stop)...")
        linker.start_sync(
            bidirectional=args.bidirectional,
            auto_import=args.auto_import,
            auto_export=args.auto_export,
        )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SciTeX Scholar - Zotero Integration CLI"
    )
    parser.add_argument(
        "--library-id", help="Zotero library ID (or set ZOTERO_LIBRARY_ID env var)"
    )
    parser.add_argument(
        "--library-type",
        default="user",
        choices=["user", "group"],
        help="Library type (default: user)",
    )
    parser.add_argument(
        "--api-key", help="Zotero API key (or set ZOTERO_API_KEY env var)"
    )
    parser.add_argument(
        "--project", default="default", help="Scholar project name (default: default)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import from Zotero")
    import_parser.add_argument("--collection", help="Collection name")
    import_parser.add_argument("--tags", help="Comma-separated tags")
    import_parser.add_argument(
        "--match-all", action="store_true", help="Require all tags (AND logic)"
    )
    import_parser.add_argument("--limit", type=int, help="Maximum items to import")
    import_parser.add_argument(
        "--no-pdfs",
        dest="include_pdfs",
        action="store_false",
        help="Skip PDF attachments",
    )
    import_parser.add_argument(
        "--no-annotations",
        dest="include_annotations",
        action="store_false",
        help="Skip PDF annotations",
    )
    import_parser.add_argument(
        "--save-to-library", action="store_true", help="Save to Scholar library"
    )
    import_parser.set_defaults(include_pdfs=True, include_annotations=True)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export to Zotero")
    export_parser.add_argument(
        "--format",
        default="zotero",
        choices=["zotero", "bibtex", "ris"],
        help="Export format (default: zotero)",
    )
    export_parser.add_argument("--collection", help="Zotero collection name")
    export_parser.add_argument(
        "--create-collection",
        action="store_true",
        help="Create collection if not exists",
    )
    export_parser.add_argument(
        "--update-existing", action="store_true", help="Update existing items"
    )
    export_parser.add_argument("--output", help="Output file path (for bibtex/ris)")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Live synchronization")
    sync_parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Sync interval in seconds (default: 60)",
    )
    sync_parser.add_argument("--once", action="store_true", help="Sync once and exit")
    sync_parser.add_argument(
        "--bidirectional", action="store_true", help="Sync both directions"
    )
    sync_parser.add_argument(
        "--auto-import",
        action="store_true",
        default=True,
        help="Auto-import from Zotero",
    )
    sync_parser.add_argument(
        "--auto-export", action="store_true", help="Auto-export to Zotero"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate credentials
    library_id = args.library_id or os.getenv("ZOTERO_LIBRARY_ID")
    api_key = args.api_key or os.getenv("ZOTERO_API_KEY")

    if not library_id or not api_key:
        logger.error(
            "Zotero credentials required. Set --library-id and --api-key, "
            "or set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY environment variables.\n"
            "Get API key from: https://www.zotero.org/settings/keys"
        )
        sys.exit(1)

    # Execute command
    try:
        if args.command == "import":
            cmd_import(args)
        elif args.command == "export":
            cmd_export(args)
        elif args.command == "sync":
            cmd_sync(args)

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
