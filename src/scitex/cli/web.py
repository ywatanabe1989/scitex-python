#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX CLI - Web Scraping Commands
"""

import os
import sys
import click
from pathlib import Path

from scitex.web import get_urls, download_images, get_image_urls
from scitex.web._scraping import _get_default_download_dir
from scitex.logging import getLogger

logger = getLogger(__name__)


@click.group()
def web():
    """
    Web scraping utilities

    \b
    Commands:
      get-urls        Extract URLs from a webpage
      get-image-urls  Extract image URLs from a webpage
      download-images Download images from a webpage

    \b
    Examples:
      scitex web get-urls https://example.com
      scitex web get-urls https://example.com --pattern "\.pdf$"
      scitex web get-image-urls https://example.com
      scitex web download-images https://example.com --output ./downloads
      scitex web download-images https://example.com --min-size 100x100
    """
    pass


@web.command()
@click.argument('url')
@click.option(
    '--pattern', '-p',
    help='Regex pattern to filter URLs (e.g., "\\.pdf$" for PDF files)'
)
@click.option(
    '--same-domain',
    is_flag=True,
    help='Only include URLs from the same domain'
)
@click.option(
    '--relative',
    is_flag=True,
    help='Keep URLs as relative instead of converting to absolute'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Save URLs to file instead of printing to stdout'
)
def get_urls_cmd(url, pattern, same_domain, relative, output):
    """Extract all URLs from a webpage."""
    try:
        click.echo(f"Extracting URLs from: {url}")

        urls = get_urls(
            url,
            pattern=pattern,
            absolute=not relative,
            same_domain=same_domain
        )

        if not urls:
            click.secho("No URLs found", fg='yellow')
            sys.exit(0)

        click.secho(f"Found {len(urls)} URLs", fg='green')

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(urls))
            click.echo(f"URLs saved to: {output_path}")
        else:
            click.echo()
            for url_item in urls:
                click.echo(url_item)

        sys.exit(0)

    except Exception as e:
        click.secho(f"ERROR: {e}", fg='red', err=True)
        sys.exit(1)


@web.command()
@click.argument('url')
@click.option(
    '--pattern', '-p',
    help='Regex pattern to filter image URLs (e.g., "\\.jpg$")'
)
@click.option(
    '--same-domain',
    is_flag=True,
    help='Only include images from the same domain'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Save image URLs to file instead of printing to stdout'
)
def get_image_urls_cmd(url, pattern, same_domain, output):
    """Extract image URLs from a webpage without downloading them."""
    try:
        click.echo(f"Extracting image URLs from: {url}")

        img_urls = get_image_urls(
            url,
            pattern=pattern,
            same_domain=same_domain
        )

        if not img_urls:
            click.secho("No image URLs found", fg='yellow')
            sys.exit(0)

        click.secho(f"Found {len(img_urls)} image URLs", fg='green')

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(img_urls))
            click.echo(f"Image URLs saved to: {output_path}")
        else:
            click.echo()
            for img_url in img_urls:
                click.echo(img_url)

        sys.exit(0)

    except Exception as e:
        click.secho(f"ERROR: {e}", fg='red', err=True)
        sys.exit(1)


@web.command()
@click.argument('url')
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Output directory (default: $SCITEX_WEB_DOWNLOADS_DIR or $SCITEX_DIR/web/downloads)'
)
@click.option(
    '--pattern', '-p',
    help='Regex pattern to filter image URLs (e.g., "\\.jpg$")'
)
@click.option(
    '--min-size',
    default='100x100',
    help='Minimum image size as WIDTHxHEIGHT (default: 100x100)'
)
@click.option(
    '--same-domain',
    is_flag=True,
    help='Only download images from the same domain'
)
@click.option(
    '--max-workers',
    type=int,
    default=5,
    help='Number of concurrent download threads (default: 5)'
)
def download_images_cmd(url, output, pattern, min_size, same_domain, max_workers):
    """
    Download all images from a webpage.

    \b
    Output directory priority:
      1. --output option if specified
      2. SCITEX_WEB_DOWNLOADS_DIR environment variable
      3. $SCITEX_DIR/web/downloads (default)

    \b
    Note:
      - Images are saved in a timestamped subdirectory: images-YYYYMMDD_HHMMSS/
      - Only images >= 100x100 pixels are downloaded by default
      - SVG files are automatically skipped (vector graphics)
    """
    try:
        click.echo(f"Downloading images from: {url}")

        # Parse min_size if provided
        min_size_tuple = None
        if min_size:
            try:
                width, height = map(int, min_size.split('x'))
                min_size_tuple = (width, height)
            except ValueError:
                click.secho(
                    f"ERROR: Invalid min-size format. Use WIDTHxHEIGHT (e.g., '100x100')",
                    fg='red',
                    err=True
                )
                sys.exit(1)

        paths = download_images(
            url,
            output_dir=output,
            pattern=pattern,
            min_size=min_size_tuple,
            max_workers=max_workers,
            same_domain=same_domain
        )

        if not paths:
            click.secho("No images downloaded", fg='yellow')
            sys.exit(0)

        # Show where images were saved (get actual directory from first path)
        if paths:
            actual_output = str(Path(paths[0]).parent)
        else:
            actual_output = output or _get_default_download_dir()

        click.secho(f"Successfully downloaded {len(paths)} images", fg='green')
        click.echo(f"Images saved to: {actual_output}")

        sys.exit(0)

    except Exception as e:
        click.secho(f"ERROR: {e}", fg='red', err=True)
        sys.exit(1)


# Register command aliases
web.add_command(get_urls_cmd, name='get-urls')
web.add_command(get_image_urls_cmd, name='get-image-urls')
web.add_command(download_images_cmd, name='download-images')


if __name__ == '__main__':
    web()
