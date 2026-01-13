#!/usr/bin/env python3
"""
SciTeX Browser Commands - Browser automation utilities with session management
"""

import asyncio
import json
import sys
from pathlib import Path

import click

# Browser session storage
BROWSER_STATE_FILE = Path.home() / ".scitex" / "browser" / "sessions.json"


def _load_sessions():
    """Load browser session state."""
    if BROWSER_STATE_FILE.exists():
        return json.loads(BROWSER_STATE_FILE.read_text())
    return {}


def _save_sessions(sessions):
    """Save browser session state."""
    BROWSER_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BROWSER_STATE_FILE.write_text(json.dumps(sessions, indent=2))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def browser():
    """
    Browser automation utilities

    \b
    Launch and control Chrome browsers:
    - Interactive mode: Visible browser for user interaction
    - Stealth mode: Headless browser for automation
    - Switch modes dynamically with show/hide commands

    \b
    Examples:
        scitex browser open                    # Open interactive browser
        scitex browser open --stealth          # Open headless browser
        scitex browser show <id>               # Make browser visible
        scitex browser hide <id>               # Make browser headless
        scitex browser list                    # List active browsers
    """
    pass


@browser.command()
@click.argument("url", default="about:blank", required=False)
@click.option("--stealth", is_flag=True, help="Use stealth/headless mode")
@click.option(
    "--timeout", type=int, default=0, help="Auto-close after N seconds (0=manual close)"
)
@click.option("--background", "-b", is_flag=True, help="Run in background, return ID")
def open(url, stealth, timeout, background):
    """
    Open a Chrome browser

    \b
    Examples:
        scitex browser open                           # Open blank page
        scitex browser open google.com                # Open URL
        scitex browser open https://google.com        # Open URL (with scheme)
        scitex browser open google.com --stealth      # Headless mode
    """
    # Add https:// if no scheme provided
    if url != "about:blank" and not url.startswith(("http://", "https://", "about:")):
        url = f"https://{url}"
    from playwright.async_api import async_playwright

    from scitex.browser.core import BrowserMixin

    class SimpleBrowser(BrowserMixin):
        pass

    mode = "stealth" if stealth else "interactive"

    async def run():
        browser_instance = SimpleBrowser(mode=mode)

        # Use create_browser_context_async for proper headless support
        pw = await async_playwright().start()
        browser_obj, context = await browser_instance.create_browser_context_async(pw)
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")

        # Store references for mode switching
        browser_instance._playwright = pw
        browser_instance._browser = browser_obj
        browser_instance.contexts = [context]
        browser_instance.pages = [page]

        # Generate browser ID
        import uuid

        browser_id = str(uuid.uuid4())[:8]

        # Show browser ID in the page title
        await page.evaluate(f"""
            document.title = '[{browser_id}] ' + document.title;
        """)

        click.echo(f"Browser ID: {browser_id}")
        click.echo(f"Mode: {mode}")
        click.echo(f"URL: {url}")

        if background:
            # Save session info
            sessions = _load_sessions()
            sessions[browser_id] = {
                "mode": mode,
                "url": url,
                "pid": None,  # Would need process management for true background
            }
            _save_sessions(sessions)
            click.echo(f"Use 'scitex browser show {browser_id}' to make visible")
            click.echo(f"Use 'scitex browser hide {browser_id}' to make headless")
            return browser_id

        click.echo("Commands: 's'=show, 'h'=hide, 'q'=quit")

        async def switch_mode(new_mode):
            """Switch browser mode by recreating with proper headless setting."""
            nonlocal browser_obj, context, page, browser_instance

            if browser_instance.mode == new_mode:
                return

            # Get current URL
            current_url = page.url

            # Close old browser
            await browser_instance.close_all_pages()
            await browser_obj.close()

            # Create new browser with correct mode
            browser_instance.mode = new_mode
            browser_obj, context = await browser_instance.create_browser_context_async(
                pw
            )
            page = await context.new_page()
            await page.goto(current_url, wait_until="domcontentloaded")

            # Update references
            browser_instance._browser = browser_obj
            browser_instance.contexts = [context]
            browser_instance.pages = [page]

            # Restore browser ID in title
            await page.evaluate(f"""
                document.title = '[{browser_id}] ' + document.title;
            """)

        if timeout > 0:
            click.echo(f"Auto-closing in {timeout} seconds...")
            await asyncio.sleep(timeout)
        else:
            # Interactive control loop
            import select
            import sys as _sys

            try:
                while True:
                    # Check for keyboard input (non-blocking on Unix)
                    if select.select([_sys.stdin], [], [], 0.5)[0]:
                        cmd = _sys.stdin.readline().strip().lower()
                        if cmd == "s":
                            await switch_mode("interactive")
                            click.echo("Browser: now visible (interactive)")
                        elif cmd == "h":
                            await switch_mode("stealth")
                            click.echo("Browser: now hidden (stealth)")
                        elif cmd == "q":
                            break
                    await asyncio.sleep(0.1)
            except (KeyboardInterrupt, EOFError):
                pass

        await browser_instance.close_all_pages()
        await browser_obj.close()
        await pw.stop()
        click.echo("Browser closed")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        click.echo("\nBrowser closed")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@browser.command()
def list():
    """List browser sessions."""
    sessions = _load_sessions()
    if not sessions:
        click.echo("No browser sessions found")
        return

    click.echo("Browser Sessions:")
    for bid, info in sessions.items():
        click.echo(f"  {bid}: mode={info.get('mode', '?')}, url={info.get('url', '?')}")


@browser.command()
@click.argument("browser_id", required=False)
def show(browser_id):
    """
    Switch browser to visible/interactive mode

    \b
    For running browser, enter 's' in terminal.
    For background browser, use: scitex browser show <id>
    """
    if not browser_id:
        click.echo("For running browser: type 's' + Enter")
        click.echo("For background: scitex browser show <browser_id>")
        return

    click.echo("Note: Background browser control requires process management.")
    click.echo("For running browser, type 's' + Enter in browser terminal.")


@browser.command()
@click.argument("browser_id", required=False)
def hide(browser_id):
    """
    Switch browser to headless/stealth mode

    \b
    For running browser, enter 'h' in terminal.
    For background browser, use: scitex browser hide <id>
    """
    if not browser_id:
        click.echo("For running browser: type 'h' + Enter")
        click.echo("For background: scitex browser hide <browser_id>")
        return

    click.echo("Note: Background browser control requires process management.")
    click.echo("For running browser, type 'h' + Enter in browser terminal.")


# EOF
