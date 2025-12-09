#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX Cloud Commands - Wrapper for tea (Gitea CLI)

Provides git/repository operations by wrapping the tea command.
This gives users familiar git hosting workflows (like gh/tea).
"""

import click
import subprocess
import sys
import os
import socket
import time
import fcntl
from pathlib import Path
from datetime import datetime


def run_tea(*args):
    """
    Execute tea command and return result.

    Args:
        *args: Arguments to pass to tea

    Returns:
        subprocess.CompletedProcess
    """
    tea_path = Path.home() / ".local" / "bin" / "tea"

    if not tea_path.exists():
        click.echo("Error: tea CLI not found", err=True)
        click.echo(
            "Install: wget https://dl.gitea.com/tea/0.9.2/tea-0.9.2-linux-amd64 -O ~/.local/bin/tea && chmod +x ~/.local/bin/tea",
            err=True,
        )
        sys.exit(1)

    try:
        result = subprocess.run(
            [str(tea_path)] + list(args),
            capture_output=False,  # Show output directly
            text=True,
        )
        return result
    except Exception as e:
        click.echo(f"Error running tea: {e}", err=True)
        sys.exit(1)


def is_in_workspace():
    """
    Check if current environment is a SciTeX workspace container.

    Returns:
        bool: True if running in workspace, False otherwise
    """
    # Method A: Environment variable
    if os.environ.get("SCITEX_WORKSPACE"):
        return True

    # Method B: Check hostname pattern
    hostname = socket.gethostname()
    if hostname.startswith("scitex-workspace-"):
        return True

    # Method C: Check special marker file
    if Path("/.scitex-workspace").exists():
        return True

    return False


def ensure_not_in_workspace():
    """
    Ensure user is NOT in workspace container.

    The 'scitex cloud' commands are designed for local machines only.
    Inside workspace, users should use regular git commands or just edit files
    (auto-backups handle the rest).

    Raises:
        SystemExit: If running inside workspace container
    """
    if is_in_workspace():
        click.echo("", err=True)
        click.echo("❌ Error: You are inside a SciTeX workspace!", err=True)
        click.echo("", err=True)
        click.echo("The 'scitex cloud' commands are for your LOCAL machine", err=True)
        click.echo("to interact with your cloud workspace.", err=True)
        click.echo("", err=True)
        click.echo("Inside the workspace, use regular git commands:", err=True)
        click.echo("", err=True)
        click.echo("  git status              # Check current state", err=True)
        click.echo("  git add .", err=True)
        click.echo("  git commit -m 'msg'", err=True)
        click.echo("  git push origin main", err=True)
        click.echo("", err=True)
        click.echo("💡 Or just edit and save files - automatic backups", err=True)
        click.echo("   run every 5 minutes for disaster recovery!", err=True)
        click.echo("", err=True)
        sys.exit(1)


def check_workspace_sync_status():
    """
    Check if workspace has uncommitted or unpushed changes.

    Returns:
        tuple: (needs_sync: bool, status_message: str)
    """
    try:
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        if result.stdout.strip():
            return True, "⚠ Uncommitted changes"

        # Check for unpushed commits
        result = subprocess.run(
            ["git", "rev-list", "--count", "origin/main..HEAD"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        if result.returncode == 0 and int(result.stdout.strip()) > 0:
            return True, "⚠ Unpushed commits"

        return False, "✓ Synced"

    except Exception:
        return False, "⚠ Cannot determine status"


def check_large_files(threshold_mb=100):
    """
    Check for files larger than threshold.

    Args:
        threshold_mb: Size threshold in megabytes

    Returns:
        list: List of (filepath, size_mb) tuples for large files
    """
    large_files = []
    threshold_bytes = threshold_mb * 1024 * 1024

    try:
        # Get all tracked and untracked files
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )

        untracked_files = (
            result.stdout.strip().split("\n") if result.stdout.strip() else []
        )

        # Check sizes
        for filepath in untracked_files:
            full_path = Path(os.getcwd()) / filepath
            if full_path.exists() and full_path.is_file():
                size = full_path.stat().st_size
                if size > threshold_bytes:
                    size_mb = size / (1024 * 1024)
                    large_files.append((filepath, size_mb))

    except Exception as e:
        click.echo(f"⚠ Warning: Could not check file sizes: {e}", err=True)

    return large_files


class SyncLock:
    """File-based lock for preventing concurrent sync operations."""

    def __init__(self, lock_path="/tmp/scitex-workspace-sync.lock", timeout=30):
        self.lock_path = lock_path
        self.timeout = timeout
        self.lock_file = None

    def __enter__(self):
        """Acquire lock."""
        self.lock_file = open(self.lock_path, "w")
        start_time = time.time()

        while True:
            try:
                # Try to acquire exclusive lock
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Write PID for debugging
                self.lock_file.write(str(os.getpid()))
                self.lock_file.flush()
                return self
            except IOError:
                # Lock held by another process
                if time.time() - start_time > self.timeout:
                    raise TimeoutError(
                        "Could not acquire sync lock - another sync in progress"
                    )
                click.echo("⏳ Waiting for ongoing sync to complete...", err=True)
                time.sleep(1)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release lock."""
        if self.lock_file:
            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                self.lock_file.close()
            except Exception:
                pass


# NOTE: ensure_workspace_synced() removed - no longer needed
# New architecture: Users manually sync from local machine only
# Workspace files are backed up via rsync snapshots (not auto-git-sync)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cloud():
    """
    Cloud/Git operations (wraps tea for Gitea)

    \b
    Provides standard git hosting operations:
    - Repository management (create, list, delete)
    - Cloning and forking
    - Pull requests and issues
    - Collaboration workflows

    \b
    Backend: Gitea (git.scitex.ai)
    Similar to: gh (GitHub), tea (Gitea)
    """
    pass


@cloud.command()
@click.option("--url", default="http://localhost:3001", help="Gitea instance URL")
@click.option("--token", help="API token")
def login(url, token):
    """
    Login to SciTeX Cloud (Gitea)

    Example:
        scitex cloud login
        scitex cloud login --url https://git.scitex.ai --token YOUR_TOKEN
    """
    args = ["login", "add", "--name", "scitex", "--url", url]
    if token:
        args.extend(["--token", token])

    run_tea(*args)


@cloud.command()
@click.argument("repository")
@click.argument("destination", required=False)
@click.option("--login", "-l", default="scitex-dev", help="Tea login to use")
def clone(repository, destination, login):
    """
    Clone a repository from SciTeX Cloud

    \b
    Arguments:
        REPOSITORY  Repository name or username/repo format
        DESTINATION Local directory (optional)

    \b
    Examples:
        scitex cloud clone django-gitea-demo
        scitex cloud clone ywatanabe/my-research
        scitex cloud clone my-research ./local-dir
    """
    # If repository doesn't contain '/', try to find it in the repo list
    if "/" not in repository:
        # Get list of repos to find the matching one
        try:
            result = subprocess.run(
                [
                    str(Path.home() / ".local" / "bin" / "tea"),
                    "repos",
                    "ls",
                    "--login",
                    login,
                    "--fields",
                    "name,owner",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the output to find matching repo
            for line in result.stdout.split("\n"):
                if repository in line:
                    # Extract owner from the line
                    parts = line.split("|")
                    if len(parts) >= 2:
                        owner = parts[1].strip()
                        if owner and owner != "OWNER":  # Skip header
                            repository = f"{owner}/{repository}"
                            break

            # If still no '/', it means we didn't find it
            if "/" not in repository:
                click.echo(
                    f"Error: Repository '{repository}' not found. Please use format 'username/repo'",
                    err=True,
                )
                sys.exit(1)

        except subprocess.CalledProcessError:
            click.echo(
                f"Error: Could not list repositories. Please use format 'username/repo'",
                err=True,
            )
            sys.exit(1)

    # Use tea clone command
    args = ["clone", "--login", login, repository]
    if destination:
        args.append(destination)

    run_tea(*args)


@cloud.command()
@click.argument("name")
@click.option("--description", "-d", help="Repository description")
@click.option("--private", is_flag=True, help="Make repository private")
@click.option("--login", "-l", default="scitex-dev", help="Tea login to use")
def create(name, description, private, login):
    """
    Create a new repository

    \b
    Examples:
        scitex cloud create my-new-project
        scitex cloud create my-project --description "My research" --private
    """
    args = ["repo", "create", "--name", name, "--login", login]

    if description:
        args.extend(["--description", description])
    if private:
        args.append("--private")

    run_tea(*args)


@cloud.command(name="list")
@click.option("--user", "-u", help="List repos for specific user")
@click.option("--login", "-l", default="scitex-dev", help="Tea login to use")
@click.option("--starred", "-s", is_flag=True, help="List starred repos")
@click.option("--watched", "-w", is_flag=True, help="List watched repos")
def list_repos(user, login, starred, watched):
    """
    List repositories

    \b
    Examples:
        scitex cloud list
        scitex cloud list --user ywatanabe
        scitex cloud list --starred
        scitex cloud list --watched
    """
    args = ["repos", "--login", login, "--output", "table"]

    if starred:
        args.append("--starred")
    if watched:
        args.append("--watched")
    if user:
        args.append(user)

    run_tea(*args)


@cloud.command()
@click.argument("query")
@click.option("--login", "-l", default="scitex-dev", help="Tea login to use")
@click.option("--limit", type=int, default=10, help="Maximum results to show")
def search(query, login, limit):
    """
    Search for repositories

    \b
    Arguments:
        QUERY  Search query string

    \b
    Examples:
        scitex cloud search neural
        scitex cloud search "machine learning" --limit 20
    """
    args = ["repos", "search", "--login", login, "--limit", str(limit), query]
    run_tea(*args)


@cloud.command()
@click.argument("repository")
@click.option("--login", "-l", default="scitex-dev", help="Tea login to use")
@click.confirmation_option(prompt="Are you sure you want to delete this repository?")
def delete(repository, login):
    """
    Delete a repository (DANGEROUS!)

    \b
    Arguments:
        REPOSITORY  Repository in format: username/repo

    \b
    Examples:
        scitex cloud delete ywatanabe/test-repo

    \b
    WARNING: This action cannot be undone!
    """
    # tea doesn't have a delete command, use API directly
    import requests
    import json

    # Read tea config to get token and URL
    config_path = Path.home() / ".config" / "tea" / "config.yml"
    if not config_path.exists():
        click.echo("Error: Tea configuration not found", err=True)
        sys.exit(1)

    try:
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Find the login
        login_config = None
        for l in config.get("logins", []):
            if l["name"] == login:
                login_config = l
                break

        if not login_config:
            click.echo(f"Error: Login '{login}' not found", err=True)
            sys.exit(1)

        url = login_config["url"]
        token = login_config["token"]

        # Parse repository
        if "/" not in repository:
            click.echo("Error: Repository must be in format 'username/repo'", err=True)
            sys.exit(1)

        owner, repo = repository.split("/", 1)

        # Delete via API
        api_url = f"{url}/api/v1/repos/{owner}/{repo}"
        headers = {"Authorization": f"token {token}"}

        response = requests.delete(api_url, headers=headers)

        if response.status_code == 204:
            click.echo(f"✓ Repository '{repository}' deleted successfully")
        elif response.status_code == 404:
            click.echo(f"Error: Repository '{repository}' not found", err=True)
            sys.exit(1)
        else:
            click.echo(
                f"Error: Failed to delete repository (status {response.status_code})",
                err=True,
            )
            sys.exit(1)

    except ImportError:
        click.echo("Error: PyYAML not installed. Run: pip install pyyaml", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cloud.command()
@click.argument("repository")
def fork(repository):
    """
    Fork a repository

    \b
    Arguments:
        REPOSITORY  Repository to fork (username/repo)

    \b
    Example:
        scitex cloud fork lab-pi/shared-project
    """
    run_tea("repo", "fork", repository)


@cloud.group()
def pr():
    """Pull request operations"""
    pass


@pr.command(name="create")
@click.option("--title", "-t", help="PR title")
@click.option("--description", "-d", help="PR description")
@click.option("--base", "-b", default="main", help="Base branch")
@click.option("--head", "-h", help="Head branch")
def pr_create(title, description, base, head):
    """
    Create a pull request

    \b
    Example:
        scitex cloud pr create --title "Add analysis" --base main --head feature
    """
    args = ["pr", "create"]

    if title:
        args.extend(["--title", title])
    if description:
        args.extend(["--description", description])
    if base:
        args.extend(["--base", base])
    if head:
        args.extend(["--head", head])

    run_tea(*args)


@pr.command(name="list")
def pr_list():
    """List pull requests"""
    run_tea("pr", "list")


@cloud.group()
def issue():
    """Issue operations"""
    pass


@issue.command(name="create")
@click.option("--title", "-t", required=True, help="Issue title")
@click.option("--body", "-b", help="Issue body")
def issue_create(title, body):
    """
    Create an issue

    \b
    Example:
        scitex cloud issue create --title "Bug in analysis" --body "Details here"
    """
    args = ["issue", "create", "--title", title]
    if body:
        args.extend(["--body", body])

    run_tea(*args)


@issue.command(name="list")
def issue_list():
    """List issues"""
    run_tea("issue", "list")


@cloud.command()
def push():
    """
    Push local changes to workspace

    Pushes your local Git commits to the SciTeX Cloud workspace.
    Use this after committing changes locally that you want to sync to the cloud.
    """
    # Ensure this is run from LOCAL machine only
    ensure_not_in_workspace()

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()

        if not branch:
            click.echo("Error: Not on any branch", err=True)
            sys.exit(1)

        # Push to workspace
        click.echo("📤 Pushing to workspace...")
        subprocess.run(["git", "push", "origin", branch], check=True)
        click.echo(f"✓ Pushed to workspace (origin/{branch})")

    except subprocess.CalledProcessError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cloud.command()
def pull():
    """
    Pull workspace changes to local machine

    Pulls changes from your SciTeX Cloud workspace to your local machine.
    Use this to sync the latest changes from the cloud workspace.
    """
    # Ensure this is run from LOCAL machine only
    ensure_not_in_workspace()

    try:
        # Get current branch
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()

        if not branch:
            click.echo("Error: Not on any branch", err=True)
            sys.exit(1)

        # Pull from workspace
        click.echo("📥 Pulling from workspace...")
        subprocess.run(["git", "pull", "origin", branch], check=True)
        click.echo(f"✓ Pulled from workspace (origin/{branch})")

    except subprocess.CalledProcessError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cloud.command()
def status():
    """
    Show repository status

    Shows the current Git status of your local repository.
    """
    # Ensure this is run from LOCAL machine only
    ensure_not_in_workspace()

    # Show regular git status
    subprocess.run(["git", "status"])


@cloud.command()
@click.option(
    "-i",
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input BibTeX file",
)
@click.option(
    "-o",
    "--output",
    "output_file",
    required=True,
    type=click.Path(),
    help="Output BibTeX file",
)
@click.option(
    "-a",
    "--api-key",
    envvar="SCITEX_API_KEY",
    help="SciTeX API key (or set SCITEX_API_KEY env var)",
)
@click.option("--no-cache", is_flag=True, help="Disable cache (force fresh metadata)")
@click.option("--url", default="https://scitex.cloud", help="SciTeX Cloud URL")
def enrich(input_file, output_file, api_key, no_cache, url):
    """
    Enrich BibTeX file with metadata

    Usage:
        scitex cloud enrich -i refs.bib -o enriched.bib -a $SCITEX_API_KEY
    """
    import requests
    import time

    if not api_key:
        click.echo("Error: API key required", err=True)
        click.echo("Set SCITEX_API_KEY env var or use --api-key", err=True)
        click.echo("Get key at: https://scitex.cloud/api-keys/", err=True)
        sys.exit(1)

    click.echo(f"Enriching: {input_file}")

    # Upload
    with open(input_file, "rb") as f:
        files = {"bibtex_file": f}
        data = {"use_cache": "false" if no_cache else "true"}
        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-Requested-With": "XMLHttpRequest",
        }

        response = requests.post(
            f"{url}/scholar/bibtex/upload/", headers=headers, files=files, data=data
        )

    if response.status_code != 200:
        click.echo(f"Error: Upload failed ({response.status_code})", err=True)
        sys.exit(1)

    result = response.json()
    if not result.get("success"):
        click.echo(f"Error: {result.get('error', 'Upload failed')}", err=True)
        sys.exit(1)

    job_id = result["job_id"]
    click.echo(f"Job ID: {job_id}")
    click.echo("Processing", nl=False)

    # Poll status
    while True:
        response = requests.get(
            f"{url}/scholar/api/bibtex/job/{job_id}/status/", headers=headers
        )
        data = response.json()
        status = data["status"]

        if status == "completed":
            click.echo(" Done!")
            break
        elif status in ("failed", "cancelled"):
            click.echo(f" {status.capitalize()}!", err=True)
            sys.exit(1)

        click.echo(".", nl=False)
        time.sleep(2)

    # Download
    response = requests.get(
        f"{url}/scholar/api/bibtex/job/{job_id}/download/", headers=headers
    )

    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        click.echo(f"✓ Saved: {output_file}")
    else:
        click.echo("Error: Download failed", err=True)
        sys.exit(1)


# EOF
