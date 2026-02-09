#!/usr/bin/env python3
# Timestamp: "2026-02-09 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/cli/_verify_claims.py
"""CLI commands for claim layer, external timestamping, and backward-compat aliases."""

import sys

import click


def register_claim_commands(verify_group):  # noqa: C901
    """Register claim-related commands on the verify CLI group."""

    @verify_group.command("add-claim")
    @click.option(
        "--file", "-f", "file_path", required=True, help="Manuscript file path"
    )
    @click.option(
        "--line", "-l", "line_number", type=int, help="Line number in manuscript"
    )
    @click.option(
        "--type",
        "-t",
        "claim_type",
        type=click.Choice(["statistic", "figure", "table", "text", "value"]),
        required=True,
        help="Claim type",
    )
    @click.option(
        "--value", "-V", "claim_value", help="Asserted value (e.g., 'p = 0.003')"
    )
    @click.option(
        "--source", "-s", "source_file", help="Source file that produced this claim"
    )
    @click.option(
        "--session", "source_session", help="Session ID that produced the source"
    )
    def add_claim_cmd(
        file_path, line_number, claim_type, claim_value, source_file, source_session
    ):
        """
        Register a claim linking a manuscript assertion to the verification chain.

        \\b
        Examples:
          scitex verify add-claim -f paper.tex -l 42 -t statistic -V "p = 0.003" -s results.csv
          scitex verify add-claim -f paper.tex -l 58 -t figure -s figure1.yaml
          scitex verify add-claim -f paper.tex -l 10 -t text -V "N = 150" -s summary.csv
        """
        try:
            from scitex.verify import add_claim

            claim = add_claim(
                file_path=file_path,
                claim_type=claim_type,
                line_number=line_number,
                claim_value=claim_value,
                source_file=source_file,
                source_session=source_session,
            )
            click.secho(f"Registered: {claim.claim_id}", fg="green")
            click.echo(f"  Location: {claim.location}")
            if claim.claim_value:
                click.echo(f"  Value:    {claim.claim_value}")
            if claim.source_file:
                click.echo(f"  Source:   {claim.source_file}")
            if claim.source_hash:
                click.echo(f"  Hash:     {claim.source_hash[:16]}...")

        except Exception as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)

    @verify_group.command("list-claims")
    @click.option("--file", "-f", "file_path", help="Filter by manuscript file")
    @click.option(
        "--type",
        "-t",
        "claim_type",
        type=click.Choice(["statistic", "figure", "table", "text", "value"]),
        help="Filter by claim type",
    )
    @click.option("--status", "-s", help="Filter by status")
    @click.option("-v", "--verbose", is_flag=True, help="Show source details")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    def list_claims_cmd(file_path, claim_type, status, verbose, as_json):
        """
        List registered claims.

        \\b
        Examples:
          scitex verify list-claims
          scitex verify list-claims -f paper.tex
          scitex verify list-claims -t statistic -v
        """
        try:
            from scitex.verify import format_claims, list_claims

            claims = list_claims(
                file_path=file_path,
                claim_type=claim_type,
                status=status,
            )

            if as_json:
                import json

                click.echo(json.dumps([c.to_dict() for c in claims], indent=2))
            else:
                if not claims:
                    click.echo("No claims registered.")
                    click.echo("\nTo register claims, use: scitex verify add-claim")
                    return
                output = format_claims(claims, verbose=verbose)
                click.echo(output)
                click.echo(f"\n{len(claims)} claim(s) registered")

        except Exception as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)

    @verify_group.command("verify-claim")
    @click.argument("target", required=True)
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    def verify_claim_cmd(target, as_json):
        """
        Verify a specific claim against its source chain.

        TARGET can be a claim_id or a location like "paper.tex:L42".

        \\b
        Examples:
          scitex verify verify-claim claim_a1b2c3d4e5f6
          scitex verify verify-claim paper.tex:L42
        """
        try:
            from scitex.verify import verify_claim

            result = verify_claim(target)

            if as_json:
                import json

                click.echo(json.dumps(result, indent=2))
            else:
                claim = result.get("claim", {})
                status = result.get("claim", {}).get("status", "unknown")

                if status == "verified":
                    click.secho(
                        f"\u2713 Claim verified: {claim.get('claim_id', target)}",
                        fg="green",
                    )
                elif status == "not_found":
                    click.secho(f"? Claim not found: {target}", fg="yellow")
                elif status == "mismatch":
                    click.secho(
                        f"\u2717 Claim mismatch: {claim.get('claim_id', target)}",
                        fg="red",
                    )
                elif status == "missing":
                    click.secho(
                        f"? Source missing: {claim.get('claim_id', target)}",
                        fg="yellow",
                    )
                else:
                    click.secho(f"~ Claim status: {status}", fg="yellow")

                for detail in result.get("details", []):
                    click.echo(f"  {detail}")

                if status in ("mismatch", "missing"):
                    sys.exit(1)

        except Exception as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)

    # ── External timestamping commands ──

    @verify_group.command("stamp")
    @click.option(
        "--backend",
        "-b",
        type=click.Choice(["file", "rfc3161", "zenodo"]),
        default="file",
        help="Timestamping backend",
    )
    @click.option("--service-url", help="TSA or API URL")
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    def stamp_cmd(backend, service_url, as_json):
        """
        Record root hash with external timestamp for temporal proof.

        \\b
        Examples:
          scitex verify stamp                       # Local file stamp
          scitex verify stamp -b rfc3161            # RFC 3161 TSA
          scitex verify stamp --json                # JSON output
        """
        try:
            from scitex.verify import stamp as do_stamp

            result = do_stamp(backend=backend, service_url=service_url)

            if as_json:
                import json

                click.echo(json.dumps(result.to_dict(), indent=2))
            else:
                click.secho(f"Stamped: {result.stamp_id}", fg="green")
                click.echo(f"  Root hash:  {result.root_hash[:32]}...")
                click.echo(f"  Timestamp:  {result.timestamp}")
                click.echo(f"  Backend:    {result.backend}")
                click.echo(f"  Runs:       {result.run_count}")
                if result.service_url:
                    click.echo(f"  Proof:      {result.service_url}")

        except Exception as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)

    @verify_group.command("check-stamp")
    @click.argument("stamp_id", required=False)
    @click.option("--json", "as_json", is_flag=True, help="Output as JSON")
    def check_stamp_cmd(stamp_id, as_json):
        """
        Verify a stamp against current verification state.

        If no STAMP_ID given, checks the most recent stamp.

        \\b
        Examples:
          scitex verify check-stamp                 # Check latest
          scitex verify check-stamp stamp_abc123    # Check specific
        """
        try:
            from scitex.verify import check_stamp

            result = check_stamp(stamp_id=stamp_id)

            if as_json:
                import json

                click.echo(json.dumps(result, indent=2))
            else:
                if result.get("status") == "not_found":
                    click.secho("No stamps found.", fg="yellow")
                    return

                matches = result.get("matches", False)
                stamp_info = result.get("stamp", {})

                if matches:
                    click.secho(
                        f"\u2713 Stamp verified: {stamp_info.get('stamp_id')}",
                        fg="green",
                    )
                else:
                    click.secho(
                        f"\u2717 Stamp mismatch: {stamp_info.get('stamp_id')}", fg="red"
                    )

                for detail in result.get("details", []):
                    click.echo(f"  {detail}")

                if not matches:
                    sys.exit(1)

        except Exception as e:
            click.secho(f"Error: {e}", fg="red", err=True)
            sys.exit(1)

    # Backward compat: hidden 'bpv' alias redirects to 'vbp'
    @verify_group.command("bpv", hidden=True)
    @click.argument("target_file", type=click.Path(exists=True))
    @click.option("-v", "--verbose", is_flag=True)
    @click.option("--mermaid", is_flag=True)
    @click.option("--json", "as_json", is_flag=True)
    @click.pass_context
    def bpv_compat_cmd(ctx, target_file, verbose, mermaid, as_json):
        """Deprecated: use 'scitex verify vbp' instead."""
        from scitex.cli.verify import verify_vbp_cmd

        ctx.invoke(
            verify_vbp_cmd,
            target_file=target_file,
            verbose=verbose,
            mermaid=mermaid,
            as_json=as_json,
        )
