#!/usr/bin/env python3
"""
SciTeX CLI - Audio Commands (Text-to-Speech)

Provides text-to-speech with multiple backend support.
"""

import sys

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def audio():
    """
    Text-to-speech utilities

    \b
    Backends (fallback order):
      pyttsx3    - System TTS (offline, free)
      gtts       - Google TTS (free, needs internet)
      elevenlabs - ElevenLabs (paid, high quality)

    \b
    Examples:
      scitex audio speak "Hello world"
      scitex audio speak "Bonjour" --backend gtts --voice fr
      scitex audio backends              # List available backends
      scitex audio check                 # Check audio status (WSL)
    """
    pass


@audio.command()
@click.argument("text")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["pyttsx3", "gtts", "elevenlabs"]),
    help="TTS backend (auto-selects with fallback if not specified)",
)
@click.option("--voice", "-v", help="Voice name, ID, or language code")
@click.option("--output", "-o", type=click.Path(), help="Save audio to file")
@click.option("--no-play", is_flag=True, help="Don't play audio (only save)")
@click.option("--rate", "-r", type=int, help="Speech rate (pyttsx3 only, default: 150)")
@click.option(
    "--speed", "-s", type=float, help="Speed multiplier (gtts only, e.g., 1.5)"
)
@click.option("--no-fallback", is_flag=True, help="Disable backend fallback on error")
def speak(text, backend, voice, output, no_play, rate, speed, no_fallback):
    """
    Convert text to speech

    \b
    Examples:
      scitex audio speak "Hello world"
      scitex audio speak "Bonjour" --backend gtts --voice fr
      scitex audio speak "Test" --output speech.mp3 --no-play
      scitex audio speak "Fast speech" --backend pyttsx3 --rate 200
      scitex audio speak "Slow speech" --backend gtts --speed 0.8
    """
    try:
        from scitex.audio import speak as tts_speak

        kwargs = {
            "text": text,
            "play": not no_play,
            "fallback": not no_fallback,
        }

        if backend:
            kwargs["backend"] = backend
        if voice:
            kwargs["voice"] = voice
        if output:
            kwargs["output_path"] = output
        if rate:
            kwargs["rate"] = rate
        if speed:
            kwargs["speed"] = speed

        result = tts_speak(**kwargs)

        if output and result:
            click.secho(f"Audio saved: {result}", fg="green")
        elif not no_play:
            click.secho("Speech completed", fg="green")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@audio.command(name="backends")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def list_backends(as_json):
    """
    List available TTS backends

    \b
    Example:
      scitex audio backends
      scitex audio backends --json
    """
    try:
        from scitex.audio import FALLBACK_ORDER, available_backends

        backends = available_backends()

        if as_json:
            import json

            output = {
                "available": backends,
                "fallback_order": FALLBACK_ORDER,
            }
            click.echo(json.dumps(output, indent=2))
        else:
            click.secho("Available TTS Backends", fg="cyan", bold=True)
            click.echo("=" * 40)

            click.echo("\nFallback order:")
            for i, b in enumerate(FALLBACK_ORDER, 1):
                status = (
                    click.style("available", fg="green")
                    if b in backends
                    else click.style("not installed", fg="red")
                )
                click.echo(f"  {i}. {b}: {status}")

            if not backends:
                click.echo()
                click.secho("No backends available!", fg="red")
                click.echo("Install one of:")
                click.echo("  pip install pyttsx3  # + apt install espeak-ng")
                click.echo("  pip install gTTS")
                click.echo("  pip install elevenlabs")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@audio.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def check(as_json):
    """
    Check audio status (especially for WSL)

    \b
    Checks:
      - WSL detection
      - WSLg availability
      - PulseAudio connection
      - Windows fallback availability

    \b
    Example:
      scitex audio check
      scitex audio check --json
    """
    try:
        from scitex.audio import check_wsl_audio

        status = check_wsl_audio()

        if as_json:
            import json

            click.echo(json.dumps(status, indent=2))
        else:
            click.secho("Audio Status Check", fg="cyan", bold=True)
            click.echo("=" * 40)

            def status_mark(val):
                return (
                    click.style("Yes", fg="green")
                    if val
                    else click.style("No", fg="red")
                )

            click.echo(f"\nWSL Environment: {status_mark(status['is_wsl'])}")

            if status["is_wsl"]:
                click.echo(f"WSLg Available: {status_mark(status['wslg_available'])}")
                click.echo(
                    f"PulseServer Socket: {status_mark(status['pulse_server_exists'])}"
                )
                click.echo(
                    f"PulseAudio Connected: {status_mark(status['pulse_connected'])}"
                )
                click.echo(
                    f"Windows Fallback: {status_mark(status['windows_fallback_available'])}"
                )

            click.echo()
            rec = status["recommended"]
            if rec == "linux":
                click.secho("Recommended: Linux audio (PulseAudio)", fg="green")
            elif rec == "windows":
                click.secho(
                    "Recommended: Windows fallback (powershell.exe)", fg="yellow"
                )
            else:
                click.secho("No audio output available", fg="red")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@audio.command()
def stop():
    """
    Stop any currently playing speech

    \b
    Example:
      scitex audio stop
    """
    try:
        from scitex.audio import stop_speech

        stop_speech()
        click.secho("Speech stopped", fg="green")

    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    audio()
