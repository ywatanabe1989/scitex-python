#!/usr/bin/env python3
# Timestamp: 2026-01-24
# File: src/scitex/config/_env_registry.py

"""
Registry of all SCITEX environment variables.

Provides documentation and template generation for environment configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class EnvVar:
    """Environment variable definition."""

    name: str
    description: str
    module: str
    default: Optional[str] = None
    required: bool = False
    sensitive: bool = False


# Registry of all SCITEX environment variables
ENV_REGISTRY: List[EnvVar] = [
    # Core
    EnvVar("SCITEX_DIR", "Base directory for scitex data", "core", "~/.scitex"),
    EnvVar("SCITEX_ENV_SRC", "Path to env source file/directory", "core"),
    EnvVar("SCITEX_LOGGING_LEVEL", "Logging level", "logging", "INFO"),
    EnvVar("SCITEX_LOGGING_FORMAT", "Log format style", "logging", "default"),
    EnvVar("SCITEX_LOGGING_FORCE_COLOR", "Force colored output", "logging", "false"),
    # Audio
    EnvVar("SCITEX_AUDIO_MODE", "Audio mode: local/remote/auto", "audio", "auto"),
    EnvVar("SCITEX_AUDIO_RELAY_URL", "Relay server URL for remote audio", "audio"),
    EnvVar("SCITEX_AUDIO_RELAY_HOST", "Relay server host", "audio"),
    EnvVar("SCITEX_AUDIO_RELAY_PORT", "Relay server port", "audio", "31293"),
    EnvVar("SCITEX_AUDIO_PORT", "Audio server port", "audio", "31293"),
    EnvVar(
        "SCITEX_AUDIO_ELEVENLABS_API_KEY", "ElevenLabs API key", "audio", sensitive=True
    ),
    # Scholar
    EnvVar("SCITEX_SCHOLAR_DIR", "Scholar library directory", "scholar"),
    EnvVar("SCITEX_SCHOLAR_CROSSREF_EMAIL", "Email for Crossref API", "scholar"),
    EnvVar("SCITEX_SCHOLAR_PUBMED_EMAIL", "Email for PubMed API", "scholar"),
    EnvVar("SCITEX_SCHOLAR_CROSSREF_DB", "Local Crossref database path", "scholar"),
    EnvVar("SCITEX_SCHOLAR_CROSSREF_API_URL", "Crossref API URL", "scholar"),
    EnvVar(
        "SCITEX_SCHOLAR_CROSSREF_MODE", "Crossref mode: local/api/hybrid", "scholar"
    ),
    EnvVar("SCITEX_SCHOLAR_OPENATHENS_EMAIL", "OpenAthens login email", "scholar"),
    EnvVar(
        "SCITEX_SCHOLAR_OPENATHENS_ENABLED",
        "Enable OpenAthens auth",
        "scholar",
        "false",
    ),
    EnvVar(
        "SCITEX_SCHOLAR_EZPROXY_URL", "EZProxy URL for institutional access", "scholar"
    ),
    EnvVar("SCITEX_SCHOLAR_OPENURL_RESOLVER_URL", "OpenURL resolver URL", "scholar"),
    EnvVar(
        "SCITEX_SCHOLAR_ZENROWS_API_KEY",
        "ZenRows API key for scraping",
        "scholar",
        sensitive=True,
    ),
    # Social
    EnvVar(
        "SCITEX_SOCIAL_X_CONSUMER_KEY",
        "Twitter/X consumer key",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_X_CONSUMER_KEY_SECRET",
        "Twitter/X consumer secret",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_X_ACCESS_TOKEN",
        "Twitter/X access token",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_X_ACCESS_TOKEN_SECRET",
        "Twitter/X access token secret",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_X_BEARER_TOKEN",
        "Twitter/X bearer token",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_LINKEDIN_CLIENT_ID",
        "LinkedIn client ID",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_LINKEDIN_CLIENT_SECRET",
        "LinkedIn client secret",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_LINKEDIN_ACCESS_TOKEN",
        "LinkedIn access token",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_REDDIT_CLIENT_ID",
        "Reddit client ID",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_REDDIT_CLIENT_SECRET",
        "Reddit client secret",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_YOUTUBE_API_KEY",
        "YouTube API key",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_YOUTUBE_CLIENT_SECRETS_FILE",
        "YouTube client secrets file path",
        "social",
    ),
    EnvVar(
        "SCITEX_SOCIAL_GOOGLE_ANALYTICS_PROPERTY_ID",
        "Google Analytics property ID",
        "social",
    ),
    EnvVar(
        "SCITEX_SOCIAL_GOOGLE_ANALYTICS_MEASUREMENT_ID",
        "Google Analytics measurement ID",
        "social",
    ),
    EnvVar(
        "SCITEX_SOCIAL_GOOGLE_ANALYTICS_API_SECRET",
        "Google Analytics API secret",
        "social",
        sensitive=True,
    ),
    EnvVar(
        "SCITEX_SOCIAL_GOOGLE_APPLICATION_CREDENTIALS",
        "Google service account credentials path",
        "social",
    ),
    # Cloud
    EnvVar("SCITEX_CLOUD_USERNAME", "Cloud username", "cloud", sensitive=True),
    EnvVar("SCITEX_CLOUD_PASSWORD", "Cloud password", "cloud", sensitive=True),
    EnvVar(
        "SCITEX_CLOUD_CODE_WORKSPACE", "Running in cloud workspace", "cloud", "false"
    ),
    EnvVar("SCITEX_CLOUD_CODE_PROJECT_ROOT", "Cloud project root", "cloud"),
    EnvVar("SCITEX_CLOUD_CODE_BACKEND", "Cloud backend type", "cloud"),
    # UI/Notification
    EnvVar("SCITEX_UI_DEFAULT_BACKEND", "Default notification backend", "ui"),
    EnvVar(
        "SCITEX_UI_BACKEND_PRIORITY",
        "Notification backend priority (comma-sep)",
        "ui",
    ),
    EnvVar("SCITEX_UI_INFO_BACKENDS", "Backends for info notifications", "ui"),
    EnvVar("SCITEX_UI_WARNING_BACKENDS", "Backends for warning notifications", "ui"),
    EnvVar("SCITEX_UI_ERROR_BACKENDS", "Backends for error notifications", "ui"),
    EnvVar("SCITEX_UI_CRITICAL_BACKENDS", "Backends for critical notifications", "ui"),
    EnvVar("SCITEX_UI_EMAIL_NOTIFICATION_FROM", "Email notification sender", "ui"),
    EnvVar("SCITEX_UI_EMAIL_NOTIFICATION_TO", "Email notification recipient", "ui"),
    EnvVar("SCITEX_UI_WEBHOOK_URL", "Webhook URL for notifications", "ui"),
    # Capture
    EnvVar("SCITEX_CAPTURE_DIR", "Screenshot capture directory", "capture"),
    # Web
    EnvVar("SCITEX_WEB_DOWNLOADS_DIR", "Web downloads directory", "web"),
    # PLT
    EnvVar("SCITEX_PLT_AXES_WIDTH_MM", "Default axes width in mm", "plt"),
    EnvVar("SCITEX_PLT_LINES_TRACE_MM", "Default line trace width in mm", "plt"),
    EnvVar("SCITEX_PLT_STYLE", "Default plot style", "plt"),
    EnvVar("SCITEX_PLT_COLORS", "Color palette to use", "plt"),
]


def get_env_by_module(module: str) -> List[EnvVar]:
    """Get all environment variables for a module."""
    return [e for e in ENV_REGISTRY if e.module == module]


def get_all_modules() -> List[str]:
    """Get list of all modules with env vars."""
    return sorted(set(e.module for e in ENV_REGISTRY))


def generate_template(
    include_sensitive: bool = True, include_defaults: bool = True
) -> str:
    """Generate a template .src file with all environment variables."""
    lines = [
        "#!/bin/bash",
        "# SciTeX Environment Variables Template",
        "# Generated by scitex.config.generate_template()",
        "#",
        "# Usage: source this file or set SCITEX_ENV_SRC to this path",
        "",
    ]

    for module in get_all_modules():
        lines.append(f"# === {module.upper()} ===")
        for env in get_env_by_module(module):
            if env.sensitive and not include_sensitive:
                continue

            if env.description:
                lines.append(f"# {env.description}")

            if env.default and include_defaults:
                lines.append(f'export {env.name}="{env.default}"')
            elif env.sensitive:
                lines.append(f'# export {env.name}="YOUR_{env.name}_HERE"')
            else:
                lines.append(f"# export {env.name}=")
        lines.append("")

    return "\n".join(lines)


def get_env_docs() -> Dict[str, Dict]:
    """Get documentation for all environment variables."""
    return {
        e.name: {
            "description": e.description,
            "module": e.module,
            "default": e.default,
            "required": e.required,
            "sensitive": e.sensitive,
        }
        for e in ENV_REGISTRY
    }


# EOF
