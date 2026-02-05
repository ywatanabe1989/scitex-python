#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_dashboard/_templates.py

"""HTML template generation for the dashboard."""

from ._scripts import get_javascript
from ._styles import get_css


def get_dashboard_html() -> str:
    """Generate the main dashboard HTML."""
    css = get_css()
    js = get_javascript()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SciTeX Version Dashboard</title>
    <link rel="icon" type="image/svg+xml" href="/static/version-dashboard-favicon.svg">
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>SciTeX Version Dashboard</h1>
                <span class="last-updated" id="lastUpdated">Loading...</span>
            </div>
            <div class="actions">
                <button class="btn-secondary" onclick="copyResults()">Copy</button>
                <button class="btn-secondary" onclick="exportJSON()">Export JSON</button>
                <button class="btn-secondary" id="autoRefreshBtn" onclick="cycleAutoRefresh()">Auto: Off</button>
                <button class="btn-primary" onclick="refreshData()">Refresh</button>
            </div>
        </header>

        <div class="filters collapsed">
            <div class="filters-toggle" onclick="toggleFilters()">
                <span class="fold-icon">▶</span>
                <span>Filters &amp; Config</span>
            </div>
            <div class="filter-group">
                <h3>Packages</h3>
                <div class="filter-options" id="packageFilters"></div>
            </div>
            <div class="filter-group">
                <h3>Status</h3>
                <div class="filter-options" id="statusFilters">
                    <label><input type="checkbox" value="ok" checked> OK</label>
                    <label><input type="checkbox" value="unreleased" checked> Unreleased</label>
                    <label><input type="checkbox" value="mismatch" checked> Mismatch</label>
                    <label><input type="checkbox" value="outdated" checked> Outdated</label>
                    <label><input type="checkbox" value="unavailable" checked> Unavailable</label>
                </div>
            </div>
            <div class="filter-group">
                <h3>Hosts</h3>
                <div class="filter-options" id="hostFilters"></div>
            </div>
            <div class="filter-group">
                <h3>Remotes</h3>
                <div class="filter-options" id="remoteFilters"></div>
            </div>
            <div class="filter-group">
                <h3>RTD</h3>
                <div class="filter-options" id="rtdFilters"></div>
            </div>
        </div>

        <div class="expand-controls">
            <button class="btn-secondary" onclick="toggleAllCards(true)">Expand All</button>
            <button class="btn-secondary" onclick="toggleAllCards(false)">Collapse All</button>
        </div>

        <div class="summary" id="summary"></div>
        <div class="packages" id="packages"></div>
    </div>

    <div class="overlay" id="overlay"></div>
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <p>Loading version data...</p>
    </div>

    <script>{js}</script>
</body>
</html>
"""


def get_error_html(error: str) -> str:
    """Generate error page HTML."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>Error - SciTeX Dashboard</title>
    <style>
        body {{
            font-family: sans-serif;
            background: #1a1a2e;
            color: #eee;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }}
        .error {{
            background: #16213e;
            padding: 40px;
            border-radius: 10px;
            text-align: center;
        }}
        .error h1 {{ color: #e94560; }}
        .error p {{ color: #aaa; }}
    </style>
</head>
<body>
    <div class="error">
        <h1>Error</h1>
        <p>{error}</p>
    </div>
</body>
</html>
"""


# EOF
