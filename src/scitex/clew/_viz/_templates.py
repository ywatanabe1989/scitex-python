#!/usr/bin/env python3
# Timestamp: "2026-02-01 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/verify/_viz/_templates.py
"""HTML templates for verification DAG visualization."""

from __future__ import annotations

from datetime import datetime


def get_timestamp() -> str:
    """Get current timestamp for footer."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_html_template(title: str, mermaid_code: str) -> str:
    """
    Generate sleek HTML template with icons for verification DAG.

    Parameters
    ----------
    title : str
        Page title
    mermaid_code : str
        Mermaid diagram code

    Returns
    -------
    str
        Complete HTML document
    """
    timestamp = get_timestamp()

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <style>
        :root {{
            --verified-bg: #d4edda;
            --verified-border: #28a745;
            --verified-text: #155724;
            --failed-bg: #f8d7da;
            --failed-border: #dc3545;
            --failed-text: #721c24;
            --file-bg: #fff3cd;
            --file-border: #ffc107;
            --script-bg: #cce5ff;
            --script-border: #4a6baf;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        .header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 3px solid var(--script-border);
        }}
        .header-icon {{
            width: 48px;
            height: 48px;
            background: var(--script-border);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 24px;
        }}
        h1 {{
            color: #333;
            margin: 0;
            font-size: 1.8rem;
        }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 20px 0;
            padding: 15px;
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            border-radius: 12px;
            border: 1px solid #dee2e6;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .legend-icon {{
            width: 28px;
            height: 28px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }}
        .legend-icon.verified {{
            background: var(--verified-bg);
            color: var(--verified-text);
            border: 2px solid var(--verified-border);
        }}
        .legend-icon.from-scratch {{
            background: var(--verified-bg);
            color: var(--verified-text);
            border: 3px solid var(--verified-border);
            box-shadow: 0 0 0 2px var(--verified-bg);
        }}
        .legend-icon.failed {{
            background: var(--failed-bg);
            color: var(--failed-text);
            border: 2px solid var(--failed-border);
        }}
        .legend-icon.file {{
            background: var(--file-bg);
            color: #856404;
            border: 2px solid var(--file-border);
        }}
        .legend-icon.script {{
            background: var(--script-bg);
            color: #004085;
            border: 2px solid var(--script-border);
        }}
        .legend-text {{
            font-size: 0.9rem;
            color: #495057;
        }}
        .mermaid {{
            margin: 25px 0;
            padding: 20px;
            background: #fafafa;
            border-radius: 12px;
            border: 1px solid #e9ecef;
            overflow-x: auto;
        }}
        .footer {{
            margin-top: 25px;
            padding-top: 15px;
            border-top: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #6c757d;
            font-size: 0.85rem;
        }}
        .footer-brand {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .footer-brand i {{
            color: var(--script-border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-icon">
                <i class="fas fa-project-diagram"></i>
            </div>
            <h1>{title}</h1>
        </div>
        <div class="legend">
            <div class="legend-item">
                <span style="font-size:1.2em">🐍</span>
                <span class="legend-text">Python</span>
            </div>
            <div class="legend-item">
                <span style="font-size:1.2em">📊</span>
                <span class="legend-text">CSV</span>
            </div>
            <div class="legend-item">
                <span style="font-size:1.2em">📋</span>
                <span class="legend-text">JSON</span>
            </div>
            <div class="legend-item">
                <span style="font-size:1.2em">⚙️</span>
                <span class="legend-text">Config</span>
            </div>
            <div class="legend-item">
                <div class="legend-icon verified"><i class="fas fa-check"></i></div>
                <span class="legend-text">Verified</span>
            </div>
            <div class="legend-item">
                <div class="legend-icon from-scratch"><i class="fas fa-check-double"></i></div>
                <span class="legend-text">From-scratch</span>
            </div>
            <div class="legend-item">
                <div class="legend-icon failed"><i class="fas fa-times"></i></div>
                <span class="legend-text">Failed</span>
            </div>
        </div>
        <div class="mermaid">
{mermaid_code}
        </div>
        <div class="footer">
            <div class="footer-brand">
                <i class="fas fa-flask"></i>
                <span>Generated by SciTeX Verify</span>
            </div>
            <div>
                <i class="far fa-clock"></i> Generated at: {timestamp}
            </div>
        </div>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            flowchart: {{
                curve: 'basis',
                padding: 20,
                nodeSpacing: 50,
                rankSpacing: 60
            }}
        }});
    </script>
</body>
</html>"""


# EOF
