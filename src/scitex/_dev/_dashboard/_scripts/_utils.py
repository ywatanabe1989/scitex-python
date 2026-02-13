#!/usr/bin/env python3
# Timestamp: 2026-02-05
# File: scitex/_dev/_dashboard/_scripts/_utils.py

"""Utility functions for dashboard JavaScript."""


def get_utils_js() -> str:
    """Return JavaScript utility functions."""
    return """
function toggleCard(header) {
    const card = header.parentElement;
    card.classList.toggle('collapsed');
}

function toggleAllCards(expand) {
    document.querySelectorAll('.package-card').forEach(card => {
        if (expand) {
            card.classList.remove('collapsed');
        } else {
            card.classList.add('collapsed');
        }
    });
}

function exportJSON() {
    if (!cachedData) return;
    const blob = new Blob([JSON.stringify(cachedData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scitex-versions.json';
    a.click();
    URL.revokeObjectURL(url);
}

function copyResults() {
    if (!cachedData) return;
    const packages = cachedData.packages || {};
    const hosts = cachedData.hosts || {};
    const hostMeta = hosts._meta || {};

    const lines = ['# SciTeX Version Report', ''];

    Object.entries(packages).forEach(([name, info]) => {
        const local = info.local || {};
        const git = info.git || {};
        const remote = info.remote || {};
        const status = getEffectiveStatus(name, info);

        lines.push(`## ${name} [${status}]`);
        lines.push(`LOCAL: toml=${local.pyproject_toml || '-'}, installed=${local.installed || '-'}, tag=${git.latest_tag || '-'}, branch=${git.branch || '-'}`);

        // Add host info
        Object.entries(hosts).forEach(([hostName, hostInfo]) => {
            if (hostName.startsWith('_')) return;
            const pkgInfo = hostInfo[name] || {};
            const meta = hostMeta[hostName] || {};
            lines.push(`${hostName.toUpperCase()} (${meta.hostname || '?'}): toml=${pkgInfo.toml || '-'}, installed=${pkgInfo.installed || '-'}, tag=${pkgInfo.git_tag || '-'}, branch=${pkgInfo.git_branch || '-'}`);
        });

        lines.push(`PYPI: ${remote.pypi || '-'}`);
        lines.push('');
    });

    navigator.clipboard.writeText(lines.join('\\n')).then(() => {
        const btn = document.querySelector('[onclick="copyResults()"]');
        const orig = btn.textContent;
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = orig; }, 2000);
    });
}
"""


# EOF
