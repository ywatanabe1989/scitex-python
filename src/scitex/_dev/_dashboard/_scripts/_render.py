#!/usr/bin/env python3
# Timestamp: 2026-02-05
# File: scitex/_dev/_dashboard/_scripts/_render.py

"""Data rendering functions for dashboard JavaScript."""


def get_render_js() -> str:
    """Return JavaScript for data rendering."""
    return """
function renderData() {
    if (!cachedData) return;

    // Save expanded states before re-render
    const expandedCards = new Set();
    document.querySelectorAll('.package-card:not(.collapsed)').forEach(card => {
        const nameEl = card.querySelector('.package-name');
        if (nameEl) expandedCards.add(nameEl.textContent);
    });

    const filters = getSelectedFilters();
    const packages = cachedData.packages || {};

    const filteredPackages = Object.entries(packages)
        .filter(([name, info]) => {
            if (!filters.packages.includes(name)) return false;
            const effectiveStatus = getEffectiveStatus(name, info);
            if (!filters.statuses.includes(effectiveStatus)) return false;
            return true;
        });

    const summary = {
        total: filteredPackages.length,
        ok: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'ok').length,
        unreleased: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'unreleased').length,
        mismatch: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'mismatch').length,
        outdated: filteredPackages.filter(([n, i]) => getEffectiveStatus(n, i) === 'outdated').length
    };

    document.getElementById('summary').innerHTML = `
        <div class="summary-card total"><div class="number">${summary.total}</div><div class="label">Total</div></div>
        <div class="summary-card ok"><div class="number">${summary.ok}</div><div class="label">OK</div></div>
        <div class="summary-card unreleased"><div class="number">${summary.unreleased}</div><div class="label">Unreleased</div></div>
        <div class="summary-card mismatch"><div class="number">${summary.mismatch}</div><div class="label">Mismatch</div></div>
    `;

    document.getElementById('packages').innerHTML = filteredPackages.map(([name, info]) => {
        const local = info.local || {};
        const git = info.git || {};
        const remote = info.remote || {};
        const hostData = cachedData.hosts || {};
        const remoteData = cachedData.remotes || {};
        const rtdData = cachedData.rtd || {};

        const hostVersions = Object.entries(hostData)
            .filter(([h]) => !h.startsWith('_') && filters.hosts.includes(h))
            .map(([hostName, hostInfo]) => ({ name: hostName, ...(hostInfo[name] || {}) }));

        const remoteVersions = Object.entries(remoteData)
            .filter(([r]) => !r.startsWith('_') && filters.remotes.includes(r))
            .map(([remoteName, remoteInfo]) => ({ name: remoteName, ...(remoteInfo[name] || {}) }));

        const rtdStatus = {};
        Object.entries(rtdData).forEach(([version, pkgData]) => {
            if (pkgData[name]) {
                rtdStatus[version] = pkgData[name];
            }
        });

        return renderPackageCard(name, info, local, git, remote, hostVersions, remoteVersions, rtdStatus);
    }).join('');

    // Restore expanded states after re-render
    document.querySelectorAll('.package-card').forEach(card => {
        const nameEl = card.querySelector('.package-name');
        if (nameEl && expandedCards.has(nameEl.textContent)) {
            card.classList.remove('collapsed');
        }
    });
}
"""


# EOF
