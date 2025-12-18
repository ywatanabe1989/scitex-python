#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/_figz_modules/_legacy.py

"""Legacy compatibility methods for Figz (deprecated)."""

import warnings
from typing import Any, Dict, List, Optional


class FigzLegacyMixin:
    """Mixin providing deprecated panel-based API for backwards compatibility."""

    @property
    def panels(self) -> List[Dict[str, Any]]:
        warnings.warn(
            "'panels' is deprecated. Use 'elements' with type='plot' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return [e for e in self.elements if e.get("type") == "plot"]

    def add_panel(
        self,
        panel_id: str,
        pltz_bytes: bytes,
        position: Optional[Dict[str, float]] = None,
        size: Optional[Dict[str, float]] = None,
    ) -> None:
        warnings.warn(
            "'add_panel' is deprecated. Use add_element(id, 'plot', ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.add_element(panel_id, "plot", pltz_bytes, position, size)

    def get_panel(self, panel_id: str) -> Optional[Dict[str, Any]]:
        warnings.warn(
            "'get_panel' is deprecated. Use get_element() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_element(panel_id)

    def get_panel_pltz(self, panel_id: str) -> Optional[bytes]:
        warnings.warn(
            "'get_panel_pltz' is deprecated. Use get_element_content() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_element_content(panel_id)

    def list_panel_ids(self) -> List[str]:
        warnings.warn(
            "'list_panel_ids' is deprecated. Use list_element_ids('plot') instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.list_element_ids("plot")


# EOF
