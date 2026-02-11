#!/usr/bin/env python3
# Timestamp: "2025-08-21 20:36:50 (ywatanabe)"
# File: /home/ywatanabe/proj/SciTeX-Code/src/scitex/session/_manager.py
# ----------------------------------------
from __future__ import annotations

import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""Session manager for tracking active experiment sessions."""

from datetime import datetime
from typing import Any, Dict


class SessionManager:
    """Manages experiment sessions with tracking and lifecycle management."""

    def __init__(self):
        self.active_sessions = {}

    def create_session(
        self,
        session_id: str,
        config: Dict[str, Any],
        script_path: str = None,
    ) -> None:
        """Register a new session.

        Parameters
        ----------
        session_id : str
            Unique identifier for the session
        config : Dict[str, Any]
            Session configuration dictionary
        script_path : str, optional
            Path to the script being run
        """
        self.active_sessions[session_id] = {
            "config": config,
            "start_time": datetime.now(),
            "status": "running",
            "script_path": script_path,
        }

        # Start verification tracking (silent fail)
        try:
            from scitex.clew import on_session_start

            on_session_start(
                session_id=session_id,
                script_path=script_path,
            )
        except Exception:
            pass

    def close_session(
        self,
        session_id: str,
        status: str = "success",
        exit_code: int = 0,
    ) -> None:
        """Mark a session as closed.

        Parameters
        ----------
        session_id : str
            Unique identifier for the session to close
        status : str, optional
            Final status (success, failed, error)
        exit_code : int, optional
            Exit code of the session
        """
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["status"] = "closed"
            self.active_sessions[session_id]["end_time"] = datetime.now()
            self.active_sessions[session_id]["exit_code"] = exit_code

        # Stop verification tracking (silent fail)
        try:
            from scitex.clew import on_session_close

            on_session_close(status=status, exit_code=exit_code)
        except Exception:
            pass

    def get_active_sessions(self) -> Dict[str, Any]:
        """Get all active sessions.

        Returns
        -------
        Dict[str, Any]
            Dictionary of active session information
        """
        return {
            k: v for k, v in self.active_sessions.items() if v["status"] == "running"
        }

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get specific session information.

        Parameters
        ----------
        session_id : str
            Session ID to retrieve

        Returns
        -------
        Dict[str, Any]
            Session information dictionary
        """
        return self.active_sessions.get(session_id, {})

    def list_sessions(self) -> Dict[str, Any]:
        """Get all sessions (active and closed).

        Returns
        -------
        Dict[str, Any]
            Dictionary of all session information
        """
        return self.active_sessions.copy()


# Global session manager
_session_manager = SessionManager()


def get_global_session_manager() -> SessionManager:
    """Get the global session manager instance.

    Returns
    -------
    SessionManager
        Global session manager instance
    """
    return _session_manager


# EOF
