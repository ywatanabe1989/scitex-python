#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-11-09"
# File: ./tests/scitex/session/test__manager.py

"""Tests for SessionManager class."""

import pytest
from datetime import datetime
from scitex.session import SessionManager
from scitex.session._manager import get_global_session_manager


class TestSessionManagerBasic:
    """Basic SessionManager functionality tests."""

    def test_initialization(self):
        """Test SessionManager can be initialized."""
        manager = SessionManager()
        assert manager is not None
        assert hasattr(manager, 'active_sessions')
        assert manager.active_sessions == {}

    def test_create_session(self):
        """Test creating a new session."""
        manager = SessionManager()
        config = {'test': 'value'}

        manager.create_session('test_id', config)

        assert 'test_id' in manager.active_sessions
        assert manager.active_sessions['test_id']['config'] == config
        assert manager.active_sessions['test_id']['status'] == 'running'
        assert isinstance(manager.active_sessions['test_id']['start_time'], datetime)

    def test_close_session(self):
        """Test closing a session."""
        manager = SessionManager()
        config = {'test': 'value'}

        manager.create_session('test_id', config)
        manager.close_session('test_id')

        assert manager.active_sessions['test_id']['status'] == 'closed'
        assert 'end_time' in manager.active_sessions['test_id']
        assert isinstance(manager.active_sessions['test_id']['end_time'], datetime)

    def test_close_nonexistent_session(self):
        """Test closing non-existent session doesn't raise error."""
        manager = SessionManager()
        # Should not raise
        manager.close_session('nonexistent')


class TestSessionManagerQueries:
    """Test SessionManager query methods."""

    def test_get_active_sessions_empty(self):
        """Test get_active_sessions with no sessions."""
        manager = SessionManager()
        active = manager.get_active_sessions()

        assert active == {}

    def test_get_active_sessions_with_running(self):
        """Test get_active_sessions returns only running sessions."""
        manager = SessionManager()

        manager.create_session('session1', {'data': '1'})
        manager.create_session('session2', {'data': '2'})
        manager.close_session('session1')

        active = manager.get_active_sessions()

        assert 'session2' in active
        assert 'session1' not in active  # Closed session not returned

    def test_get_session_exists(self):
        """Test getting specific session."""
        manager = SessionManager()
        config = {'test': 'value'}

        manager.create_session('test_id', config)
        session_info = manager.get_session('test_id')

        assert session_info is not None
        assert session_info['config'] == config
        assert session_info['status'] == 'running'

    def test_get_session_nonexistent(self):
        """Test getting non-existent session returns empty dict."""
        manager = SessionManager()
        session_info = manager.get_session('nonexistent')

        assert session_info == {}

    def test_list_sessions_all(self):
        """Test list_sessions returns all sessions."""
        manager = SessionManager()

        manager.create_session('session1', {'data': '1'})
        manager.create_session('session2', {'data': '2'})
        manager.close_session('session1')

        all_sessions = manager.list_sessions()

        assert 'session1' in all_sessions
        assert 'session2' in all_sessions
        assert all_sessions['session1']['status'] == 'closed'
        assert all_sessions['session2']['status'] == 'running'


class TestSessionManagerMultiple:
    """Test SessionManager with multiple sessions."""

    def test_multiple_sessions(self):
        """Test managing multiple sessions simultaneously."""
        manager = SessionManager()

        # Create multiple sessions
        for i in range(5):
            manager.create_session(f'session{i}', {'index': i})

        active = manager.get_active_sessions()
        assert len(active) == 5

        # Close some
        manager.close_session('session0')
        manager.close_session('session2')

        active = manager.get_active_sessions()
        assert len(active) == 3

        all_sessions = manager.list_sessions()
        assert len(all_sessions) == 5

    def test_session_id_uniqueness(self):
        """Test that same ID overwrites previous session."""
        manager = SessionManager()

        manager.create_session('test', {'version': 1})
        manager.create_session('test', {'version': 2})

        session_info = manager.get_session('test')
        assert session_info['config']['version'] == 2


class TestGlobalSessionManager:
    """Test global session manager singleton."""

    def test_get_global_session_manager(self):
        """Test getting global session manager."""
        manager1 = get_global_session_manager()
        manager2 = get_global_session_manager()

        # Should be same instance
        assert manager1 is manager2

    def test_global_manager_is_session_manager(self):
        """Test global manager is SessionManager instance."""
        manager = get_global_session_manager()
        assert isinstance(manager, SessionManager)

    def test_global_manager_persists_data(self):
        """Test global manager persists data across calls."""
        manager1 = get_global_session_manager()
        manager1.create_session('persistent', {'data': 'value'})

        manager2 = get_global_session_manager()
        session_info = manager2.get_session('persistent')

        assert session_info['config']['data'] == 'value'


class TestSessionManagerIntegration:
    """Integration tests for SessionManager."""

    def test_typical_workflow(self):
        """Test typical session workflow."""
        manager = SessionManager()

        # Start session
        session_id = 'exp_001'
        config = {
            'experiment': 'test',
            'seed': 42
        }
        manager.create_session(session_id, config)

        # Verify running
        active = manager.get_active_sessions()
        assert session_id in active

        # Close session
        manager.close_session(session_id)

        # Verify closed
        active = manager.get_active_sessions()
        assert session_id not in active

        # But still in all sessions
        all_sessions = manager.list_sessions()
        assert session_id in all_sessions
        assert all_sessions[session_id]['status'] == 'closed'

    def test_session_timing(self):
        """Test session timing information."""
        import time
        manager = SessionManager()

        manager.create_session('timing_test', {})
        start_time = manager.active_sessions['timing_test']['start_time']

        time.sleep(0.1)  # Small delay

        manager.close_session('timing_test')
        end_time = manager.active_sessions['timing_test']['end_time']

        # End time should be after start time
        assert end_time > start_time


if __name__ == "__main__":
    import os
    pytest.main([os.path.abspath(__file__), "-v"])


# EOF
