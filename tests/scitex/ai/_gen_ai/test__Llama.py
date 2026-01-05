#!/usr/bin/env python3
# Time-stamp: "2025-06-01 14:40:00 (ywatanabe)"
# File: ./tests/scitex/ai/_gen_ai/test__Llama.py

"""Tests for scitex.ai._gen_ai._Llama module."""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock the llama module before importing Llama
# This is necessary because the llama package may not be installed
sys.modules["llama"] = MagicMock()

# Import after mocking
from scitex.ai._gen_ai import Llama


class TestLlama:
    """Test suite for Llama class."""

    @pytest.fixture
    def mock_llama_builder(self):
        """Create a mock Llama builder."""
        mock_generator = Mock()
        mock_result = [{"generation": {"content": "Test response"}}]
        mock_generator.chat_completion.return_value = mock_result
        return mock_generator

    @pytest.fixture
    def mock_env_setup(self):
        """Mock environment variables for distributed setup."""
        env_vars = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
            "WORLD_SIZE": "1",
            "RANK": "0",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            yield

    def test_init_with_default_paths(self, mock_env_setup):
        """Test initialization with default paths."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(model="Meta-Llama-3-8B", api_key="test-key")
                assert llama_ai.model == "Meta-Llama-3-8B"
                assert llama_ai.ckpt_dir == "Meta-Meta-Llama-3-8B/"
                assert (
                    llama_ai.tokenizer_path == "./Meta-Meta-Llama-3-8B/tokenizer.model"
                )
                assert llama_ai.max_seq_len == 32_768
                assert llama_ai.max_batch_size == 4

    def test_init_with_custom_paths(self, mock_env_setup):
        """Test initialization with custom paths."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(
                    ckpt_dir="/custom/path/",
                    tokenizer_path="/custom/tokenizer.model",
                    model="Meta-Llama-3-8B",
                    api_key="test-key",
                )
                assert llama_ai.ckpt_dir == "/custom/path/"
                assert llama_ai.tokenizer_path == "/custom/tokenizer.model"

    def test_environment_setup(self, mock_env_setup):
        """Test environment variables are set correctly."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                with patch("scitex.ai._gen_ai._Llama.print_envs") as mock_print:
                    llama_ai = Llama(model="Meta-Llama-3-8B", api_key="test-key")
                    mock_print.assert_called_once()
                    assert os.environ["MASTER_ADDR"] == "localhost"
                    assert os.environ["MASTER_PORT"] == "12355"
                    assert os.environ["WORLD_SIZE"] == "1"
                    assert os.environ["RANK"] == "0"

    def test_init_client(self, mock_env_setup, mock_llama_builder):
        """Test client initialization."""
        # Patch _init_client to return our mock builder
        with patch.object(Llama, "verify_model"):
            with patch.object(Llama, "_init_client", return_value=mock_llama_builder):
                llama_ai = Llama(model="Meta-Llama-3-8B", api_key="test-key")
                assert llama_ai.client == mock_llama_builder

    def test_api_call_static(self, mock_env_setup, mock_llama_builder):
        """Test static API call."""
        with patch.object(Llama, "verify_model"):
            with patch.object(Llama, "_init_client", return_value=mock_llama_builder):
                llama_ai = Llama(model="Meta-Llama-3-8B", api_key="test-key")
                llama_ai.history = [{"role": "user", "content": "Test"}]

                result = llama_ai._api_call_static()

                assert result == "Test response"
                mock_llama_builder.chat_completion.assert_called_once()
                call_args = mock_llama_builder.chat_completion.call_args
                assert call_args[0][0] == [llama_ai.history]  # dialogs
                assert call_args[1]["temperature"] == 1.0
                assert call_args[1]["top_p"] == 0.9

    def test_api_call_stream(self, mock_env_setup, mock_llama_builder):
        """Test streaming API call (simulated)."""
        with patch.object(Llama, "verify_model"):
            with patch.object(Llama, "_init_client", return_value=mock_llama_builder):
                llama_ai = Llama(
                    model="Meta-Llama-3-8B", stream=True, api_key="test-key"
                )
                llama_ai.history = [{"role": "user", "content": "Test"}]

                # Since Llama doesn't have native streaming, it simulates by character
                result = list(llama_ai._api_call_stream())

                assert "".join(result) == "Test response"
                assert len(result) == len("Test response")  # One char per yield

    def test_max_gen_len_parameter(self, mock_env_setup, mock_llama_builder):
        """Test max_gen_len parameter."""
        with patch.object(Llama, "verify_model"):
            with patch.object(Llama, "_init_client", return_value=mock_llama_builder):
                llama_ai = Llama(
                    model="Meta-Llama-3-8B", max_gen_len=2048, api_key="test-key"
                )
                llama_ai.history = [{"role": "user", "content": "Test"}]

                llama_ai._api_call_static()

                call_kwargs = mock_llama_builder.chat_completion.call_args[1]
                assert call_kwargs["max_gen_len"] == 2048

    def test_temperature_setting(self, mock_env_setup, mock_llama_builder):
        """Test temperature parameter is passed correctly."""
        with patch.object(Llama, "verify_model"):
            with patch.object(Llama, "_init_client", return_value=mock_llama_builder):
                llama_ai = Llama(
                    model="Meta-Llama-3-8B", temperature=0.5, api_key="test-key"
                )
                llama_ai.history = [{"role": "user", "content": "Test"}]
                llama_ai._api_call_static()

                call_kwargs = mock_llama_builder.chat_completion.call_args[1]
                assert call_kwargs["temperature"] == 0.5

    @pytest.mark.parametrize("stream", [True, False])
    def test_stream_parameter(self, mock_env_setup, stream):
        """Test stream parameter handling."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(
                    model="Meta-Llama-3-8B", stream=stream, api_key="test-key"
                )
                assert llama_ai.stream == stream

    def test_n_keep_parameter(self, mock_env_setup):
        """Test n_keep parameter for history management."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(model="Meta-Llama-3-8B", n_keep=5, api_key="test-key")
                assert llama_ai.n_keep == 5

    def test_seed_parameter(self, mock_env_setup):
        """Test seed parameter initialization."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(model="Meta-Llama-3-8B", seed=42, api_key="test-key")
                assert llama_ai.seed == 42

    def test_system_setting(self, mock_env_setup):
        """Test system setting initialization."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                system_msg = "You are a helpful assistant"
                llama_ai = Llama(
                    model="Meta-Llama-3-8B",
                    system_setting=system_msg,
                    api_key="test-key",
                )
                assert llama_ai.system_setting == system_msg

    def test_str_method(self, mock_env_setup):
        """Test __str__ method returns 'Llama'."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(model="Meta-Llama-3-8B", api_key="test-key")
                assert str(llama_ai) == "Llama"

    def test_verify_model(self, mock_env_setup):
        """Test verify_model method (should pass for Llama)."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(model="Meta-Llama-3-8B", api_key="test-key")
                # Should not raise any exception
                llama_ai.verify_model()

    @pytest.mark.parametrize(
        "model,expected_ckpt",
        [
            ("Meta-Llama-3-8B", "Meta-Meta-Llama-3-8B/"),
            ("Meta-Llama-3-70B", "Meta-Meta-Llama-3-70B/"),
            ("Llama-2-7b", "Meta-Llama-2-7b/"),
        ],
    )
    def test_different_models(self, mock_env_setup, model, expected_ckpt):
        """Test initialization with different Llama models."""
        with patch.object(Llama, "_init_client", return_value=Mock()):
            with patch.object(Llama, "verify_model"):
                llama_ai = Llama(model=model, api_key="test-key")
                assert llama_ai.model == model
                assert llama_ai.ckpt_dir == expected_ckpt

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF
