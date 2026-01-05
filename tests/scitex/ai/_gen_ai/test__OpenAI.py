#!/usr/bin/env python3
# Time-stamp: "2025-06-01 14:25:00 (ywatanabe)"
# File: ./tests/scitex/ai/_gen_ai/test__OpenAI.py

"""Tests for scitex.ai._gen_ai._OpenAI module."""

import pytest

pytest.importorskip("zarr")
import os
from unittest.mock import MagicMock, Mock, patch

from scitex.ai._gen_ai import OpenAI


class TestOpenAI:
    """Test suite for OpenAI class."""

    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_init_with_api_key(self):
        """Test initialization with explicitly provided API key."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(model="gpt-4", api_key="test-api-key")
            assert openai_ai.api_key == "test-api-key"
            assert openai_ai.model == "gpt-4"
            assert openai_ai.provider == "OpenAI"
            assert openai_ai.max_tokens == 8_192  # gpt-4 default

    def test_init_with_explicit_api_key(self):
        """Test initialization with explicitly provided API key."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(api_key="explicit-key", model="gpt-3.5-turbo")
            assert openai_ai.api_key == "explicit-key"

    def test_init_client(self):
        """Test client initialization."""
        with patch("scitex.ai._gen_ai._OpenAI._OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_openai_class.return_value = mock_client

            openai_ai = OpenAI(model="gpt-4", api_key="test-api-key")

            mock_openai_class.assert_called_once_with(api_key="test-api-key")
            assert openai_ai.client == mock_client

    @pytest.mark.parametrize(
        "model,expected_max_tokens",
        [
            ("gpt-4-turbo", 128_000),
            ("gpt-4", 8_192),
            ("gpt-3.5-turbo-16k", 16_384),
            ("gpt-3.5-turbo", 4_096),
            ("unknown-model", 4_096),  # default
        ],
    )
    def test_max_tokens_by_model(self, model, expected_max_tokens):
        """Test that different models get appropriate max tokens."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(model=model, api_key="test-key")
            assert openai_ai.max_tokens == expected_max_tokens

    def test_o1_model_initialization(self):
        """Test initialization with o1 models and reasoning effort."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            # Test various o1 model formats
            for effort in ["low", "midium", "high"]:
                openai_ai = OpenAI(model=f"o1-{effort}", api_key="test-key")
                assert openai_ai.model == "o1"  # effort stripped
                assert openai_ai.passed_model == f"o1-{effort}"

    def test_api_call_static(self, mock_openai_client):
        """Test static API call."""
        with patch.object(OpenAI, "_init_client", return_value=mock_openai_client):
            openai_ai = OpenAI(model="gpt-4", stream=False, api_key="test-key")
            openai_ai.history = [{"role": "user", "content": "Test"}]

            result = openai_ai._api_call_static()

            assert result == "Test response"
            assert openai_ai.input_tokens == 10
            assert openai_ai.output_tokens == 20

            mock_openai_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["temperature"] == 1.0
            assert call_kwargs["max_tokens"] == 8_192

    def test_api_call_static_o1_model(self, mock_openai_client):
        """Test static API call with o1 model."""
        with patch.object(OpenAI, "_init_client", return_value=mock_openai_client):
            openai_ai = OpenAI(model="o1-low", stream=False, api_key="test-key")
            openai_ai.history = [{"role": "user", "content": "Test"}]

            result = openai_ai._api_call_static()

            # Check that max_tokens was removed for o1 models
            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert "max_tokens" not in call_kwargs
            assert call_kwargs["reasoning_effort"] == "low"
            assert call_kwargs["model"] == "o1"

    def test_api_call_stream(self):
        """Test streaming API call."""
        mock_client = Mock()

        # Mock stream chunks
        chunks = [
            Mock(
                choices=[Mock(delta=Mock(content="Hello"))],
                usage=Mock(prompt_tokens=5, completion_tokens=0),
            ),
            Mock(
                choices=[Mock(delta=Mock(content=" world"))],
                usage=Mock(prompt_tokens=0, completion_tokens=10),
            ),
            Mock(choices=[Mock(delta=Mock(content="!"))], usage=None),
        ]

        # Set up side effects for attribute access
        for chunk in chunks:
            if chunk.usage:
                chunk.usage.prompt_tokens = getattr(chunk.usage, "prompt_tokens", 0)
                chunk.usage.completion_tokens = getattr(
                    chunk.usage, "completion_tokens", 0
                )

        mock_client.chat.completions.create.return_value = iter(chunks)

        with patch.object(OpenAI, "_init_client", return_value=mock_client):
            openai_ai = OpenAI(model="gpt-4", stream=True, api_key="test-key")
            openai_ai.history = [{"role": "user", "content": "Test"}]

            result = list(openai_ai._api_call_stream())

            # Should yield complete sentences/words at punctuation
            assert len(result) >= 2  # At least "Hello " and "world!"
            assert "".join(result) == "Hello world!"

    def test_api_call_stream_o1_model(self, mock_openai_client):
        """Test streaming API call with o1 model falls back to static."""
        with patch.object(OpenAI, "_init_client", return_value=mock_openai_client):
            with patch.object(
                OpenAI, "_api_call_static", return_value="Full response"
            ) as mock_static:
                openai_ai = OpenAI(model="o1-high", stream=True, api_key="test-key")
                openai_ai.history = [{"role": "user", "content": "Test"}]

                result = list(openai_ai._api_call_stream())

                # Should call static method and stream character by character
                mock_static.assert_called_once()
                assert "".join(result) == "Full response"

    def test_api_format_history_text_only(self):
        """Test formatting history with text-only messages."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(model="gpt-4", api_key="test-key")

            history = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]

            formatted = openai_ai._api_format_history(history)
            assert len(formatted) == 2
            assert formatted[0]["role"] == "user"
            assert formatted[0]["content"] == "Hello"

    def test_api_format_history_with_images(self):
        """Test formatting history with image content."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(model="gpt-4", api_key="test-key")

            history = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {"type": "_image", "_image": "base64data"},
                    ],
                }
            ]

            formatted = openai_ai._api_format_history(history)
            assert len(formatted) == 1
            assert len(formatted[0]["content"]) == 2
            assert formatted[0]["content"][0]["type"] == "text"
            assert formatted[0]["content"][1]["type"] == "image_url"
            assert (
                formatted[0]["content"][1]["image_url"]["url"]
                == "data:image/jpeg;base64,base64data"
            )

    def test_temperature_setting(self, mock_openai_client):
        """Test temperature parameter is passed correctly."""
        with patch.object(OpenAI, "_init_client", return_value=mock_openai_client):
            openai_ai = OpenAI(model="gpt-4", temperature=0.5, api_key="test-key")
            openai_ai.history = [{"role": "user", "content": "Test"}]
            openai_ai._api_call_static()

            # Check temperature was passed
            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.5

    def test_seed_parameter(self, mock_openai_client):
        """Test seed parameter is passed correctly."""
        with patch.object(OpenAI, "_init_client", return_value=mock_openai_client):
            openai_ai = OpenAI(model="gpt-4", seed=42, api_key="test-key")
            # The seed should be stored on the instance
            assert openai_ai.seed == 42

            openai_ai.history = [{"role": "user", "content": "Test"}]
            openai_ai._api_call_static()

            # Check seed was passed in API call
            call_kwargs = mock_openai_client.chat.completions.create.call_args[1]
            assert call_kwargs["seed"] == 42

    @pytest.mark.parametrize("stream", [True, False])
    def test_stream_parameter(self, stream):
        """Test stream parameter handling."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(model="gpt-4", stream=stream, api_key="test-key")
            assert openai_ai.stream == stream

    def test_n_keep_parameter(self):
        """Test n_keep parameter for history management."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(model="gpt-4", n_keep=5, api_key="test-key")
            assert openai_ai.n_keep == 5

    def test_custom_max_tokens(self):
        """Test custom max_tokens override."""
        with patch.object(OpenAI, "_init_client", return_value=Mock()):
            openai_ai = OpenAI(model="gpt-4", max_tokens=1000, api_key="test-key")
            assert openai_ai.max_tokens == 1000  # Custom value, not default 8192

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF
