"""
Tests for OpenAIAPIClient

This client handles ALL OpenAI API communication.
Critical for ensuring proper error handling and API interaction.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dst_data_builder.gpt_generators.openai_api_client import OpenAIAPIClient


# ============================================================================
# Mock Response Factory
# ============================================================================

def create_mock_response(content):
    """Create a mock OpenAI API response"""
    class MockMessage:
        def __init__(self, content):
            self.content = content
    
    class MockChoice:
        def __init__(self, message):
            self.message = message
    
    class MockResponse:
        def __init__(self, choices):
            self.choices = choices
    
    return MockResponse([MockChoice(MockMessage(content))])


def mock_api_call(client, monkeypatch, response_or_exception):
    """Helper to mock the API call with either a response or exception"""
    async def fake_create(*args, **kwargs):
        if isinstance(response_or_exception, Exception):
            raise response_or_exception
        return response_or_exception
    
    mock_completions = MagicMock()
    mock_completions.create = AsyncMock(side_effect=fake_create)
    monkeypatch.setattr(client.client.chat, "completions", mock_completions)


# ============================================================================
# Initialization Tests
# ============================================================================

@pytest.mark.parametrize("api_key,base_url,expected_url", [
    ("test_key_123", "https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
    ("test_key_123", None, None),
    ("invalid", None, None),
])
def test_client_initialization(api_key, base_url, expected_url):
    """Test client initialization with various configurations"""
    client = OpenAIAPIClient(api_key=api_key, base_url=base_url)
    
    assert client.api_key == api_key
    assert client.base_url == expected_url
    assert client.client is not None


# ============================================================================
# API Call Tests
# ============================================================================

@pytest.mark.asyncio
async def test_generate_completion_success(monkeypatch):
    """Test successful API call"""
    client = OpenAIAPIClient(api_key="test_key")
    response = create_mock_response('{"steps": [{"step_id": "S1"}]}')
    mock_api_call(client, monkeypatch, response)
    
    success, result = await client.generate_completion(
        prompt="Test prompt", model="gpt-4o", temperature=0.1, max_tokens=1000
    )
    
    assert success
    assert "steps" in result


@pytest.mark.asyncio
@pytest.mark.parametrize("exception,expected_in_msg", [
    (Exception("Rate limit exceeded"), "rate limit"),
    (TimeoutError("Connection timeout"), "api_error"),
    (ValueError("Invalid parameter"), "api_error"),
])
async def test_generate_completion_errors(monkeypatch, exception, expected_in_msg):
    """Test various API error scenarios"""
    client = OpenAIAPIClient(api_key="test_key")
    mock_api_call(client, monkeypatch, exception)
    
    success, result = await client.generate_completion(
        prompt="Test prompt", model="gpt-4o", temperature=0.1, max_tokens=1000
    )
    
    assert not success
    assert expected_in_msg in result.lower()


@pytest.mark.asyncio
async def test_generate_completion_client_none():
    """Test handling when client is None"""
    client = OpenAIAPIClient(api_key="test_key")
    client.client = None
    
    success, result = await client.generate_completion(
        prompt="Test prompt", model="gpt-4o", temperature=0.1, max_tokens=1000
    )
    
    assert not success
    assert "not initialized" in result.lower()


@pytest.mark.asyncio
async def test_generate_completion_empty_response(monkeypatch):
    """Test handling of empty API response"""
    client = OpenAIAPIClient(api_key="test_key")
    response = create_mock_response("")
    mock_api_call(client, monkeypatch, response)
    
    success, result = await client.generate_completion(
        prompt="Test prompt", model="gpt-4o", temperature=0.1, max_tokens=1000
    )
    
    assert success
    assert result == ""


@pytest.mark.asyncio
async def test_generate_completion_with_long_prompt(monkeypatch):
    """Test handling of very long prompts"""
    client = OpenAIAPIClient(api_key="test_key")
    response = create_mock_response('{"result": "ok"}')
    mock_api_call(client, monkeypatch, response)
    
    long_prompt = "Test " * 10000  # Very long prompt
    success, result = await client.generate_completion(
        prompt=long_prompt, model="gpt-4o", temperature=0.1, max_tokens=1000
    )
    
    assert success
    assert "ok" in result
