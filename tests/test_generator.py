import pytest
import os
from unittest.mock import patch, MagicMock
from src.generator import generate

@pytest.fixture(autouse=True)
def mock_env_api_key(monkeypatch):
    """
    Sets a dummy GROQ_API_KEY environment variable for all tests.
    """
    monkeypatch.setenv("GROQ_API_KEY", "mock-api-key")

@patch("src.generator.OpenAI")
def test_generate_success(mock_openai_class):
    """
    Verifies that generate correctly builds prompts, calls OpenAI Chat Completions API,
    and returns the response.
    """
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Mock return value of client.chat.completions.create
    mock_choice = MagicMock()
    mock_choice.message.content = "Neural networks are models inspired by the brain."
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    
    context = [
        {"text": "Neural networks are brain-inspired models.", "source": "deep_learning.txt"}
    ]
    query = "What are neural networks?"
    
    result = generate(query, context)
    
    assert result == "Neural networks are models inspired by the brain."
    
    # Verify that client.chat.completions.create was called with formatted prompts
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["temperature"] == 0.0
    
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    
    # Check system prompt
    assert messages[0]["role"] == "system"
    assert "ONLY the context provided below" in messages[0]["content"]
    
    # Check user prompt contents
    assert messages[1]["role"] == "user"
    assert "Context:\n[1] Neural networks are brain-inspired models.  (source: deep_learning.txt)" in messages[1]["content"]
    assert "User Question:\nWhat are neural networks?" in messages[1]["content"]

def test_generate_empty_context():
    """
    Verifies that calling generate with empty context returns "I don't know..."
    gracefully without invoking the OpenAI client.
    """
    with patch("src.generator.OpenAI") as mock_openai_class:
        result = generate("What is AI?", [])
        assert result == "I don't know based on the provided context."
        # Verify OpenAI client is not initialized or called
        mock_openai_class.assert_not_called()

def test_generate_missing_api_key(monkeypatch):
    """
    Verifies that generate raises a ValueError if GROQ_API_KEY is missing.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    
    context = [{"text": "Some text", "source": "source.txt"}]
    with pytest.raises(ValueError) as excinfo:
        generate("query", context)
        
    assert "GROQ_API_KEY is missing" in str(excinfo.value)
