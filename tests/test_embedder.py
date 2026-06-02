import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from src.embedder import embed_texts, embed_query

@patch("src.embedder.get_model")
def test_embed_texts_success(mock_get_model):
    """
    Verifies that embed_texts calls SentenceTransformer model encode and returns embeddings.
    """
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    
    # Mock return value of model.encode (which returns a numpy array)
    mock_model.encode.return_value = np.array([[0.1] * 384])
    
    result = embed_texts(["Hello world"])
    
    assert len(result) == 1
    assert result[0] == [0.1] * 384
    mock_model.encode.assert_called_once_with(["Hello world"])

@patch("src.embedder.get_model")
def test_embed_texts_empty(mock_get_model):
    """
    Verifies that passing empty list to embed_texts returns empty list.
    """
    result = embed_texts([])
    assert result == []
    mock_get_model.assert_not_called()

@patch("src.embedder.get_model")
def test_embed_query(mock_get_model):
    """
    Verifies embed_query calls SentenceTransformer and returns a single list of floats.
    """
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    mock_model.encode.return_value = np.array([[0.5] * 384])
    
    result = embed_query("hello query")
    
    assert len(result) == 384
    assert result == [0.5] * 384
    mock_model.encode.assert_called_once_with(["hello query"])
