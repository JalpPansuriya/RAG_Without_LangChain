import pytest
from unittest.mock import MagicMock
from src.retriever import retrieve
from src.vector_store import VectorStore

def test_retrieve_success():
    """
    Verifies that retrieve successfully embeds query, queries vector store,
    and returns correctly formatted chunks.
    """
    # 1. Setup mock embedder
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2]
    
    # 2. Setup actual vector store
    store = VectorStore()
    chunk1 = {"chunk_id": "c1", "source": "doc1.txt", "text": "This is doc 1 content."}
    chunk2 = {"chunk_id": "c2", "source": "doc2.txt", "text": "This is doc 2 content."}
    
    # Simple 2D vectors
    store.add(chunk1, [1.0, 0.0])
    store.add(chunk2, [0.0, 1.0])
    
    # 3. Retrieve
    results = retrieve("dummy query", store, mock_embedder, top_k=2)
    
    assert len(results) == 2
    mock_embedder.embed_query.assert_called_once_with("dummy query")
    
    # Verify structure and contents
    assert "text" in results[0]
    assert "source" in results[0]
    assert "score" in results[0]
    
    # The scores should be cosine similarity of [0.1, 0.2] with [1.0, 0.0] and [0.0, 1.0]
    # [0.1, 0.2] dot [1.0, 0.0] = 0.1
    # [0.1, 0.2] dot [0.0, 1.0] = 0.2
    # So chunk2 should be ranked first since 0.2 > 0.1
    assert results[0]["source"] == "doc2.txt"
    assert results[0]["text"] == "This is doc 2 content."
    assert results[1]["source"] == "doc1.txt"

def test_retrieve_empty_store():
    """
    Verifies that retrieving from an empty vector store returns an empty list.
    """
    mock_embedder = MagicMock()
    store = VectorStore()
    
    results = retrieve("some query", store, mock_embedder, top_k=5)
    assert results == []
    # Embedder should not even be called if store is empty
    mock_embedder.embed_query.assert_not_called()
