import pytest
import os
import numpy as np
from src.vector_store import VectorStore

def test_cosine_similarity_calculation():
    """
    Verifies that the custom numpy-based cosine similarity calculates correct values
    and ranks results accurately.
    """
    store = VectorStore()
    
    # Add dummy chunks with orthogonal and opposite vectors
    chunk_a = {"chunk_id": "a", "text": "orthogonal"}
    chunk_b = {"chunk_id": "b", "text": "identical"}
    chunk_c = {"chunk_id": "c", "text": "opposite"}
    
    # We use 2D vectors for simplicity of verification
    store.add(chunk_a, [1.0, 0.0])
    store.add(chunk_b, [0.0, 1.0])
    store.add(chunk_c, [-1.0, 0.0])
    
    # Query identical to chunk_a
    results = store.search([1.0, 0.0], top_k=3)
    
    assert len(results) == 3
    # First result should be chunk_a with score ~1.0
    assert results[0]["chunk_id"] == "a"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    
    # Second result should be chunk_b with score ~0.0 (orthogonal)
    assert results[1]["chunk_id"] == "b"
    assert results[1]["score"] == pytest.approx(0.0, abs=1e-5)
    
    # Third result should be chunk_c with score ~-1.0 (opposite)
    assert results[2]["chunk_id"] == "c"
    assert results[2]["score"] == pytest.approx(-1.0, abs=1e-5)
    
    # Verify scores are between -1.0 and 1.0
    for res in results:
        assert -1.0 <= res["score"] <= 1.0

def test_bulk_add_and_search():
    """
    Verifies adding multiple records in bulk and performing top-K retrieval.
    """
    store = VectorStore()
    
    chunks = [
        {"chunk_id": f"chunk_{i}", "text": f"text {i}"}
        for i in range(10)
    ]
    vectors = [
        [float(i), 1.0] for i in range(10)
    ]
    
    store.add(chunks, vectors)
    
    assert len(store.chunks) == 10
    assert len(store.vectors) == 10
    
    # Retrieve top 3
    results = store.search([9.0, 1.0], top_k=3)
    assert len(results) == 3
    # Top result should be chunk_9 since its vector is [9.0, 1.0]
    assert results[0]["chunk_id"] == "chunk_9"
    # Order should be descending similarity
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]

def test_save_and_load(tmp_path):
    """
    Verifies that the vector store can be saved to and loaded from a JSON file,
    preserving all chunk and vector data.
    """
    store = VectorStore()
    chunk = {"chunk_id": "test_chunk", "text": "hello file"}
    vector = [0.25, 0.5, 0.75]
    
    store.add(chunk, vector)
    
    save_path = tmp_path / "store.json"
    store.save(str(save_path))
    
    assert os.path.exists(save_path)
    
    # Create new store and load from the file
    new_store = VectorStore()
    new_store.load(str(save_path))
    
    assert len(new_store.chunks) == 1
    assert len(new_store.vectors) == 1
    assert new_store.chunks[0] == chunk
    np.testing.assert_array_almost_equal(new_store.vectors[0], np.array(vector, dtype=np.float32))

def test_search_empty_store():
    """
    Verifies that search returns an empty list if store is empty.
    """
    store = VectorStore()
    results = store.search([1.0, 0.0], top_k=5)
    assert results == []

def test_load_missing_file_raises_error():
    """
    Verifies loading a missing file raises FileNotFoundError.
    """
    store = VectorStore()
    with pytest.raises(FileNotFoundError):
        store.load("non_existent_vector_store_file.json")
