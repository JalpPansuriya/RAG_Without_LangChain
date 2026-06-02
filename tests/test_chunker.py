import pytest
from src.chunker import chunk_fixed, chunk_by_sentence

def test_chunk_fixed_success():
    """
    Verifies fixed-size character chunking splits text with overlap and metadata.
    """
    doc = {
        "filename": "test_doc.txt",
        "filepath": "data/test_doc.txt",
        "content": "This is paragraph one. This is paragraph two. This is paragraph three.",
        "char_count": 70,
        "word_count": 12
    }
    
    # 20 chars per chunk, 5 chars overlap
    chunks = chunk_fixed(doc, chunk_size=20, overlap=5)
    
    assert len(chunks) > 0
    
    # Check that metadata exists
    for i, chunk in enumerate(chunks):
        assert chunk["source"] == "test_doc.txt"
        assert chunk["chunk_index"] == i
        assert chunk["chunk_id"] == f"test_doc_{i}"
        assert "start_char" in chunk
        assert "end_char" in chunk
        assert len(chunk["text"]) <= 20
        assert chunk["text"].strip() != ""
        
    # Check overlap: chunk 1 end_char - overlap should equal chunk 2 start_char
    if len(chunks) > 1:
        assert chunks[1]["start_char"] == chunks[0]["end_char"] - 5

def test_chunk_by_sentence_success():
    """
    Verifies sentence-based chunking groups sentences up to size limit.
    """
    doc = {
        "filename": "test_doc.txt",
        "content": "First sentence. Second sentence. Third sentence here. Fourth sentence.",
    }
    
    # Let's chunk with size that fits 2 sentences per chunk
    # "First sentence. Second sentence." is 31 chars.
    chunks = chunk_by_sentence(doc, chunk_size=40)
    
    assert len(chunks) == 2
    assert chunks[0]["text"] == "First sentence. Second sentence."
    assert chunks[1]["text"] == "Third sentence here. Fourth sentence."
    
    for i, chunk in enumerate(chunks):
        assert chunk["source"] == "test_doc.txt"
        assert chunk["chunk_index"] == i
        assert chunk["chunk_id"] == f"test_doc_{i}"
        assert chunk["text"] in doc["content"]

def test_chunk_fixed_invalid_overlap():
    """
    Verifies that invalid overlap values raise ValueError.
    """
    doc = {"filename": "test.txt", "content": "Sample text content."}
    
    # Overlap >= chunk_size
    with pytest.raises(ValueError):
        chunk_fixed(doc, chunk_size=10, overlap=10)
        
    # Negative overlap
    with pytest.raises(ValueError):
        chunk_fixed(doc, chunk_size=10, overlap=-2)

def test_chunk_empty_content():
    """
    Verifies that empty content returns an empty chunk list.
    """
    doc = {"filename": "empty.txt", "content": "    "}
    chunks = chunk_fixed(doc, chunk_size=10, overlap=2)
    assert len(chunks) == 0
    
    chunks_s = chunk_by_sentence(doc, chunk_size=10)
    assert len(chunks_s) == 0
