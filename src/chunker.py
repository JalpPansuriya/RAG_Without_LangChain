"""
Text Chunking Module for RAG from Scratch.
Implements text splitting strategies with metadata tracking.
"""

import re
from pathlib import Path

def chunk_fixed(document: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Splits the content of a document into overlapping chunks of a fixed character size.
    
    Args:
        document (dict): Document dictionary containing content and metadata.
        chunk_size (int): Max character length of each chunk.
        overlap (int): Overlap character length between sequential chunks.
        
    Returns:
        list[dict]: List of chunk dictionaries containing ID, source, index, character indices, and text.
        
    Raises:
        ValueError: If overlap is negative or not less than chunk_size.
    """
    if overlap < 0:
        raise ValueError("Overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("Overlap must always be less than chunk_size")
        
    content = document.get("content", "")
    filename = document.get("filename", "")
    content_len = len(content)
    
    chunks = []
    start = 0
    chunk_index = 0
    base_name = Path(filename).stem
    
    while start < content_len:
        end = min(start + chunk_size, content_len)
        text = content[start:end]
        
        # Only output non-empty/non-whitespace chunks
        if text.strip():
            chunk_id = f"{base_name}_{chunk_index}"
            chunks.append({
                "chunk_id": chunk_id,
                "source": filename,
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end,
                "text": text
            })
            chunk_index += 1
            
        if end == content_len:
            break
            
        start += chunk_size - overlap
        
    return chunks

def chunk_by_sentence(document: dict, chunk_size: int = 500) -> list[dict]:
    """
    Splits the content of a document by sentence boundaries, grouping them
    into chunks up to a maximum character size.
    
    Args:
        document (dict): Document dictionary containing content and metadata.
        chunk_size (int): Maximum character size for each chunk.
        
    Returns:
        list[dict]: List of chunk dictionaries with metadata.
    """
    content = document.get("content", "")
    filename = document.get("filename", "")
    base_name = Path(filename).stem
    
    # Split sentences using punctuation (.!? followed by whitespace or end of string)
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    chunks = []
    chunk_index = 0
    current_sentences = []
    current_len = 0
    start_char = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_len = len(sentence)
        
        # If adding this sentence exceeds chunk_size, emit the current chunk first
        if current_sentences and current_len + 1 + sentence_len > chunk_size:
            chunk_text = " ".join(current_sentences)
            pos = content.find(chunk_text, start_char)
            if pos != -1:
                start_char = pos
            end_char = start_char + len(chunk_text)
            
            chunks.append({
                "chunk_id": f"{base_name}_{chunk_index}",
                "source": filename,
                "chunk_index": chunk_index,
                "start_char": start_char,
                "end_char": end_char,
                "text": chunk_text
            })
            chunk_index += 1
            start_char = end_char
            
            current_sentences = [sentence]
            current_len = sentence_len
        else:
            current_sentences.append(sentence)
            current_len += (1 if current_len > 0 else 0) + sentence_len
            
    # Emit any remaining sentences in the final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        pos = content.find(chunk_text, start_char)
        if pos != -1:
            start_char = pos
        end_char = start_char + len(chunk_text)
        
        chunks.append({
            "chunk_id": f"{base_name}_{chunk_index}",
            "source": filename,
            "chunk_index": chunk_index,
            "start_char": start_char,
            "end_char": end_char,
            "text": chunk_text
        })
        
    return chunks
