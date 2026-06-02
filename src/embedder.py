"""
Embedding Generator Module for RAG from Scratch.
Uses the sentence-transformers library with model all-MiniLM-L6-v2 running fully locally.
"""

import logging

# Logger configuration
logger = logging.getLogger(__name__)

# Cached model instance
_model = None

def get_model():
    """
    Lazily instantiates and caches the SentenceTransformer model.
    """
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers model 'all-MiniLM-L6-v2'...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generates embedding vectors for a list of texts using the local sentence-transformers
    model 'all-MiniLM-L6-v2'.
    
    Args:
        texts (list[str]): List of text strings to embed.
        
    Returns:
        list[list[float]]: List of 384-dimensional embedding vectors.
    """
    if not texts:
        return []
        
    model = get_model()
    # encode returns a numpy array of vectors, convert each to a list of floats
    embeddings = model.encode(texts)
    return [emb.tolist() for emb in embeddings]

def embed_query(query: str) -> list[float]:
    """
    Generates an embedding vector for a single query text.
    
    Args:
        query (str): Query string.
        
    Returns:
        list[float]: Embedding vector.
    """
    embeddings = embed_texts([query])
    return embeddings[0]
