"""
Query Retriever Module for RAG from Scratch.
Coordinates embedding of queries and retrieving corresponding text chunks.
"""

def retrieve(query: str, vector_store, embedder, top_k: int = 5) -> list[dict]:
    """
    Retrieves the top-K relevant chunks for a user query.
    Embeds the query, searches the vector store, and returns matches.
    
    Args:
        query (str): The user's question.
        vector_store (VectorStore): The initialized VectorStore instance.
        embedder: The module or class instance that exposes embed_query or embed_texts.
        top_k (int): Number of chunks to retrieve.
        
    Returns:
        list[dict]: Top-K matching chunks, each containing text, source filename, and similarity score.
    """
    # Guard against empty vector store
    if not vector_store.chunks:
        return []
        
    # Get the embedding for the query
    if hasattr(embedder, "embed_query"):
        query_vector = embedder.embed_query(query)
    elif hasattr(embedder, "embed_texts"):
        query_vector = embedder.embed_texts([query])[0]
    else:
        query_vector = embedder([query])[0]
        
    # Search the vector store
    search_results = vector_store.search(query_vector, top_k=top_k)
    
    # Map vector store search results to the retriever output format
    retrieved_chunks = []
    for res in search_results:
        chunk = res.get("chunk", res)
        retrieved_chunks.append({
            "text": chunk.get("text", ""),
            "source": chunk.get("source", ""),
            "score": res.get("score", 0.0)
        })
        
    return retrieved_chunks
