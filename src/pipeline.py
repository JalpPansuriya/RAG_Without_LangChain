"""
Orchestration Pipeline Module for RAG from Scratch.
Coordinates document loading, chunking, embedding, index building, and querying.
"""

import os
from src.loader import load_folder
from src.chunker import chunk_fixed
from src.embedder import embed_texts
from src.vector_store import VectorStore
from src.retriever import retrieve
from src.generator import generate

def build_and_save_index(data_dir: str = "data/", index_path: str = "data/vector_store.json") -> None:
    """
    Offline Phase: Ingests documents, chunks them, generates embeddings,
    and saves the serialized vector store index to disk.
    
    Args:
        data_dir (str): Directory containing the text files.
        index_path (str): Filepath to save the serialized JSON index.
    """
    print(f"Loading documents from '{data_dir}'...")
    documents = load_folder(data_dir)
    print(f"Loaded {len(documents)} documents.")
    
    all_chunks = []
    for doc in documents:
        # Default parameters from PRD/AGENTS: chunk_size=500, overlap=50
        chunks = chunk_fixed(doc, chunk_size=500, overlap=50)
        all_chunks.extend(chunks)
        
    print(f"Split documents into {len(all_chunks)} chunks.")
    
    print("Generating embeddings for all chunks...")
    # Extract raw text from each chunk to embed
    texts_to_embed = [c["text"] for c in all_chunks]
    vectors = embed_texts(texts_to_embed)
    print(f"Generated {len(vectors)} vectors.")
    
    # Store in custom VectorStore
    vector_store = VectorStore()
    vector_store.add(all_chunks, vectors)
    
    # Save to disk
    vector_store.save(index_path)
    print(f"Index successfully built and saved to '{index_path}'.")

def execute_query(query: str, index_path: str = "data/vector_store.json", top_k: int = 5) -> tuple[str, list[dict]]:
    """
    Online Phase: Loads index from disk, retrieves top-K source chunks,
    and calls the LLM generator to produce a grounded response.
    
    Args:
        query (str): User question.
        index_path (str): Path to the serialized vector store.
        top_k (int): Number of source documents to retrieve.
        
    Returns:
        tuple[str, list[dict]]: A tuple containing the generated answer (str)
                                 and list of retrieved source chunk metadata (list of dicts).
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index file not found at '{index_path}'. "
            "Please build the index first using 'python main.py --build-index'."
        )
        
    # Load vector store from disk
    vector_store = VectorStore()
    vector_store.load(index_path)
    
    # Retrieve relevant chunks using retrieve coordination helper
    import src.embedder as embedder
    retrieved_chunks = retrieve(query, vector_store, embedder, top_k=top_k)
    
    # Generate response
    answer = generate(query, retrieved_chunks)
    
    return answer, retrieved_chunks
