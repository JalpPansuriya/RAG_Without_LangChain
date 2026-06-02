"""
In-Memory Vector Store Module for RAG from Scratch.
Implements vector storage, JSON serialization, and NumPy-based cosine similarity.
"""

import json
import os
import numpy as np
from pathlib import Path

class VectorStore:
    """
    An in-memory vector store that holds text chunks and their corresponding
    embedding vectors, implementing exact cosine similarity search.
    """
    def __init__(self):
        """
        Initializes an empty VectorStore.
        """
        self.chunks = []      # List of chunk dicts
        self.vectors = []     # List of numpy arrays

    def add(self, chunk: dict, vector: list[float]) -> None:
        """
        Stores a chunk alongside its embedding vector.
        Supports bulk adding if a list of chunks and a list of vectors are passed.
        
        Args:
            chunk (dict or list[dict]): A single chunk dict or a list of chunk dicts.
            vector (list[float] or list[list[float]]): A single embedding vector or a list of vectors.
        """
        if isinstance(chunk, list) and isinstance(vector, list):
            for c, v in zip(chunk, vector):
                self.chunks.append(c)
                self.vectors.append(np.array(v, dtype=np.float32))
        else:
            self.chunks.append(chunk)
            self.vectors.append(np.array(vector, dtype=np.float32))

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        """
        Computes cosine similarity between the query vector and all stored vectors
        using raw NumPy operations, returning the top-K chunks sorted by score.
        
        Args:
            query_vector (list[float]): The 1536-dimensional embedding of the query.
            top_k (int): Number of top results to return.
            
        Returns:
            list[dict]: List of results matching both nested "chunk" and flat key structures.
        """
        if not self.vectors:
            return []

        # Convert query vector to a numpy array for vector operations
        q_arr = np.array(query_vector, dtype=np.float32)
        
        # Convert all stored vectors to a single 2D numpy matrix for batch calculation
        vecs_matrix = np.array(self.vectors, dtype=np.float32)
        
        # Calculate dot products between the query vector and all stored vectors
        # Formula: A . B
        dot_products = np.dot(vecs_matrix, q_arr)
        
        # Calculate the Euclidean norm (magnitude) of the query vector
        # Formula: ||q|| = sqrt(sum(q_i^2))
        query_norm = np.linalg.norm(q_arr)
        
        # Calculate the Euclidean norm (magnitude) of each stored vector along the rows
        # Formula: ||v_j|| = sqrt(sum(v_ji^2)) for each document vector j
        vector_norms = np.linalg.norm(vecs_matrix, axis=1)
        
        # Multiply query norm by document norms to get the denominator
        # We add 1e-9 (epsilon) to avoid division by zero if a norm is zero
        denominators = vector_norms * query_norm + 1e-9
        
        # Compute cosine similarities: dot product divided by norm product
        # Formula: cos(theta) = (A . B) / (||A|| * ||B||)
        similarities = dot_products / denominators
        
        # Get sorting indices in descending order of similarity score
        indices = np.argsort(similarities)[::-1]
        
        results = []
        # Return top-K matches
        for idx in indices[:top_k]:
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            
            # Form a dict that satisfies both nested and flat output structures
            res = {
                "chunk": chunk,
                "score": score
            }
            # Copy all fields of the chunk to the top level of res
            for key, val in chunk.items():
                res[key] = val
                
            results.append(res)
            
        return results

    def save(self, filepath: str) -> None:
        """
        Serializes and saves the vector store database to a JSON file.
        
        Args:
            filepath (str): Destination path for the JSON file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy vectors back to lists of floats for JSON serialization
        serialized_vectors = [v.tolist() for v in self.vectors]
        
        data = {
            "chunks": self.chunks,
            "vectors": serialized_vectors
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Loads and deserializes the vector store database from a JSON file.
        
        Args:
            filepath (str): Path to the JSON database.
            
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Vector store file not found: {filepath}")
            
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        self.chunks = data.get("chunks", [])
        self.vectors = [np.array(v, dtype=np.float32) for v in data.get("vectors", [])]
