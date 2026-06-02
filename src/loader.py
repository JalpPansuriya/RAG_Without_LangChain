"""
Document Loader Module for RAG from Scratch.
Responsible for reading text files and attaching appropriate metadata.
"""

import os
from pathlib import Path

def load_txt(filepath: str) -> str:
    """
    Reads a text file and returns its content as a raw string.
    Supports UTF-8 encoding with a fallback to latin-1 for compatibility.
    
    Args:
        filepath (str): Path to the text file to read.
        
    Returns:
        str: The content of the file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"File not found at: {filepath}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback encoding
        with open(path, "r", encoding="latin-1") as f:
            return f.read()

def load(filepath: str) -> dict:
    """
    Loads a single document from a file path, detects file type by extension,
    and returns a dictionary containing its content and metadata.
    
    Args:
        filepath (str): Path to the file.
        
    Returns:
        dict: Document object with keys: filename, filepath, content, char_count, word_count.
        
    Raises:
        ValueError: If the file type is unsupported (only .txt is supported).
        FileNotFoundError: If the file does not exist.
    """
    path = Path(filepath)
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Unsupported file type: {path.suffix}. Only .txt is supported.")
        
    content = load_txt(str(path))
    return {
        "filename": path.name,
        "filepath": str(path.as_posix()),
        "content": content,
        "char_count": len(content),
        "word_count": len(content.split())
    }

def load_folder(data_dir: str = "data/") -> list[dict]:
    """
    Walks the specified directory, discovers all .txt files dynamically,
    loads them, and returns a list of document dictionaries.
    
    Args:
        data_dir (str): Path to the directory containing documents.
        
    Returns:
        list[dict]: List of document dictionaries, each representing a loaded file.
        
    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    dir_path = Path(data_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
        
    documents = []
    # Discover files dynamically
    for filename in sorted(os.listdir(dir_path)):
        filepath = dir_path / filename
        if filepath.is_file() and filepath.suffix.lower() == ".txt":
            documents.append(load(str(filepath)))
            
    return documents
