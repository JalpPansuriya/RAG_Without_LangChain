import pytest
import os
from src.loader import load_txt, load, load_folder

def test_load_folder_success():
    """
    Verifies that load_folder successfully loads all 5 text documents
    from the data directory and that each document contains the required keys
    and non-empty content.
    """
    # The actual data directory has the 5 files
    docs = load_folder("data")
    assert len(docs) == 5
    
    expected_keys = {"filename", "filepath", "content", "char_count", "word_count"}
    
    for doc in docs:
        # Check all keys exist
        assert expected_keys.issubset(doc.keys())
        # Check content is non-empty
        assert len(doc["content"].strip()) > 0
        # Check character and word counts are correct
        assert doc["char_count"] == len(doc["content"])
        assert doc["word_count"] == len(doc["content"].split())
        # Check filename is not a full path but just the base name
        assert not os.path.isabs(doc["filename"])
        assert "/" not in doc["filename"]
        assert "\\" not in doc["filename"]

def test_load_unsupported_file_type(tmp_path):
    """
    Verifies that calling load on an unsupported file type (like PDF)
    raises a ValueError.
    """
    pdf_file = tmp_path / "test_doc.pdf"
    pdf_file.write_text("dummy PDF content")
    
    with pytest.raises(ValueError) as excinfo:
        load(str(pdf_file))
    assert "Unsupported file type" in str(excinfo.value)

def test_load_folder_missing_directory():
    """
    Verifies that attempting to load a non-existent folder raises a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_folder("non_existent_folder_path_1234")
