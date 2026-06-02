# Custom RAG from Scratch (No LangChain)

A modular, lightweight Retrieval-Augmented Generation (RAG) pipeline built completely from scratch using standard Python libraries, NumPy, and the raw OpenAI API. Designed to demonstrate the inner workings of document loading, text chunking, vector similarity, retrieval ranking, and grounding without RAG frameworks.

---

## Folder Structure

```
rag-from-scratch/
│
├── data/
│   ├── artificial_intelligence.txt
│   ├── deep_learning.txt
│   ├── history_of_ai.txt
│   ├── machine_learning.txt
│   └── natural_language_processing.txt
│
├── src/
│   ├── __init__.py
│   ├── loader.py           # Document Loader
│   ├── chunker.py          # Text Chunker
│   ├── embedder.py         # Embedding Generator
│   ├── vector_store.py     # Custom In-Memory Vector Store
│   ├── retriever.py        # Query Retriever
│   └── generator.py        # Answer Generator
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   └── test_generator.py
│
├── main.py                 # CLI entrypoint
├── requirements.txt
├── .env.example
├── PRD.md
├── AGENTS (3).md
└── LEARNINGS.md            # Reflections & math derivations
```

---

## Installation & Setup

1. **Clone or navigate** to the project directory:
   ```bash
   cd "c:\Users\rudra\Desktop\Codexial\RAG without Langchain"
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Copy `.env.example` to `.env` and fill in your API Key:
   ```bash
   cp .env.example .env
   ```
   Open `.env` in a text editor and add:
   ```env
   GROQ_API_KEY=your-groq-api-key-here
   ```

---

## How to Run

The pipeline operates in two phases: offline index building and online querying.

### 1. Build the Vector Store Index
Before querying, load the 5 Wikipedia text files from the `data/` directory, chunk them, compute embeddings, and serialize the index to disk:
```bash
python main.py --build-index
```
This generates the serialized index file at `data/vector_store.json`.

### 2. Run a Single Query
Query the built index to get a grounded answer along with similarity scores and sources used:
```bash
python main.py --query "What is the Turing test?"
```

### 3. Run in Interactive Mode (REPL)
Launch a continuous chat session to query the pipeline repeatedly:
```bash
python main.py --interactive
```
Type `exit` or `quit` to exit interactive mode.

---

## Running Tests

All unit tests mock external APIs (OpenAI) and run locally and fast. Run them with `pytest`:
```bash
python -m pytest
```
For verbose output:
```bash
python -m pytest -v
```
