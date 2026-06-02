# Product Requirements Document (PRD)
## Project: RAG From Scratch — No LangChain

**Version:** 1.0  
**Status:** Draft  
**Author:** You  
**Last Updated:** 2026-06-02

---

## 1. Overview

### 1.1 Problem Statement

Most developers learning Retrieval-Augmented Generation (RAG) rely on frameworks like LangChain or LlamaIndex as black boxes. This makes it hard to understand what's actually happening under the hood — chunking strategies, embedding math, vector similarity, and prompt construction.

### 1.2 Goal

Build a fully functional RAG pipeline from scratch using only Python primitives and low-level libraries, so the builder deeply understands every layer of how RAG works.

### 1.3 Non-Goals

- This is **not** a production-ready system.
- This is **not** a replacement for LangChain.
- This project will **not** use LangChain, LlamaIndex, or any high-level RAG framework.

---

## 2. Target User

**Primary user:** You — a developer learning RAG internals.

**Secondary users:** Anyone who clones this repo to learn RAG from first principles.

---

## 3. Success Criteria

| Metric | Target |
|--------|--------|
| Can ingest a PDF/text document | ✅ Required |
| Can chunk and embed documents without a framework | ✅ Required |
| Can retrieve top-K relevant chunks for a query | ✅ Required |
| Can generate a grounded answer using an LLM | ✅ Required |
| Each component is independently testable | ✅ Required |
| Code is readable and well-commented for learning | ✅ Required |

---

## 4. System Architecture

The pipeline is composed of five independent stages:

```
[Document Loader]
      ↓
[Text Chunker]
      ↓
[Embedder]
      ↓
[Vector Store]
      ↓
[Retriever + Generator]
```

Each stage is a standalone module with a clear input/output contract.

---

## 5. Functional Requirements

### 5.1 Document Loader

- Accept plain text files (.txt) and PDF files (.pdf)
- Return a list of raw text strings (one per page or document)
- No external document parsing frameworks (use `pdfplumber` or `pypdf` only for PDF byte-reading)

### 5.2 Text Chunker

- Split raw text into overlapping chunks
- Configurable `chunk_size` (default: 500 tokens) and `chunk_overlap` (default: 50 tokens)
- Return a list of chunk strings with metadata (source file, chunk index)
- Implement at least two strategies: fixed-size and sentence-boundary splitting

### 5.3 Embedder

- Convert text chunks into dense vector embeddings
- Use the OpenAI Embeddings API (`text-embedding-3-small`) or a local model via `sentence-transformers`
- Return a list of (chunk, vector) pairs
- No wrapper libraries — make raw HTTP calls or use the minimal SDK

### 5.4 Vector Store

- Store (chunk text, embedding vector, metadata) in memory as a list
- Implement cosine similarity search from scratch (no FAISS, no Chroma, no Pinecone)
- Support `top_k` retrieval given a query embedding
- Optional: persist to a JSON file on disk for reuse

### 5.5 Retriever

- Accept a user query string
- Embed the query using the same Embedder
- Return top-K chunks from the Vector Store ranked by cosine similarity score

### 5.6 Generator

- Accept the user query + retrieved context chunks
- Construct a prompt that instructs the LLM to answer only from the context
- Call the LLM API (OpenAI `gpt-4o-mini` or Anthropic Claude)
- Return the generated answer as a string

### 5.7 Pipeline Orchestrator

- Wire all stages together in a single `rag_pipeline.py` entry point
- Accept a document path and a user query as inputs
- Print the final answer plus the source chunks used

---

## 6. Non-Functional Requirements

- **Readability:** Every module must have a docstring explaining what it does and why.
- **Modularity:** Each stage can be imported and tested independently.
- **No hidden magic:** No abstractions that hide the math or the HTTP calls.
- **Learning comments:** Key algorithmic steps (e.g., cosine similarity formula) should have inline comments explaining the math.

---

## 7. Allowed Libraries

| Purpose | Allowed |
|---------|---------|
| PDF reading | `pdfplumber` or `pypdf` |
| Embeddings | `openai` SDK or raw `httpx` calls |
| LLM calls | `openai` SDK or `anthropic` SDK |
| Vector math | `numpy` only |
| Testing | `pytest` |
| Environment variables | `python-dotenv` |

**Banned:** `langchain`, `llama_index`, `haystack`, `chromadb`, `faiss`, `pinecone-client`

---

## 8. Project Structure

```
rag-from-scratch/
├── README.md
├── PRD.md
├── AGENTS.md
├── .env.example
├── requirements.txt
├── data/                        # Cleaned Wikipedia articles (5 files, ~48k words)
│   ├── artificial_intelligence.txt
│   ├── deep_learning.txt
│   ├── history_of_ai.txt
│   ├── machine_learning.txt
│   └── natural_language_processing.txt
├── src/
│   ├── loader.py                # Document Loader
│   ├── chunker.py               # Text Chunker
│   ├── embedder.py              # Embedder
│   ├── vector_store.py          # In-memory Vector Store
│   ├── retriever.py             # Retriever
│   ├── generator.py             # LLM Generator
│   └── pipeline.py              # Orchestrator
└── tests/
    ├── test_loader.py
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_vector_store.py
    ├── test_retriever.py
    └── test_generator.py
```

---

## 9. Phased Execution Plan

Each phase has a clear goal, a definition of done, and a gate check before the next phase starts. Never move to the next phase unless the gate check passes.

---

### Phase 0 — Project Scaffold
**Goal:** Get the repo, folder structure, and environment ready before writing any logic.

**Tasks:**
- Create the folder structure exactly as defined in Section 8
- Create `.env.example` with placeholder keys (`GROQ_API_KEY=`)
- Create `requirements.txt` with allowed libraries only
- Place the 5 cleaned Wikipedia `.txt` files into `data/` — they are already prepared:
  - `artificial_intelligence.txt`
  - `deep_learning.txt`
  - `history_of_ai.txt`
  - `machine_learning.txt`
  - `natural_language_processing.txt`
- Confirm `pytest` runs (zero tests, zero failures)

**Definition of Done:**
- Folder exists, `pytest` exits cleanly, `.env` is set up, `data/` has all 5 files

**Gate Check:** ✅ `pytest` passes with 0 errors → proceed to Phase 1

---

### Phase 1 — Document Loader
**Goal:** Be able to read all 5 data files and get back a clean list of text strings with source metadata.

**Agent:** `loader-agent`  
**Files:** `src/loader.py`, `tests/test_loader.py`

**Tasks:**
- Implement `load_txt(filepath: str) -> list[str]` for plain text files
- Implement `load_pdf(filepath: str) -> list[str]` using `pdfplumber` or `pypdf`
- Implement `load(filepath: str) -> list[dict]` dispatcher that detects file type by extension
  - Each returned dict: `{ "text": str, "source": str }` where source is the filename
- Implement `load_folder(folder_path: str) -> list[dict]` to load all `.txt` files in a folder
  - This is needed since we have 5 files in `data/`
- Write tests: load one of the 5 txt files, assert output is a non-empty list of dicts with `text` and `source` keys

**What You Should Be Able To Do After This Phase:**
- Call `load_folder("data/")` and get back all 5 files as a flat list of text blocks with source labels
- Understand how file I/O works without any framework magic

**Definition of Done:**
- `load()` works for `.txt` files
- `load_folder()` loads all files in `data/` correctly
- Each returned chunk has a `source` field with the filename
- At least 3 passing tests in `test_loader.py`

**Gate Check:** ✅ `pytest tests/test_loader.py` passes → proceed to Phase 2

---

### Phase 2 — Text Chunker
**Goal:** Break raw text into overlapping chunks that are the right size for embedding.

**Agent:** `chunker-agent`  
**Files:** `src/chunker.py`, `tests/test_chunker.py`

**Tasks:**
- Implement `chunk_fixed()` — fixed character-size chunks with overlap using a sliding window
- Implement `chunk_by_sentence()` — split on `.`, `?`, `!` then group into size-limited chunks
- Each chunk must be a dict: `{ "text": str, "chunk_index": int, "start_char": int }`
- Write tests: pass a known string, assert chunk count and overlap are correct

**What You Should Learn After This Phase:**
- Why chunking strategy matters (too big = irrelevant context; too small = loss of meaning)
- How overlap helps preserve context across chunk boundaries

**Definition of Done:**
- Both chunking strategies work
- No zero-length chunks produced
- At least 3 passing tests in `test_chunker.py`

**Gate Check:** ✅ `pytest tests/test_chunker.py` passes → proceed to Phase 3

---

### Phase 3 — Embedder
**Goal:** Turn text chunks into dense vectors using a real embedding model.

**Agent:** `embedder-agent`  
**Files:** `src/embedder.py`, `tests/test_embedder.py`

**Tasks:**
- Implement `embed_texts(texts: list[str]) -> list[list[float]]` calling OpenAI embeddings API
- Implement `embed_query(query: str) -> list[float]` for single query embedding
- Load API key from `.env` via `python-dotenv`
- Write tests: mock the API call, assert output shape (list of lists of floats)

**What You Should Learn After This Phase:**
- What an embedding vector actually is (a list of ~1536 floats)
- Why the same model must be used for both documents and queries
- How to make a raw API call without a framework wrapper

**Definition of Done:**
- `embed_texts()` returns a list of float vectors, one per input
- Tests mock the API and pass without a real API call
- API key is never hardcoded

**Gate Check:** ✅ `pytest tests/test_embedder.py` passes → proceed to Phase 4

---

### Phase 4 — Vector Store
**Goal:** Store embeddings and retrieve the most similar ones using cosine similarity — written by hand.

**Agent:** `vector-store-agent`  
**Files:** `src/vector_store.py`, `tests/test_vector_store.py`

**Tasks:**
- Implement `VectorStore` class with `add()`, `search()`, `save()`, `load()` methods
- Implement cosine similarity from scratch using `numpy` — formula must be commented
- `search()` returns top-K chunks sorted by score descending, each chunk dict gets a `score` field added
- `save()` serializes to JSON; `load()` restores from JSON

**What You Should Learn After This Phase:**
- The math behind cosine similarity: `dot(A, B) / (norm(A) * norm(B))`
- Why cosine similarity works better than Euclidean distance for high-dimensional vectors
- How a real vector DB works at its core (this is exactly what FAISS does, just slower)

**Definition of Done:**
- `search()` returns correct top-K results for a known set of vectors
- Cosine similarity formula is present and commented in the code
- `save/load` round-trip preserves all data
- At least 4 passing tests

**Gate Check:** ✅ `pytest tests/test_vector_store.py` passes → proceed to Phase 5

---

### Phase 5 — Retriever
**Goal:** Wire the Embedder and Vector Store together into a single retrieval call.

**Agent:** `retriever-agent`  
**Files:** `src/retriever.py`, `tests/test_retriever.py`

**Tasks:**
- Implement `retrieve(query, vector_store, embedder, top_k=5) -> list[dict]`
- It should embed the query, call `vector_store.search()`, and return results
- Write tests using mock embedder and a pre-filled vector store

**What You Should Learn After This Phase:**
- The difference between the indexing pipeline (offline) and the retrieval pipeline (online, per query)
- Why retrieval is fast (one embed call + dot products) vs. indexing (embed every chunk)

**Definition of Done:**
- `retrieve()` returns top-K chunks for a query
- Empty vector store returns empty list (no crash)
- At least 2 passing tests

**Gate Check:** ✅ `pytest tests/test_retriever.py` passes → proceed to Phase 6

---

### Phase 6 — Generator
**Goal:** Take retrieved chunks and a query, build a grounded prompt, and get an answer from the LLM.

**Agent:** `generator-agent`  
**Files:** `src/generator.py`, `tests/test_generator.py`

**Tasks:**
- Implement `generate(query: str, context_chunks: list[dict]) -> str`
- Build the prompt as a plain Python f-string — no template libraries
- System message must instruct the LLM: "Answer only from the context. If the answer is not in the context, say 'I don't know.'"
- Call the LLM API (OpenAI or Anthropic)
- Write tests: mock the LLM call, assert the prompt contains the query and chunk text

**What You Should Learn After This Phase:**
- How prompt construction directly affects answer quality
- Why grounding ("answer only from context") reduces hallucination
- The structure of a chat API call: system message + user message

**Definition of Done:**
- `generate()` returns a string answer
- Empty context chunks returns "I don't know" gracefully
- Prompt is constructed as a plain f-string, visible in the code
- At least 3 passing tests with mocked LLM

**Gate Check:** ✅ `pytest tests/test_generator.py` passes → proceed to Phase 7

---

### Phase 7 — Pipeline Orchestrator
**Goal:** Connect all six stages into a single working end-to-end RAG pipeline.

**Agent:** `pipeline-agent`  
**Files:** `src/pipeline.py`, `README.md`

**Tasks:**
- Implement `run_pipeline(document_path, query, top_k=5) -> dict`
- It calls: Loader → Chunker → Embedder → VectorStore.add() → Retriever → Generator
- Returns `{ "answer": str, "sources": list[dict] }`
- Add a CLI entry point so the user can run: `python src/pipeline.py --doc data/sample.txt --query "..."`
- Print the answer and source chunks to the terminal
- Update `README.md` with setup and usage instructions

**What You Should Be Able To Do After This Phase:**
- Ask any question about your document and get a grounded answer
- See exactly which chunks were used to form the answer

**Definition of Done:**
- End-to-end pipeline runs on `data/sample.txt` with a real query
- CLI works as documented
- README is complete with install + run instructions

**Gate Check:** ✅ Full pipeline produces a non-empty answer for a real query

---

### Phase 8 — Polish & Reflection
**Goal:** Clean up, document learnings, and make the repo shareable.

**Tasks:**
- Ensure every function has a docstring
- Ensure every math formula has an inline comment
- Run `pytest` on all test files — all must pass
- Write a `LEARNINGS.md` reflecting on what each phase taught you
- Optional: add a `notebooks/demo.ipynb` showing the pipeline step by step

**Definition of Done:**
- `pytest` passes across all test files
- `LEARNINGS.md` exists with at least one insight per phase
- Repo is clean enough to share or reference later

---

### Phase Summary Table

| Phase | Focus | Key Output | Gate |
|-------|-------|------------|------|
| 0 | Scaffold | Folder structure + env | `pytest` runs |
| 1 | Loader | `load()` function | `test_loader` passes |
| 2 | Chunker | `chunk_fixed()` + `chunk_by_sentence()` | `test_chunker` passes |
| 3 | Embedder | `embed_texts()` + `embed_query()` | `test_embedder` passes |
| 4 | Vector Store | `VectorStore` with cosine similarity | `test_vector_store` passes |
| 5 | Retriever | `retrieve()` | `test_retriever` passes |
| 6 | Generator | `generate()` with grounded prompt | `test_generator` passes |
| 7 | Orchestrator | End-to-end CLI pipeline | Real query returns answer |
| 8 | Polish | Docs + reflection | All tests green |

---

## 10. Resolved Decisions

These were previously open questions — now locked in based on project setup:

| Decision | Choice | Reason |
|----------|--------|--------|
| Embedding model | OpenAI `text-embedding-3-small` | API-based, no local setup needed, 1536-dim vectors |
| LLM | OpenAI `gpt-4o-mini` | Fast, cheap, great for learning |
| Data source | 5 cleaned Wikipedia articles | Already prepared and cleaned — see `data/` folder |
| Vector store persistence | Phase 4 (JSON on disk) | Avoids re-embedding on every run |
| Multi-file support | Yes — loader accepts a folder or list of files | Needed since we have 5 data files |

---

## 11. Out of Scope (Future Ideas)

- Hybrid search (BM25 + dense retrieval)
- Re-ranking with a cross-encoder
- Multi-document RAG
- Streaming LLM responses
- A simple web UI
