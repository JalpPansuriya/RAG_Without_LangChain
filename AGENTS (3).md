# AGENTS.md — RAG From Scratch (No LangChain)
> Harness Engineering Style · Version 2.0
> Updated to reflect: 5-file corpus, 9-phase execution plan, multi-file loader

---

## PURPOSE OF THIS FILE

This file defines every AI agent used in this project.
Paste this file into any AI assistant (Claude, Cursor, Copilot, GPT-4) to give it a named role, a strict scope, and clear guardrails before you start a phase.

**How to invoke an agent:**
> "You are the `chunker-agent`. Follow AGENTS.md rules. Here is my task: ..."

---

## GLOBAL RULES — ALL AGENTS MUST FOLLOW

These rules are non-negotiable and apply to every agent in this file:

| Rule | Description |
|------|-------------|
| **NO_LANGCHAIN** | Never import or suggest LangChain, LlamaIndex, Haystack, or any RAG framework |
| **NO_MAGIC** | Never use a library that hides the math. If it does cosine similarity for you, it's banned |
| **STDLIB_FIRST** | Prefer Python standard library. Add a dependency only when there is no stdlib alternative |
| **ALLOWED_LIBS** | `numpy`, `openai`, `tiktoken`, `PyPDF2` or `pdfplumber`, `pytest`, `python-dotenv` |
| **BANNED_LIBS** | `langchain`, `llama_index`, `haystack`, `chromadb`, `faiss`, `pinecone`, `weaviate`, `sentence_transformers` |
| **DOCSTRING_ALL** | Every function must have a docstring: what it does, args, returns |
| **COMMENT_MATH** | Every line doing math (dot product, norm, cosine) must have an inline comment explaining it |
| **NO_GLOBAL_STATE** | No global variables. Pass state explicitly through function arguments |
| **PURE_FUNCTIONS** | Functions must not have side effects unless they are explicitly I/O agents (loader, writer) |
| **TEST_GATE** | Every phase ends with a passing pytest. Do not move to the next phase until the gate passes |
| **PHASE_SCOPE** | Each agent works only within its designated phase. It does not reach into another agent's domain |

---

## CORPUS — DATA FILES IN SCOPE

All agents that deal with data must operate on these 5 files located in `data/`:

| File | Topic | Approx. Words |
|------|-------|---------------|
| `artificial_intelligence.txt` | Broad AI — goals, techniques, ethics, history | ~13,000 |
| `deep_learning.txt` | Neural nets, CNNs, RNNs, adversarial examples | ~8,500 |
| `history_of_ai.txt` | AI winters, milestones, key figures | ~12,000 |
| `machine_learning.txt` | Supervised, unsupervised, hardware | ~9,500 |
| `natural_language_processing.txt` | NLP history, transformers, linguistics | ~4,500 |

**Total corpus: ~48,000 words of clean AI content.**
All files are pre-cleaned — no Wikipedia nav bars, no references sections, no citation markers.

---

## AGENT REGISTRY

---

### `loader-agent`
**Phase:** 1 — Document Loader
**File:** `src/loader.py`

**Trigger:** User says "build the loader", "load documents", "Phase 1", or asks how to read text files into memory.

**Scope:**
- Read all `.txt` files from the `data/` directory
- Return a list of document dicts — one per file
- Handle file encoding gracefully (UTF-8 with fallback)

**Responsibilities:**
- Walk the `data/` directory and discover all `.txt` files automatically
- For each file: read content, attach metadata (filename, filepath, char_count, word_count)
- Return a clean list of dicts — do NOT return raw strings

**Input:**
```
data_dir: str  →  path to the data folder (default: "data/")
```

**Output:**
```python
[
  {
    "filename": "artificial_intelligence.txt",
    "filepath": "data/artificial_intelligence.txt",
    "content": "Artificial intelligence (AI) is ...",
    "char_count": 90000,
    "word_count": 13200
  },
  # ... one dict per file
]
```

**Guardrails:**
- MUST support loading all 5 corpus files in one call
- MUST NOT hardcode filenames — discover them dynamically with `os.listdir` or `pathlib.glob`
- MUST attach metadata to every document dict
- MUST NOT do any chunking, cleaning, or embedding — that is not this agent's job
- MUST handle missing `data/` directory with a clear `FileNotFoundError` message

**Gate Check:**
```bash
pytest tests/test_loader.py -v
```
Tests must verify: 5 docs loaded, each has all 4 keys, no doc has empty content.

---

### `chunker-agent`
**Phase:** 2 — Text Chunker
**File:** `src/chunker.py`

**Trigger:** User says "build the chunker", "split into chunks", "Phase 2", or asks about chunking strategy.

**Scope:**
- Accept a single document dict (output from loader-agent)
- Split its `content` field into overlapping text chunks
- Return a list of chunk dicts with positional metadata

**Responsibilities:**
- Implement fixed-size character chunking with configurable `chunk_size` and `overlap`
- Attach metadata to every chunk: source filename, chunk index, start/end character positions
- Default values: `chunk_size=500`, `overlap=50`

**Input:**
```python
document: dict       # one document dict from loader-agent
chunk_size: int      # number of characters per chunk (default: 500)
overlap: int         # number of overlapping characters between chunks (default: 50)
```

**Output:**
```python
[
  {
    "chunk_id": "artificial_intelligence_0",
    "source": "artificial_intelligence.txt",
    "chunk_index": 0,
    "start_char": 0,
    "end_char": 500,
    "text": "Artificial intelligence (AI) is the capability of..."
  },
  # ... one dict per chunk
]
```

**Guardrails:**
- MUST implement overlap manually using string slicing — no `textwrap`, no `nltk`
- MUST NOT produce empty chunks (filter out any chunk where `text.strip() == ""`)
- MUST preserve the source filename in every chunk
- MUST NOT call the embedder or any external API
- MUST work on all 5 corpus files when called in a loop from the pipeline

**Gate Check:**
```bash
pytest tests/test_chunker.py -v
```
Tests must verify: overlap is working, no empty chunks, chunk_id is unique, metadata is attached.

---

### `embedder-agent`
**Phase:** 3 — Embedding Generator
**File:** `src/embedder.py`

**Trigger:** User says "build the embedder", "generate embeddings", "Phase 3", or asks about the OpenAI embeddings API.

**Scope:**
- Accept a list of text strings
- Call OpenAI `text-embedding-3-small` and return vectors
- Handle batching and rate limits

**Responsibilities:**
- Load the OpenAI API key from `.env` using `python-dotenv`
- Call `client.embeddings.create()` with `model="text-embedding-3-small"`
- Return a list of float vectors in the same order as the input texts
- Implement simple retry logic (3 attempts, exponential backoff) for rate limit errors

**Input:**
```python
texts: list[str]   # list of text strings to embed
```

**Output:**
```python
[
  [0.021, -0.043, 0.118, ...],   # 1536-dimensional vector for texts[0]
  [0.003,  0.091, -0.027, ...],  # 1536-dimensional vector for texts[1]
  # ... one vector per input text
]
```

**Guardrails:**
- MUST load API key from `.env` — NEVER hardcode it
- MUST return vectors in the same order as input texts
- MUST NOT use `sentence_transformers` or any local embedding model
- MUST NOT embed more than 100 texts in a single API call (batch if needed)
- MUST print a warning if the API key is missing, not raise a silent error
- Tests MUST mock the OpenAI API — no real API calls during `pytest`

**Gate Check:**
```bash
pytest tests/test_embedder.py -v
```
Tests must use `unittest.mock` to mock `openai.OpenAI`. No real API calls in tests.

---

### `vector-store-agent`
**Phase:** 4 — In-Memory Vector Store
**File:** `src/vector_store.py`

**Trigger:** User says "build the vector store", "store embeddings", "Phase 4", or asks about storing and searching vectors.

**Scope:**
- Accept chunks + their embedding vectors
- Store them in memory as a list of dicts
- Implement cosine similarity search from scratch using only `numpy`

**Responsibilities:**
- `add(chunk, vector)` — store a chunk dict alongside its embedding vector
- `search(query_vector, top_k)` — compute cosine similarity between the query and all stored vectors, return top-K chunks sorted by score
- Implement cosine similarity manually: `dot(a, b) / (norm(a) * norm(b))`

**Input for `search`:**
```python
query_vector: list[float]   # 1536-dim embedding of the user's query
top_k: int                  # number of results to return (default: 5)
```

**Output of `search`:**
```python
[
  {
    "chunk": { ...chunk dict... },
    "score": 0.87   # cosine similarity score
  },
  # ... top_k results, sorted descending by score
]
```

**Guardrails:**
- MUST implement cosine similarity using only `numpy` — no `sklearn`, no `faiss`, no `scipy`
- MUST comment every line of the cosine similarity math
- MUST NOT use a vector database (ChromaDB, Pinecone, Weaviate, FAISS are all banned)
- MUST store vectors as `numpy` arrays internally for fast dot product
- MUST return results sorted by score descending

**Gate Check:**
```bash
pytest tests/test_vector_store.py -v
```
Tests must verify: scores are between -1 and 1, top result is the most similar, order is correct.

---

### `retriever-agent`
**Phase:** 5 — Query Retriever
**File:** `src/retriever.py`

**Trigger:** User says "build the retriever", "retrieve chunks", "Phase 5", or asks how to connect a query to the vector store.

**Scope:**
- Accept a raw user query string
- Embed it using embedder-agent
- Search the vector store and return the top-K relevant chunks

**Responsibilities:**
- Call `embedder.embed([query])` to get the query vector
- Call `vector_store.search(query_vector, top_k)` to retrieve matches
- Return the top-K chunk texts + their scores

**Input:**
```python
query: str              # the user's question in plain English
vector_store: object    # an initialized VectorStore instance
top_k: int              # number of chunks to retrieve (default: 5)
```

**Output:**
```python
[
  {
    "text": "Deep learning is a subset of machine learning...",
    "source": "deep_learning.txt",
    "score": 0.91
  },
  # ... top_k results
]
```

**Guardrails:**
- MUST embed the query using the same model used to embed the corpus (`text-embedding-3-small`)
- MUST NOT re-embed the entire corpus on every query — the vector store is pre-built
- MUST pass `top_k` through to the vector store search
- MUST return source filename alongside each retrieved chunk
- Tests MUST mock the embedder — no real API calls during `pytest`

**Gate Check:**
```bash
pytest tests/test_retriever.py -v
```

---

### `generator-agent`
**Phase:** 6 — Answer Generator
**File:** `src/generator.py`

**Trigger:** User says "build the generator", "generate an answer", "Phase 6", or asks how to call the LLM with context.

**Scope:**
- Accept a user query + a list of retrieved context chunks
- Build a prompt that injects the context
- Call `gpt-4o-mini` and return the final answer

**Responsibilities:**
- Build a system prompt that instructs the LLM to answer ONLY from provided context
- Format retrieved chunks into a numbered context block
- Call `client.chat.completions.create()` with `model="gpt-4o-mini"`
- Return the answer string

**Prompt Template:**
```
System:
You are a helpful assistant. Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I don't know based on the provided context."
Do not make up information.

Context:
[1] {chunk_1_text}  (source: deep_learning.txt)
[2] {chunk_2_text}  (source: history_of_ai.txt)
...

User Question:
{query}
```

**Input:**
```python
query: str              # the user's original question
context_chunks: list    # list of retrieved chunk dicts from retriever-agent
```

**Output:**
```python
"Deep learning is a class of machine learning techniques that use neural networks..."
# a plain string answer from the LLM
```

**Guardrails:**
- MUST include source filenames in the context block sent to the LLM
- MUST instruct the LLM to only use provided context — no hallucination
- MUST NOT use `gpt-4` or `gpt-4-turbo` (too expensive for dev) — use `gpt-4o-mini`
- MUST keep the system prompt under 200 words
- Tests MUST mock the OpenAI chat completion — no real API calls during `pytest`

**Gate Check:**
```bash
pytest tests/test_generator.py -v
```

---

### `pipeline-agent`
**Phase:** 7 — End-to-End Pipeline
**File:** `src/pipeline.py` + `main.py`

**Trigger:** User says "wire everything together", "build the pipeline", "Phase 7", or asks for the final CLI tool.

**Scope:**
- Orchestrate all 5 agents in the correct order
- Expose a CLI interface so the user can ask questions from the terminal
- Handle the offline (index build) and online (query) phases separately

**Responsibilities:**

**Offline — Index Build (run once):**
```
loader → chunker → embedder → vector_store.add()
```
1. Load all 5 docs from `data/`
2. Chunk every doc
3. Embed all chunks in batches
4. Store all vectors in the VectorStore

**Online — Query (run per question):**
```
user_query → embedder → vector_store.search() → generator → print answer
```
1. Take user query from CLI input
2. Embed the query
3. Retrieve top-5 chunks
4. Generate answer with context
5. Print the answer + sources

**CLI Interface:**
```bash
python main.py --build-index        # builds the vector store from data/
python main.py --query "What is deep learning?"  # asks a question
python main.py --interactive        # REPL mode: keep asking questions
```

**Guardrails:**
- MUST separate index-build from query — do not re-embed the corpus on every query
- MUST print retrieved sources alongside the answer so the user can verify grounding
- MUST handle the case where `--query` is run before `--build-index` with a clear error
- MUST NOT hardcode any file paths — use `argparse` or `pathlib`
- The pipeline is a thin orchestrator — it calls the other agents, it does not reimplement their logic

**Gate Check:**
```bash
# Full end-to-end test
python main.py --build-index
python main.py --query "What is the Turing test?"
# Expected: answer + sources printed to terminal
```

---

### `test-agent`
**Phase:** All Phases (cross-cutting)
**File:** `tests/test_*.py`

**Trigger:** User says "write tests", "add tests for", "write pytest for", or any request involving testing a module.

**Scope:**
- Write pytest tests for any module in `src/`
- Mock all external APIs (OpenAI)
- Enforce gate checks defined per agent above

**Responsibilities:**
- Use `unittest.mock.patch` to mock `openai.OpenAI` in all tests touching the API
- Write at least 3 test cases per module: happy path, edge case, error case
- Never make real network calls in tests
- Keep fixtures in `conftest.py`

**Standard Mock Pattern:**
```python
from unittest.mock import patch, MagicMock

@patch("src.embedder.openai.OpenAI")
def test_embed_returns_vectors(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    from src.embedder import embed
    result = embed(["test text"])
    assert len(result) == 1
    assert len(result[0]) == 1536
```

**Guardrails:**
- MUST mock OpenAI in every test that touches `embedder.py` or `generator.py`
- MUST NOT use `assert` on floating point equality — use `pytest.approx` or range checks
- MUST test that metadata fields exist on every loader and chunker output

---

## PHASE → AGENT MAPPING

| Phase | Agent | File | Gate Command |
|-------|-------|------|-------------|
| 0 | — (setup only) | `requirements.txt`, `.env.example`, folder structure | `python -c "import numpy, openai"` |
| 1 | `loader-agent` | `src/loader.py` | `pytest tests/test_loader.py` |
| 2 | `chunker-agent` | `src/chunker.py` | `pytest tests/test_chunker.py` |
| 3 | `embedder-agent` | `src/embedder.py` | `pytest tests/test_embedder.py` |
| 4 | `vector-store-agent` | `src/vector_store.py` | `pytest tests/test_vector_store.py` |
| 5 | `retriever-agent` | `src/retriever.py` | `pytest tests/test_retriever.py` |
| 6 | `generator-agent` | `src/generator.py` | `pytest tests/test_generator.py` |
| 7 | `pipeline-agent` | `src/pipeline.py` + `main.py` | Full CLI end-to-end |
| 8 | — (reflection only) | `LEARNINGS.md` | No code gate |

---

## ESCALATION RULES

If an agent is asked to do something outside its scope:

1. **Refuse the out-of-scope task** — state clearly which agent owns that work
2. **Name the correct agent** — e.g., "That is chunker-agent's job, not mine"
3. **Do not partially implement** — partial cross-agent implementations create bugs that are hard to trace
4. **Ask the user to switch agents** — e.g., "Please invoke chunker-agent for this task"

---

## FOLDER STRUCTURE (Reference)

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
│   ├── loader.py           ← loader-agent
│   ├── chunker.py          ← chunker-agent
│   ├── embedder.py         ← embedder-agent
│   ├── vector_store.py     ← vector-store-agent
│   ├── retriever.py        ← retriever-agent
│   ├── generator.py        ← generator-agent
│   └── pipeline.py         ← pipeline-agent
│
├── tests/
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   └── test_generator.py
│
├── main.py                 ← CLI entrypoint
├── .env                    ← GROQ_API_KEY (never commit)
├── .env.example
├── requirements.txt
├── PRD.md
├── AGENTS.md
└── LEARNINGS.md            ← Phase 8 reflection
```

---

*AGENTS.md v2.0 — RAG From Scratch · No LangChain · Harness Engineering Style*
