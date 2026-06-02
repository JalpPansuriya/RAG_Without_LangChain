# LEARNINGS.md — RAG From Scratch Reflections

This document captures the insights, mathematical formulations, and engineering takeaways gained from constructing a Retrieval-Augmented Generation (RAG) system from first principles without relying on high-level frameworks like LangChain or LlamaIndex.

---

## Phase 0: Project Scaffold
- **Key Takeaway**: Establishing clean directory separations (`src/` for core modules and `tests/` for tests) and dependency minimalization is critical.
- **Insight**: Keeping dependencies limited to `numpy`, `openai`, `pytest`, and `python-dotenv` minimizes environment bloat and makes the codebase easy to audit and learn from.

---

## Phase 1: Document Loader
- **Key Takeaway**: Low-level file I/O operations require handling encoding issues.
- **Insight**: In the real world, document corpora contain diverse character formats. Implementing UTF-8 parsing with a fallback (e.g. `latin-1` or `cp1252`) prevents pipeline crashes during mass ingestion.

---

## Phase 2: Text Chunker
- **Key Takeaway**: Text chunking is a trade-off between retaining context and query resolution.
- **Insight**: 
  - Fixed-size character splitting ensures consistent vector store loads but can truncate thoughts mid-sentence.
  - Overlap is critical to bridge the semantic gap across boundaries.
  - Sentence splitting preserves complete ideas but can create highly variable chunk sizes.

---

## Phase 3: Embedding Generator
- **Key Takeaway**: OpenAI's embedding API has rate limits that require custom error mitigation.
- **Insight**: 
  - High-dimension embeddings (1536 elements for `text-embedding-3-small`) capture semantic relationships.
  - In a raw API client, bulk requests must be sliced (max 100 texts per call) to avoid HTTP payload issues.
  - Implementing exponential backoff on API rate limit codes ensures reliable execution.
  - Mocking the OpenAI SDK during testing is crucial to verify format validation without running up API costs or requiring internet access.

---

## Phase 4: Custom Vector Store & Similarity Math
- **Key Takeaway**: Vector databases are fundamentally matrices optimized for cosine similarity.
- **Insight**: Under the hood, vector search is linear algebra. By utilizing `numpy` dot products and norms, we can search hundreds of documents in milliseconds without complex vector databases.

### Cosine Similarity Mathematics

Given a query vector $Q$ and a document chunk vector $D$ in a 1536-dimensional space, the cosine similarity is defined as:

$$\text{Similarity}(Q, D) = \cos(\theta) = \frac{Q \cdot D}{\|Q\| \|D\|} = \frac{\sum_{i=1}^{n} Q_i D_i}{\sqrt{\sum_{i=1}^{n} Q_i^2} \sqrt{\sum_{i=1}^{n} D_i^2}}$$

In our implementation, we leverage NumPy to perform this calculation in batch across all stored vectors simultaneously:
1. **Dot Product**: `np.dot(vecs_matrix, q_arr)` multiplies corresponding elements of the matrices and sums them.
2. **Euclidean Norms**: `np.linalg.norm(q_arr)` and `np.linalg.norm(vecs_matrix, axis=1)` compute lengths.
3. **Division**: Dividing dot products by the product of lengths yields the cosine similarity ranging from $[-1, 1]$. An epsilon (`1e-9`) prevents division by zero.

---

## Phase 5: Query Retriever
- **Key Takeaway**: Retrieval bridges offline index storage with online search query parameters.
- **Insight**: Retrieval must map the query vector into the same vector space as the index. Using helper modules allows retrieval to remain simple, fast, and purely focused on matching coordinates.

---

## Phase 6: Answer Generator
- **Key Takeaway**: Grounding is key to eliminating LLM hallucinations.
- **Insight**: 
  - LLMs are naturally creative generators. By writing strict system prompt rules ("Answer ONLY from the provided context") and mapping source documents, we restrict the model to acting as a summarization engine.
  - If retrieved context chunks are empty, the generator should bypass calling OpenAI completely and return the fallback message immediately, saving costs and guaranteeing grounding.

---

## Phase 7: Pipeline Orchestrator
- **Key Takeaway**: Separating index building (offline) from querying (online) is essential for scalability.
- **Insight**: 
  - Building the index is computationally expensive since it embeds every chunk of the corpus.
  - Querying is cheap, requiring only one embedding API call for the user's query followed by high-speed NumPy similarity math.
  - Saving the index to `data/vector_store.json` enables instant query start-ups.
