# Learning Guide

Start with the synthetic samples, then use the SEPP manuals for local retrieval. Each section connects a concept to executable code. Read README for setup and privacy details.

## 1. What is RAG?

RAG combines document retrieval with answer generation. It matters because a model may not know your private or updated material. The pipeline retrieves chunks before calling the LLM.

Implementation: `backend/rag/pipeline.py`. Read `RAGPipeline.answer` to follow how it works.

**Exercise:** Ask the same question before and after uploading a reference document; inspect the source difference.

## 2. Why retrieve instead of sending full documents?

Retrieval selects focused evidence rather than repeatedly sending an entire manual. This reduces context size and makes the evidence inspectable, while introducing a chance of missing important passages.

Implementation: `backend/rag/retriever.py`. Read `Retriever.search` to follow how it works.

**Exercise:** Compare top-k=1 and top-k=10 on a question requiring two sections. Record whether each answer has enough evidence.

## 3. Document loading

Loading converts a file into text sections. PDF pages keep physical page numbers; DOCX tables and paragraphs retain their order. Extraction quality limits every later stage.

Implementation: `backend/ingestion/loader.py`. Read `extract, safe_filename` to follow how it works.

**Exercise:** Upload a text PDF, a DOCX table, and a scanned PDF. Compare extractable text and the error for an image-only file.

## 4. Text chunking

A chunk is a searchable slice of text. Recursive character splitting prefers natural boundaries, and overlap repeats some boundary context. Tiny chunks lose meaning; large ones may dilute matches and exceed model token windows.

Implementation: `backend/ingestion/chunker.py`. Read `Chunker.split` to follow how it works.

**Exercise:** Change CHUNK_SIZE from 800 to 400 and re-import. Compare chunk count, split sentences, and retrieval. Restore settings to reopen the earlier index.

## 5. Embeddings

An embedding is a learned numeric representation of text. MiniLM embeds both queries and chunks in the same space on your CPU. Normalization makes vectors unit length. Embedding computation does not contact Gemini.

Implementation: `backend/ingestion/embedder.py`. Read `Embedder.documents, Embedder.query, load_model` to follow how it works.

**Exercise:** Run embedding_demo.py. Replace the sentences with SEPP terms, paraphrases and unrelated statements.

## 6. Vector dimensions

A dimension is one coordinate in a learned space, not a named human feature. This model outputs 384 coordinates. All vectors in one collection must use the same model and dimension.

Implementation: `backend/ingestion/embedder.py`. Read `Embedder.dimension` to follow how it works.

**Exercise:** Print the shape of ten sentence embeddings and explain why it is (10, 384). Try the wrong-dimension test.

## 7. Cosine similarity

Cosine compares vector directions using dot(a,b)/(norm(a)*norm(b)). Similarity is not a probability. Cosine distance is 1 minus similarity, so lower distance is closer.

Implementation: `examples/similarity_demo.py`. Read `cosine` to follow how it works.

**Exercise:** Add lexical overlap without semantic overlap, then a synonym-heavy paraphrase. Compare scores and note failures.

## 8. Vector databases

A vector database stores vectors, text and metadata and finds nearby vectors. PersistentClient saves local Chroma data across restarts; metadata enables source tracing and filtering.

Implementation: `backend/rag/vector_store.py`. Read `VectorStore.insert, search, by_document, delete, count` to follow how it works.

**Exercise:** Run retrieval_demo.py. Index a file in the app, stop and restart the API, and confirm its chunks remain. Delete it and check the count.

## 9. Semantic search

Semantic search embeds a question and compares its vector to stored chunk vectors. It can match paraphrases that share few exact words, but general-purpose embeddings can miss technical identifiers.

Implementation: `backend/rag/retriever.py`. Read `Retriever.search` to follow how it works.

**Exercise:** Use /search for a paraphrase of a passage and for an exact error code. Inspect whether the expected passage is found.

## 10. Retrieval top-k

Top-k sets the maximum number of candidate chunks returned. More chunks may improve recall but add noise, latency and tokens. A document filter narrows the candidate set before ranking.

Implementation: `backend/models/schemas.py`. Read `SearchRequest, ChatRequest` to follow how it works.

**Exercise:** Try k=1, 3, 5 and 10 against ten known questions. Note when more context makes evidence worse.

## 11. Context construction

Retrieved chunks are packaged as reference JSON with source IDs, names and pages. The generator receives this selected evidence, not the entire collection. IDs map the answer back to the hits.

Implementation: `backend/rag/prompt.py`. Read `build_prompt` to follow how it works.

**Exercise:** Inspect the debugger context. Verify every S-number corresponds to a real chunk and PDF page.

## 12. Prompt engineering

The system instruction tells Gemini to use only supplied evidence, refuse insufficient questions and cite source IDs. A separate user prompt carries JSON data and the question. Documents are untrusted reference data.

Implementation: `backend/rag/prompt.py`. Read `SYSTEM, build_prompt` to follow how it works.

**Exercise:** Try a synthetic document containing “ignore previous instructions.” Check behavior without assuming prompt separation is foolproof.

## 13. Gemini API integration

The official SDK sends the question and context to a hosted model. Credentials remain in backend environment configuration. Low temperature reduces variability; retries are limited to transient failures and quota errors.

Implementation: `backend/rag/generator.py`. Read `LLMProvider, GeminiGenerator.generate` to follow how it works.

**Exercise:** Run the model-list helper. With an approved key, use synthetic sample documents and inspect token usage. Read the mocked error tests before making real calls.

## 14. Hallucinations and grounding

Grounding links an answer to supplied evidence. Retrieval, refusal instructions and citations help, but models can still invent or misread facts. A distance cutoff is only useful after evaluation.

Implementation: `backend/rag/pipeline.py`. Read `RAGPipeline.answer` to follow how it works.

**Exercise:** Ask five questions that the documents cannot answer. Score refusal quality separately from retrieval quality; test several cutoff values using measured distances.

## 15. Source citations

Citations are references to retrieved passages. PDF page metadata helps readers verify them. Identifier checking rejects nonexistent S-numbers but cannot prove a statement follows from a real source.

Implementation: `backend/rag/prompt.py`. Read `validate_citations` to follow how it works.

**Exercise:** Read the citation test. Simulate [S99], an uncited answer, and a valid citation attached to a false claim. Explain which checks catch each case.

## 16. FastAPI

FastAPI exposes typed HTTP endpoints around backend services. Pydantic validates inputs, and response models keep contracts clear. Local clients can upload, list, delete, search or chat through the same API.

Implementation: `backend/main.py`. Read `create_app; backend/api route functions` to follow how it works.

**Exercise:** Open /docs and send an empty query, top_k=0, a valid search and a missing deletion. Compare 422, 200 and 404 responses.

## 17. Streamlit

Streamlit renders a Python UI with session state. It talks to the API over HTTP and holds chat history for display and follow-up context. It sends at most four recent question/answer turns, truncated to 4,000 characters per field. The backend resolves references into a standalone retrieval question, retrieves fresh evidence, and grounds the next answer in that evidence. Streamlit does not access the vector database directly.

Implementation: `frontend/app.py`. Read `api, show_sources` to follow how it works.

**Exercise:** Ask about handshake error codes, then ask which of those are server errors. Compare the resolved retrieval query in debug output with the follow-up text. Start a new conversation and verify the earlier turns are no longer sent. Change the document filter and confirm conversation context resets.

## 18. RAG evaluation

Evaluate extraction, retrieval and generation separately. A good-looking answer can hide missed evidence. Useful measures include retrieval recall@k, citation support, factual correctness, refusal accuracy, latency and token cost.

Implementation: `tests/ and frontend/app.py`. Read `test_retriever, test_prompt; RAG Debugger` to follow how it works.

**Exercise:** Create 20 questions: 10 answerable, 5 requiring multiple sections, 5 unanswerable. Record expected document/pages, retrieved rank, answer support and latency. Compare configurations without changing the question set.

## 19. Traditional RAG vs Agentic RAG

Traditional RAG follows a fixed retrieve-then-generate path. Agentic RAG can decide to rewrite, retrieve again, assess evidence or verify an answer. Those loops increase cost and complexity and need explicit limits.

Implementation: `backend/agents/README.md`. Read `Future graph design (not implemented)` to follow how it works.

**Exercise:** Sketch a bounded graph that rewrites at most once when evidence is weak. Decide what measurable test triggers re-retrieval before adding code.

## How to Understand This Project in 7 Days

- **Day 1: Document loading and chunking.** Read loader/chunker, run their tests, extract two PDF pages, and compare chunk sizes. Explain metadata preservation and the OCR limit.
- **Day 2: Embeddings and vector similarity.** Run embedding_demo.py and similarity_demo.py. Explain 384 dimensions, normalization and cosine distance with your own sentence examples.
- **Day 3: ChromaDB and semantic retrieval.** Run retrieval_demo.py, index samples, restart, filter documents and delete one. Trace SQLite and Chroma responsibilities.
- **Day 4: Gemini API and prompt engineering.** Configure an approved key locally, check model availability, read the prompt and retry code, and make one synthetic-context request. Inspect citations and usage.
- **Day 5: Complete RAG pipeline.** Trace one question through pipeline.answer. Use debugger output to identify every stage and test the no-context path without calling Gemini.
- **Day 6: FastAPI and Streamlit.** Use /docs and curl, then inspect the HTTP-only frontend. Add a small UI improvement such as a latency metric without importing backend logic.
- **Day 7: RAG debugging, evaluation, and introduction to Agentic RAG.** Build the 20-question evaluation set, compare two configurations, inspect failure causes and sketch a bounded future LangGraph workflow.

Keep an experiment journal: configuration, expected evidence, actual top-k, answer support, refusal behavior, time and tokens. Changing one variable at a time makes the results easier to explain.

## Experiment: semantic similarity is not an exact-version check

Ask `find podman cmds with 1.0.4-3 tag`. MiniLM alone can retrieve Podman commands with other tags because the surrounding meaning is similar. Inspect `backend/rag/lexical.py` for required-literal extraction and BM25, and `backend/rag/retriever.py` for ranking fusion. Exact tags constrain eligible passages before ranking. Compare a missing tag with a known tag such as `24.2.0`, and inspect the `required_terms`, `message`, `lexical_score`, `hybrid_score`, and actual cosine `distance` in API/debug output. A source missing from the imported collection cannot be found by retrieval. Saved imports are snapshots; source coverage and ranking quality are separate concerns.

## Experiment: an exact command must beat a partial semantic match

`find podman build cmd` previously retrieved `new 30.txt` below unrelated `podman load`/`tag` chunks. The source was indexed correctly; mixing rank lists allowed partial matches to win. Read `query_tokens`, `contains_literal`, and `matching_excerpt` in `backend/rag/lexical.py`, then the match-tier selection in `backend/rag/retriever.py`. The same query now returns one exact passage from `new 30.txt`, displaying its build line without surrounding copy/push commands. Try another existing CLI phrase and compare quoted and unquoted queries. When no exact phrase or complete keyword coverage exists, results are labelled semantic suggestions, not exact evidence. These rules improve technical lookup but are not a general guarantee of relevance for every natural-language question.


## Consolidated retrieval correction — 19 October 2026

A semantic nearest neighbour answers “what looks similar,” which does not establish that a command, version, URL or complete file is present. This project therefore separates query planning, source selection and presentation. Compare links, commands, documents and semantic modes with the examples in RETRIEVAL_ACCEPTANCE.md. Exact technical lookup uses source evidence; vector search remains available for exploratory similarity.
