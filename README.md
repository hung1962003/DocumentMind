# DocuMind

Document Q&A with hybrid RAG — load PDFs, retrieve with dense + BM25 fusion and reranking, then answer with grounded LLM generation behind input / PII / prompt-injection guardrails.

## Features

- PDF ingestion (PyMuPDF) → chunking → HuggingFace embeddings → Chroma
- Hybrid retrieval: dense + BM25 → Reciprocal Rank Fusion → cross-encoder rerank
- Groq LLM generation with source citations
- Guardrails: input validation, PII masking, regex prompt-injection scoring, NeMo Guardrails
- Optional LangGraph adaptive RAG (confidence check + query rewrite)

## Requirements

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv)
- CUDA-capable GPU recommended (PyTorch cu126 index in `pyproject.toml`)
- API keys: `GROQ_API_KEY`, `NVIDIA_NIM_API_KEY`

## Setup

```bash
# Clone and install
uv sync

# Environment
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY and NVIDIA_NIM_API_KEY

# Add PDFs to answer questions about
mkdir -p data
# place *.pdf files into data/
```

## Run

```bash
# Demo batch (built-in test queries)
uv run python main.py

# Single query
uv run python main.py --query "What is the main topic of the document?"

# Interactive REPL
uv run python main.py --interactive

# Console script (same as main)
uv run documind --query "Summarize the introduction."
```

## Project layout

```
main.py                 # CLI entry
config.py               # paths, chunking, retrieval, LLM, patterns
src/
  ingestion/            # load, chunk, embed
  retrieval/            # BM25, dense, RRF, rerank, hybrid
  generation/           # prompt, generate, format
  pipeline/             # RAGPipeline
  guardrails/           # validation, PII, injection, NeMo
  orchestrator/         # LangGraph adaptive RAG
  vector_store/         # Chroma helpers
```

## Notes

- Vector store persists under `vectorstore/` (gitignored).
- BM25 index cache lives under `cache/` (gitignored).
- Logs rotate under `logs/`.
"# DocumentMind" 
