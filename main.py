import argparse
import asyncio
import time

from langchain_chroma import Chroma

from src.ingestion.loader import Ingestion
from src.ingestion.chunking import Chunker
from src.ingestion.embeddings import get_embeddings_model

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.rrf import ReciprocalRankFusion
from src.retrieval.reranker import Reranker
from src.retrieval.hybrid_retriever import HybridRetriever

from src.generation.prompt_builder import PromptBuilder
from src.generation.generator import Generator
from src.generation.formatter import ResponseFormatter

from src.pipeline.rag_pipeline import RAGPipeline

from src.guardrails.input_validation import InputValidator
from src.guardrails.pii_masking import PIIMaskingGuard
from src.guardrails.prompt_injection import PromptInjectionGuard
from src.guardrails.nemo_guard import NemoGuard

from src.test.queries import main_pipeline_test_queries

from config import VECTOR_STORE_PATH

from logger import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #

def _print_trace(timings: dict, total: float):
    width = 22
    print()
    print("=" * 50)
    print("DocuMind Pipeline Trace")
    print("=" * 50)
    print()
    for stage, elapsed in timings.items():
        print(f"  {stage:<{width}}: {elapsed:.3f} s")
    print()
    print("-" * 42)
    print(f"  {'Total':<{width}}: {total:.3f} s")
    print("=" * 50)


def _tick() -> float:
    return time.perf_counter()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DocuMind — ask questions over your PDF documents.",
    )
    parser.add_argument(
        "--query", "-q",
        action="append",
        dest="queries",
        help="Run a single query (repeatable). Overrides the built-in demo list.",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive REPL: type questions until empty line / Ctrl+C.",
    )
    return parser.parse_args(argv)


# ------------------------------------------------------------------ #
# Setup                                                                 #
# ------------------------------------------------------------------ #

def build_pipeline():
    print("\nInitializing DocuMind...\n")

    # Ingestion
    ingestion = Ingestion()
    documents = ingestion.load_pdf()

    # Chunking
    chunker = Chunker()
    chunks = chunker.chunk_documents(documents)

    # Embedding model — lru_cached, same instance reused if called again
    embedding_model = get_embeddings_model()

    # Vector store — load if it exists and has documents, otherwise build and persist
    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=embedding_model,
    )

    if vectorstore._collection.count() == 0:
        logger.info("Vector store is empty. Building and persisting at %s", VECTOR_STORE_PATH)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(VECTOR_STORE_PATH),
        )
        logger.info("Vector store built with %d chunks.", vectorstore._collection.count())
    else:
        logger.info(
            "Loaded existing vector store from %s (%d chunks).",
            VECTOR_STORE_PATH,
            vectorstore._collection.count(),
        )

    # Retriever stack
    # Reranker uses _load_cross_encoder() which is lru_cached — model loads once
    bm25     = BM25Retriever(documents=chunks)   # index loaded from cache or rebuilt
    dense    = DenseRetriever(vectorstore=vectorstore)
    rrf      = ReciprocalRankFusion()
    reranker = Reranker()

    retriever = HybridRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
        rrf=rrf,
        reranker=reranker,
    )

    # RAG pipeline
    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        generator=Generator(),
        formatter=ResponseFormatter(),
    )

    # Guardrails
    validator       = InputValidator()
    pii_masker      = PIIMaskingGuard()
    regex_guard     = PromptInjectionGuard()
    nemo_guard      = NemoGuard()

    return pipeline, validator, pii_masker, regex_guard, nemo_guard


# ------------------------------------------------------------------ #
# Single query                                                          #
# ------------------------------------------------------------------ #

async def run_one(pipeline, validator, pii_masker, regex_guard, nemo_guard, query: str, index: int | None = None):
    label = f"Query #{index}: {query}" if index is not None else f"Query: {query}"
    print("\n" + "=" * 50)
    print(label)
    print("=" * 50)

    timings: dict[str, float] = {}
    pipeline_start = _tick()

    # ---------------------------------------------------------- #
    # 1. Input Validation                                          #
    # ---------------------------------------------------------- #
    t0 = _tick()
    try:
        is_valid, reason = validator.is_valid_query(query)
    except Exception as e:
        print(f"\n[Error] Input validation failed: {e}\n")
        return
    timings["Input Validation"] = _tick() - t0

    if not is_valid:
        print(f"\n[Blocked] Invalid input — {reason}\n")
        _print_trace(timings, _tick() - pipeline_start)
        return

    # ---------------------------------------------------------- #
    # 2. PII Masking                                               #
    # ---------------------------------------------------------- #
    t0 = _tick()
    pii_result = pii_masker.check(query)
    timings["PII Masking"] = _tick() - t0

    masked_query = pii_result.masked_query

    if pii_result.pii_detected:
        detected = [e.type for e in pii_result.entities]
        print(f"\n[PII Masked] Detected: {detected}")

    # ---------------------------------------------------------- #
    # 3. Regex Guard (Prompt Injection)                            #
    # ---------------------------------------------------------- #
    t0 = _tick()
    regex_result = regex_guard.check(masked_query)
    timings["Regex Guard"] = _tick() - t0

    if regex_result.action == "BLOCK":
        print(f"\n[Blocked] Prompt injection detected (score={regex_result.score}).\n")
        _print_trace(timings, _tick() - pipeline_start)
        return

    # ---------------------------------------------------------- #
    # 4. NeMo Guardrails                                           #
    # ---------------------------------------------------------- #
    t0 = _tick()
    try:
        nemo_result = await nemo_guard.check(masked_query)
    except Exception as e:
        logger.warning("NeMo guard failed, failing open: %s", e)
        nemo_result = None
    timings["NeMo Guardrails"] = _tick() - t0

    if nemo_result and nemo_result.action == "BLOCK":
        print(f"\n[Blocked] NeMo Guardrails — {nemo_result.reason}\n")
        _print_trace(timings, _tick() - pipeline_start)
        return

    # ---------------------------------------------------------- #
    # 5 – 8. RAG Pipeline (with per-stage timing)                 #
    # ---------------------------------------------------------- #
    try:
        # Retrieval
        t0 = _tick()
        retrieved_docs = pipeline.retriever.retrieve(query=masked_query)
        timings["Hybrid Retrieval"] = _tick() - t0

        # Prompt building
        t0 = _tick()
        prompt = pipeline.prompt_builder.build(
            query=masked_query,
            retrieved_docs=retrieved_docs,
        )
        timings["Prompt Builder"] = _tick() - t0

        # Generation
        t0 = _tick()
        answer = pipeline.generator.generate(prompt=prompt)
        timings["Generation"] = _tick() - t0

        # Formatting
        t0 = _tick()
        response = pipeline.formatter.format(
            answer=answer,
            retrieved_docs=retrieved_docs,
        )
        timings["Response Formatter"] = _tick() - t0

    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}\n")
        logger.exception("Pipeline execution failed.")
        return

    total = _tick() - pipeline_start

    # ---------------------------------------------------------- #
    # Output                                                        #
    # ---------------------------------------------------------- #
    print("\nAnswer")
    print("-" * 50)
    print(response.answer)

    print(f"\nSources  ({len(response.sources)} cited)")
    print("-" * 50)
    for j, src in enumerate(response.sources, start=1):
        print(
            f"  [{j}] {src.source}  "
            f"p.{src.page}  "
            f"score={src.relevance_score}"
        )

    _print_trace(timings, total)


# ------------------------------------------------------------------ #
# Main query loop                                                       #
# ------------------------------------------------------------------ #

async def run(pipeline, validator, pii_masker, regex_guard, nemo_guard, queries: list[str]):
    print("\nDocuMind is ready.\n")

    for i, query in enumerate(queries, start=1):
        await run_one(pipeline, validator, pii_masker, regex_guard, nemo_guard, query, index=i)

    print("\n" + "=" * 50)
    print("Done.")
    print("=" * 50)


async def run_interactive(pipeline, validator, pii_masker, regex_guard, nemo_guard):
    print("\nDocuMind interactive mode. Empty line or Ctrl+C to exit.\n")
    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            print("Bye.")
            break
        await run_one(pipeline, validator, pii_masker, regex_guard, nemo_guard, query)


def main(argv: list[str] | None = None):
    args = _parse_args(argv)

    try:
        pipeline, validator, pii_masker, regex_guard, nemo_guard = build_pipeline()
    except Exception as e:
        print(f"\n[Fatal] Failed to initialize DocuMind: {e}")
        logger.exception("Startup failed.")
        return

    if args.interactive:
        asyncio.run(run_interactive(pipeline, validator, pii_masker, regex_guard, nemo_guard))
        return

    queries = args.queries if args.queries else list(main_pipeline_test_queries)
    asyncio.run(run(pipeline, validator, pii_masker, regex_guard, nemo_guard, queries))


if __name__ == "__main__":
    main()
