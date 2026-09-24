"""Bootstrap helpers for constructing the hybrid retriever stack."""

from langchain_chroma import Chroma

from config import VECTOR_STORE_PATH
from logger import get_logger
from src.ingestion.chunking import Chunker
from src.ingestion.embeddings import get_embeddings_model
from src.ingestion.loader import Ingestion
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.rrf import ReciprocalRankFusion

logger = get_logger(__name__)


def create_hybrid_retriever() -> HybridRetriever:
    """
    Build a HybridRetriever from PDFs in data/ and the persisted Chroma store.
    """
    documents = Ingestion().load_pdf()
    chunks = Chunker().chunk_documents(documents)
    embedding_model = get_embeddings_model()

    vectorstore = Chroma(
        persist_directory=str(VECTOR_STORE_PATH),
        embedding_function=embedding_model,
    )

    if vectorstore._collection.count() == 0:
        if not chunks:
            raise ValueError("No PDF chunks available to build the vector store.")
        logger.info("Vector store empty — building at %s", VECTOR_STORE_PATH)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(VECTOR_STORE_PATH),
        )

    return HybridRetriever(
        dense_retriever=DenseRetriever(vectorstore=vectorstore),
        bm25_retriever=BM25Retriever(documents=chunks),
        rrf=ReciprocalRankFusion(),
        reranker=Reranker(),
    )
