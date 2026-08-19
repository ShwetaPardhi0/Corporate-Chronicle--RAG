"""
FastAPI routes — exposes the RAG pipeline as a REST API.

Endpoints:
  GET  /health   — Health check.
  POST /ingest   — Scan documents/ and add chunks to the vector store.
  POST /query    — Ask a question and get a grounded answer.
"""
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingModel
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import LLMClient
from src.utils.helpers import load_documents_from_dir

router = APIRouter()

# ── Lazy singleton components ─────────────────────────────────────────────────
_components: dict | None = None


def get_components() -> dict:
    global _components
    if _components is None:
        embedder = EmbeddingModel()
        _components = {
            "loader":    DocumentLoader(),
            "chunker":   DocumentChunker(embedding_model=embedder),
            "embedder":  embedder,
            "store":     VectorStore(),
            "retriever": Retriever(vector_store=VectorStore(), embedding_model=embedder),
            "llm":       LLMClient(),
        }
        # Share the same store instance in retriever
        _components["retriever"] = Retriever(
            vector_store=_components["store"],
            embedding_model=embedder,
        )
    return _components



# ── Request / Response models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]


class IngestRequest(BaseModel):
    docs_dir: str = "documents"


class IngestResponse(BaseModel):
    message: str
    chunks_added: int


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/health")
def health_check():
    """Returns service health status."""
    return {"status": "ok"}


@router.post("/ingest", response_model=IngestResponse)
def ingest_documents(request: IngestRequest):
    """Scan docs_dir for PDF/text files and ingest them into the vector store."""
    c = get_components()
    sources = load_documents_from_dir(request.docs_dir)

    if not sources:
        raise HTTPException(
            status_code=404,
            detail=f"No supported documents found in '{request.docs_dir}'",
        )

    docs    = c["loader"].load(sources)
    chunks  = c["chunker"].chunk_documents(docs)
    texts   = [ch["text"] for ch in chunks]
    vectors = c["embedder"].embed(texts)

    before = c["store"].count()
    c["store"].add(chunks, vectors)
    after  = c["store"].count()

    return IngestResponse(
        message=f"Ingested {len(sources)} file(s) successfully.",
        chunks_added=after - before,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Answer a question using retrieved context from the vector store."""
    c = get_components()

    if c["store"].count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Please call /ingest first.",
        )

    matches = c["retriever"].retrieve(request.question, top_k=request.top_k)
    answer  = c["llm"].generate(request.question, matches)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[
            {
                "rank":    r["rank"],
                "score":   round(r["score"], 4),
                "source":  r["chunk"]["source"],
                "snippet": r["chunk"]["text"][:200],
            }
            for r in matches
        ],
    )

