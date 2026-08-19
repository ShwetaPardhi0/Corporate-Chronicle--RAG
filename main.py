"""
main.py — Entry point for the Production RAG Pipeline.

All orchestration is handled directly here. The pipeline is assembled
from individual modules: ingestion → chunking → embeddings → vectordb
→ retrieval → llm.

Usage:
  python main.py              # Uses existing data in PostgreSQL
  python main.py --reingest   # Clears DB and re-ingests from documents/
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Load .env before any src imports that read env vars
load_dotenv()

from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingModel
from src.vectordb.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.llm_client import LLMClient
from src.utils.helpers import load_documents_from_dir, format_sources, setup_logging

# ── Logging ───────────────────────────────────────────────────────────────────
setup_logging(log_file="logs/app.log")
logger = logging.getLogger(__name__)


# ── Pipeline assembly ─────────────────────────────────────────────────────────

def build_pipeline():
    """Instantiate and wire all RAG components together.

    Returns a dict of components so each layer can be accessed individually.
    """
    embedder = EmbeddingModel()           # shared by chunker + retriever
    loader   = DocumentLoader()
    chunker  = DocumentChunker(embedding_model=embedder, threshold_percentile=90.0)
    store    = VectorStore()              # PostgreSQL + pgvector
    retriever = Retriever(vector_store=store, embedding_model=embedder)
    llm      = LLMClient()               # Gemini 2.5 Flash (mock if no API key)

    return {
        "loader":    loader,
        "chunker":   chunker,
        "embedder":  embedder,
        "store":     store,
        "retriever": retriever,
        "llm":       llm,
    }


# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest(pipeline: dict, sources: list[dict]) -> None:
    """Load → chunk → embed → store documents into PostgreSQL."""
    logger.info("--- INGESTION PHASE STARTED ---")

    docs = pipeline["loader"].load(sources)
    if not docs:
        logger.warning("[MAIN] No documents loaded.")
        return

    chunks = pipeline["chunker"].chunk_documents(docs)
    if not chunks:
        logger.warning("[MAIN] No chunks generated.")
        return

    logger.info(f"[MAIN] Embedding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = pipeline["embedder"].embed(texts)

    pipeline["store"].add(chunks, embeddings)
    logger.info("--- INGESTION PHASE COMPLETED ---")


# ── Query ─────────────────────────────────────────────────────────────────────

def run_query(pipeline: dict, query_text: str, top_k: int = 3) -> dict:
    """Retrieve context and generate a grounded answer."""
    logger.info(f"[MAIN] Query: '{query_text}'")

    matches = pipeline["retriever"].retrieve(query_text, top_k=top_k)
    answer  = pipeline["llm"].generate(query_text, matches)

    return {"query": query_text, "results": matches, "answer": answer}


# ── CLI entry point ───────────────────────────────────────────────────────────

def run():
    print("=" * 62)
    print("   PRODUCTION RAG PIPELINE  (pgvector + Gemini 2.5 Flash)   ")
    print("=" * 62)

    pipeline = build_pipeline()
    store    = pipeline["store"]

    # Decide whether to ingest
    existing = store.count()
    force    = "--reingest" in sys.argv

    if existing > 0 and not force:
        print(f"\n✓ {existing} chunks already in PostgreSQL — skipping ingestion.")
        print("  (Run with --reingest to force re-ingestion)\n")
    else:
        if force and existing > 0:
            print(f"\n--reingest: clearing {existing} existing chunks...")
            store.clear()

        sources = load_documents_from_dir("documents")
        if not sources:
            logger.warning("[MAIN] documents/ is empty — using fallback in-memory text.")
            sources = [{
                "type": "text",
                "name": "rag_overview",
                "content": (
                    "This is a production RAG pipeline. "
                    "Drop PDF or text files into the documents/ folder and run with --reingest."
                ),
                "metadata": {"category": "overview"},
            }]

        ingest(pipeline, sources)

    # Interactive query loop
    print("\nRAG Pipeline ready. Type your question or 'exit' to quit.")
    print("-" * 62)

    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            result = run_query(pipeline, query, top_k=3)

            print("\n[SOURCES]")
            print(format_sources(result["results"]))
            print("\n[ANSWER]")
            print(result["answer"])
            print("-" * 62)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            print(f"Error: {e}")


if __name__ == "__main__":
    run()
