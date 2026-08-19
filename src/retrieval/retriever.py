"""
Retriever — wraps VectorStore similarity search.
Keeps retrieval logic decoupled from the vector storage implementation.
"""
import numpy as np

from src.vectordb.vector_store import VectorStore
from src.embeddings.embedder import EmbeddingModel


class Retriever:
    """Encapsulates query embedding + vector similarity search.

    Having a dedicated retriever layer makes it easy to add re-ranking,
    hybrid search, or metadata filtering without touching the pipeline.
    """

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Embed the query and return the top-k most similar chunks.

        Args:
            query: Natural-language query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts, each with 'chunk', 'score', and 'rank'.
        """
        query_vector: np.ndarray = self.embedding_model.embed_single(query)
        results = self.vector_store.search(query_vector, top_k=top_k)
        print(f"[RETRIEVER] Retrieved {len(results)} chunk(s) for query: '{query[:60]}...'")
        return results
