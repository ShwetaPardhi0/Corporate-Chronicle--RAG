"""
tests/test_app.py — Basic integration tests for the RAG pipeline.

Run with:
  python -m pytest tests/ -v
"""
import os
import sys
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import DocumentChunker
from src.embeddings.embedder import EmbeddingModel
from src.prompts.prompt_templates import build_rag_prompt, build_mock_response
from src.utils.helpers import load_documents_from_dir, format_sources


# ── DocumentLoader ────────────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_load_inline_text(self):
        loader = DocumentLoader()
        sources = [{
            "type": "text",
            "name": "test_doc",
            "content": "Hello world. This is a test.",
            "metadata": {"category": "test"},
        }]
        docs = loader.load(sources)
        assert len(docs) == 1
        assert docs[0]["content"] == "Hello world. This is a test."
        assert docs[0]["source"] == "test_doc"

    def test_load_missing_file_skips(self):
        loader = DocumentLoader()
        sources = [{"type": "file", "path": "nonexistent_file.pdf"}]
        docs = loader.load(sources)
        assert len(docs) == 0

    def test_load_unsupported_type_skips(self):
        loader = DocumentLoader()
        sources = [{"type": "url", "url": "http://example.com"}]
        docs = loader.load(sources)
        assert len(docs) == 0


# ── DocumentChunker ────────────────────────────────────────────────────────────

class TestDocumentChunker:
    @pytest.fixture(scope="class")
    def embedder(self):
        return EmbeddingModel()

    def test_chunk_single_sentence_returns_one_chunk(self, embedder):
        chunker = DocumentChunker(embedding_model=embedder)
        doc = {"content": "One sentence only.", "source": "test", "metadata": {}}
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "One sentence only."

    def test_chunk_preserves_source(self, embedder):
        chunker = DocumentChunker(embedding_model=embedder)
        doc = {
            "content": "First sentence. Second sentence. Third sentence.",
            "source": "my_source",
            "metadata": {},
        }
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk["source"] == "my_source"


# ── Prompt Templates ───────────────────────────────────────────────────────────

class TestPromptTemplates:
    def _make_result(self, text: str, score: float = 0.9) -> dict:
        return {
            "chunk": {"text": text, "source": "test.pdf", "chunk_index": 0},
            "score": score,
            "rank": 1,
        }

    def test_build_rag_prompt_contains_query(self):
        results = [self._make_result("Some context text.")]
        prompt = build_rag_prompt("What is RAG?", results)
        assert "What is RAG?" in prompt
        assert "Some context text." in prompt

    def test_build_mock_response_contains_query(self):
        results = [self._make_result("Chunk A"), self._make_result("Chunk B", 0.7)]
        response = build_mock_response("Test query", results)
        assert "Test query" in response
        assert "MOCK" in response


# ── Helpers ────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_load_documents_from_missing_dir(self):
        sources = load_documents_from_dir("nonexistent_test_dir_xyz")
        assert sources == []

    def test_format_sources_empty(self):
        output = format_sources([])
        assert "no sources" in output.lower()

    def test_format_sources_shows_rank_and_score(self):
        results = [{
            "chunk": {"text": "Hello world", "source": "doc.pdf"},
            "score": 0.85,
            "rank": 1,
        }]
        output = format_sources(results)
        assert "0.8500" in output
        assert "doc.pdf" in output
