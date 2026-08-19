# 🧠 Modular Production RAG Pipeline

A production-grade, modular **Retrieval-Augmented Generation (RAG)** pipeline built in Python. It supports PDF document ingestion, semantic chunking, vector similarity search via **PostgreSQL + pgvector**, and answer generation using **Gemini 2.5 Flash**.

---

## ✨ Features

- 📄 **PDF & Text Ingestion** — Extracts text from PDFs using PyMuPDF
- 🪄 **Semantic Chunking** — Splits documents based on sentence embedding similarity (no fixed character limits)
- 🔢 **Local Embeddings** — Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- 🗄️ **PostgreSQL + pgvector** — Persistent vector storage with cosine similarity search
- 🤖 **Gemini Generation** — Answers grounded in retrieved context via `gemini-2.5-flash`
- 🐳 **Docker Ready** — Multi-stage Dockerfile for clean containerized deployment

---

## 📁 Project Structure

```
RAG/
├── src/rag/
│   ├── __init__.py        # Package exports
│   ├── loader.py          # PDF & text document loader (PyMuPDF)
│   ├── chunker.py         # Semantic chunker using embedding cosine distance
│   ├── embedder.py        # SentenceTransformer embedding wrapper
│   ├── vector_store.py    # PostgreSQL + pgvector vector store
│   ├── generator.py       # Gemini API answer generator
│   └── pipeline.py        # Orchestrator tying all components together
├── documents/             # Drop PDF/text files here for ingestion
├── main.py                # CLI entry point
├── Dockerfile
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚙️ Setup

### 1. Prerequisites

- Python 3.10+
- PostgreSQL with the **pgvector** extension installed

### 2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. Enable pgvector in PostgreSQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 🚀 Usage

### Run the pipeline

```bash
# First run: ingests all files from documents/ into PostgreSQL
python main.py

# Subsequent runs: skips ingestion (data already in DB)
python main.py

# Force re-ingestion (clears existing chunks first)
python main.py --reingest
```

### Add your documents

Drop any `.pdf` or `.txt` files into the `documents/` folder before running. The pipeline will automatically discover and ingest them.

---

## 🐳 Docker

```bash
# Build
docker build -t rag-pipeline .

# Run (mounts your local documents/ folder)
docker run -it --env-file .env -v ./documents:/app/documents rag-pipeline
```

> **Note:** The default `DB_HOST` inside Docker is `host.docker.internal` to reach PostgreSQL on your host machine.

---

## 🏗️ Architecture

```
documents/          →  DocumentLoader
                    →  SemanticChunker (cosine distance split)
                    →  EmbeddingModel (all-MiniLM-L6-v2)
                    →  VectorStore (PostgreSQL + pgvector)

User Query          →  EmbeddingModel
                    →  VectorStore (cosine similarity search)
                    →  GeminiGenerator (grounded answer)
```

---

## 🔮 Roadmap

- [ ] **Phase 3**: Switch from local embeddings to Gemini `text-embedding-004` API
- [ ] **Phase 5**: FastAPI REST layer for serving queries
- [ ] **Phase 6**: Multi-document metadata filtering

---

## 📄 License

MIT
