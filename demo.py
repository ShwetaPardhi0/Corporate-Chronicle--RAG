
"""pip install sentence-transformers anthropic numpy

Then set your API key:
    export ANTHROPIC_API_KEY="your-key-here"
"""


#imports
import os
import math
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import anthropic



# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DOCUMENT LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_documents(sources: list[dict]) -> list[dict]:


    documents= []

    for source in sources :
        if source['type'] =='text' :
            # Direct text input (for demos and testing)
            doc = {
                'content': source['content'],
                'source': source.get('name','inline_text'),
                'metadata': source.get('metadata',{})
            }
            documents.append(doc)
    
        elif source['type'] == 'file':
            # Read from a .txt file on disk
            # In production: you'd handle PDF, DOCX, HTML, etc.
            path = source['path']
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8'):

                     doc = {
                        'content': f.read(),
                        'source': path,
                        'metadata': source.get('metadata',{})
                     }
                     documents.append(doc) 

            else:
                print(f"Warning file not found: {path}")

    print(f"[LOAD] Loaded {len(documents)} document(s)")
    return documents

# ─────────────────────────────────────────────────────────────────────────────
      # STEP 2: CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
          
def chunk_document(document: dict, chunk_size: int= 500, chunk_overlap: int = 50) -> list[dict]:

    text = document['content']
    chunks = []
    step = chunk_size - chunk_overlap      # e.g., 500 - 50 = 450 chars per step


    chunk_index = 0
    for start in range(0, len(text),step):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Skip tiny trailing chunks (less than 50 chars usually aren't useful)
        if len(chunk_text) < 50:
            continue
        
        # Each chunk carries metadata so we can trace it back to its source
        chunk = {
            'text': chunk_text,
            'chunk_index': chunk_index,   # Position within document
            'source': document['source'], # For citations
            'metadata': {
                **document['metadata'],   # Inherit parent doc metadata
                'char_start': start,      # Where in the original doc this chunk begins
                'char_end': min(end, len(text)),
            }
        }
        chunks.append(chunk)
        chunk_index += 1
    
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int= 500, chunk_overlap: int = 50) -> list[dict]:
    """
    Chunk all documents. Returns a flat list of all chunks.
    """
    all_chunks = []
    for doc in documents:
        doc_chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(doc_chunks)

    print(f"[CHUNK] created {len(all_chunks)} chunks from {len(documents)} document(s)")
    return all_chunks
            


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def __init__(self, model_name: str= 'all-MiniLM-L6-v2'):
    print(f"[EMBED] Loading enbedding model: {model_name}")
    print(f"[EMBED] (First run downloads ~80MB - subsequent eun use cache)")
    self.model = SentenceTransformer(model_name)
    self.model_name= model_name


def embed(self, texts: list[str]) -> np.ndarray:
    # show_progress_bar=False for cleaner output in production
    vectors = self.model.encode(texts, show_progress_bar=False)
    return np.array(vectors)   # Ensure it's a numpy array

def embed_single(self, text:str) -> np.ndarray:
    """
    Convenience method to embed one string → 1D vector.
    Used when embedding a query at search time.
    """
    return self.embed([text])[0]  # Take first (and only) row


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: VECTOR DATABASE (In-Memory Implementation)
# ─────────────────────────────────────────────────────────────────────────────



class VectorStore:

    def __init__(self):
        self.vector = None     # Will become np.ndarray of shape (N, D)
        self.chunks = []       # List of chunk dicts (text + metadata)
        print("[VECTORSTORE] Initialized empty vector store")


    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:

        assert len(chunks) == len(vectors), "Chunks and vectors must have same length"  # assert checks wether the condition is true

        if self.vectors is None:
            # First add: just store everything
            self.vectors = vectors
        else:
            # Subsequent adds: stack new vectors onto existing
            # np.vstack stacks arrays vertically: shape (A, D) + (B, D) → (A+B, D)
            self.vectors = np.vstack([self.vectors, vectors])

        
        self.chunks.extend(chunks)
        print(f"[VECTORSTORE] Added {len(chunks)} chunks. total: {len(self.chunks)}")


    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:

        if self.vectors is None or len(self.chunks) == 0:
            return []
        
        query_norm = query_vector /(np.linalg.norm(query_vector) + 1e-10)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        vectors_norm = self.vectors / norms
        similarities = vectors_norm @ query_norm
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(similarities[idx]),  # Convert numpy float → Python float
                'rank': rank + 1
            })
        
        return results
    

    def save(self, path: str) -> None:

        data= {
            'chunks': self.chunks,
            'vector_shape': self.vectors.shape if self.vectors is not None else None
        }
        with open(f"{path}.json", 'w') as f:
            json.dump(data, f)
        if self.vectors is not None:
            np.save(f"{path}.npy", self.vectors)
        print(f"[VECTORSTORE] Saved to {path}.json + {path}.npy")
    
    def load(self, path: str) -> None:
        """Load a previously saved vector store."""
        with open(f"{path}.json", 'r') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        if data['vector_shape']:
            self.vectors = np.load(f"{path}.npy")
        print(f"[VECTORSTORE] Loaded {len(self.chunks)} chunks")
