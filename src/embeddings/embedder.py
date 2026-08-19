import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    """Wraps SentenceTransformer to generate dense vector embeddings for texts."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"[EMBED] Loading embedding model: {model_name}")
        print(f"[EMBED] (First run downloads ~80MB; subsequent runs use cache)")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, show_progress_bar=False)
        return np.array(vectors)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
