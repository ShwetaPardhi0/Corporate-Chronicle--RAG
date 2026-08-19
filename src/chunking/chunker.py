import re
import numpy as np

class DocumentChunker:
    """Handles parsing and splitting document text into semantic snippets (chunks)
    using sentence-level embeddings.
    """
    
    def __init__(self, embedding_model=None, threshold_percentile: float = 90.0):
        self.embedding_model = embedding_model
        self.threshold_percentile = threshold_percentile

    def chunk_document(self, document: dict) -> list[dict]:
        text = document['content']
        source = document['source']
        metadata = document.get('metadata', {})
        
        # 1. Split text into sentences using regex
        # Splits by punctuation followed by space or newline, retaining punctuation.
        sentence_splits = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentence_splits if s.strip()]
        
        if not sentences:
            return []
            
        if len(sentences) == 1 or not self.embedding_model:
            # Fallback to single chunk if only 1 sentence or no embedder provided
            return [{
                'text': text,
                'chunk_index': 0,
                'source': source,
                'metadata': {
                    **metadata,
                    'char_start': 0,
                    'char_end': len(text)
                }
            }]
            
        # 2. Encode all sentences
        embeddings = self.embedding_model.embed(sentences)
        
        # 3. Calculate distance between adjacent sentences
        distances = []
        for i in range(len(sentences) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i+1]
            
            # Cosine distance: 1 - cosine_similarity
            norm1 = np.linalg.norm(vec1) + 1e-10
            norm2 = np.linalg.norm(vec2) + 1e-10
            sim = np.dot(vec1, vec2) / (norm1 * norm2)
            distances.append(1.0 - sim)
            
        # 4. Set dynamic threshold using percentile (e.g. 90th percentile of distance steps)
        if distances:
            threshold = float(np.percentile(distances, self.threshold_percentile))
        else:
            threshold = 0.5
            
        # 5. Group sentences based on threshold boundaries
        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_index = 0
        
        for idx, dist in enumerate(distances):
            if dist > threshold:
                # Merge current sentences and create chunk
                chunk_text = " ".join(current_chunk_sentences)
                char_start = text.find(current_chunk_sentences[0])
                last_sentence = current_chunk_sentences[-1]
                char_end = text.find(last_sentence, char_start) + len(last_sentence)
                
                chunks.append({
                    'text': chunk_text,
                    'chunk_index': chunk_index,
                    'source': source,
                    'metadata': {
                        **metadata,
                        'char_start': char_start if char_start != -1 else 0,
                        'char_end': char_end if char_end != -1 else len(text)
                    }
                })
                chunk_index += 1
                current_chunk_sentences = [sentences[idx+1]]
            else:
                current_chunk_sentences.append(sentences[idx+1])
                
        # Append final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            char_start = text.find(current_chunk_sentences[0])
            last_sentence = current_chunk_sentences[-1]
            char_end = text.find(last_sentence, char_start) + len(last_sentence)
            chunks.append({
                'text': chunk_text,
                'chunk_index': chunk_index,
                'source': source,
                'metadata': {
                    **metadata,
                    'char_start': char_start if char_start != -1 else 0,
                    'char_end': char_end if char_end != -1 else len(text)
                }
            })
            
        return chunks

    def chunk_documents(self, documents: list[dict]) -> list[dict]:
        all_chunks = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        print(f"[CHUNK-SEMANTIC] Created {len(all_chunks)} chunks from {len(documents)} document(s)")
        return all_chunks
