import os
import json
import numpy as np
import psycopg
from pgvector.psycopg import register_vector


class VectorStore:
    """PostgreSQL + pgvector backed vector store for production RAG.
    
    Stores document chunks and their embeddings in a PostgreSQL table
    using the pgvector extension for efficient cosine similarity search.
    """

    TABLE_NAME = "document_chunks"
    VECTOR_DIM = 384  # Matches all-MiniLM-L6-v2 output dimension

    def __init__(self, db_config: dict = None):
        """Initialize the PostgreSQL vector store.
        
        Args:
            db_config: Dict with keys: host, port, dbname, user, password.
                       Falls back to environment variables if not provided.
        """
        self.db_config = db_config or self._load_config_from_env()
        self._conn_string = self._build_conn_string()
        self.chunks = []  # Local cache for pipeline compatibility
        
        # Initialize database schema
        self._init_db()
        print(f"[VECTORSTORE] Connected to PostgreSQL @ {self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}")

    def _load_config_from_env(self) -> dict:
        """Load database configuration from environment variables."""
        return {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT')),
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
        }

    def _build_conn_string(self) -> str:
        """Build a psycopg connection string from config."""
        c = self.db_config
        return f"host={c['host']} port={c['port']} dbname={c['dbname']} user={c['user']} password={c['password']}"

    def _get_connection(self):
        """Create a new database connection with pgvector registered."""
        conn = psycopg.connect(self._conn_string, autocommit=True)
        register_vector(conn)
        return conn

    def _get_raw_connection(self):
        """Create a raw connection without pgvector registration (for bootstrap)."""
        return psycopg.connect(self._conn_string, autocommit=True)

    def _init_db(self):
        """Create the pgvector extension and document_chunks table if they don't exist."""
        # Step 1: Enable pgvector extension using a raw connection (before type is registered)
        with self._get_raw_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Step 2: Now that extension exists, use a registered connection for table creation
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Create table for storing chunks + embeddings
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        source TEXT,
                        chunk_index INTEGER,
                        metadata JSONB DEFAULT '{{}}'::jsonb,
                        embedding vector({self.VECTOR_DIM})
                    )
                """)
        
        # Load existing chunks count
        self._sync_local_cache()

    def _sync_local_cache(self):
        """Sync local chunks list from database for pipeline compatibility."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT text, source, chunk_index, metadata FROM {self.TABLE_NAME}")
                rows = cur.fetchall()
                self.chunks = [
                    {
                        'text': row[0],
                        'source': row[1],
                        'chunk_index': row[2],
                        'metadata': row[3] if row[3] else {}
                    }
                    for row in rows
                ]

    def add(self, chunks: list[dict], vectors: np.ndarray) -> None:
        """Insert chunks and their embedding vectors into PostgreSQL.
        
        Args:
            chunks: List of chunk dicts with 'text', 'source', 'chunk_index', 'metadata'.
            vectors: numpy array of shape (N, VECTOR_DIM).
        """
        assert len(chunks) == len(vectors), "Chunks and vectors must have the same length"
        
        if len(chunks) == 0:
            return

        vectors = np.array(vectors, dtype=np.float32)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Batch insert using executemany
                insert_sql = f"""
                    INSERT INTO {self.TABLE_NAME} (text, source, chunk_index, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                """
                
                rows = []
                for chunk, vec in zip(chunks, vectors):
                    rows.append((
                        chunk.get('text', ''),
                        chunk.get('source', ''),
                        chunk.get('chunk_index', 0),
                        json.dumps(chunk.get('metadata', {})),
                        vec.tolist(),
                    ))
                
                cur.executemany(insert_sql, rows)

        # Update local cache
        self.chunks.extend(chunks)
        print(f"[VECTORSTORE] Inserted {len(chunks)} chunks into PostgreSQL. Total: {len(self.chunks)}")

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Perform cosine similarity search using pgvector's <=> operator.
        
        Args:
            query_vector: The query embedding vector (1D numpy array).
            top_k: Number of top results to return.
            
        Returns:
            List of dicts with 'chunk', 'score', and 'rank' keys.
        """
        if query_vector.ndim > 1:
            query_vector = query_vector.flatten()

        query_list = query_vector.astype(np.float32).tolist()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # pgvector <=> is cosine distance; similarity = 1 - distance
                cur.execute(f"""
                    SELECT text, source, chunk_index, metadata,
                           1 - (embedding <=> %s::vector) AS similarity
                    FROM {self.TABLE_NAME}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (str(query_list), str(query_list), top_k))
                
                rows = cur.fetchall()

        results = []
        for rank, row in enumerate(rows):
            results.append({
                'chunk': {
                    'text': row[0],
                    'source': row[1],
                    'chunk_index': row[2],
                    'metadata': row[3] if row[3] else {}
                },
                'score': float(row[4]) if row[4] else 0.0,
                'rank': rank + 1
            })

        return results

    def clear(self) -> None:
        """Delete all chunks from the database table."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.TABLE_NAME}")
        self.chunks = []
        print("[VECTORSTORE] Cleared all chunks from PostgreSQL")

    def count(self) -> int:
        """Return the total number of chunks in the database."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.TABLE_NAME}")
                return cur.fetchone()[0]

    # ---- Legacy compatibility methods (no-op for PostgreSQL) ----
    
    def save(self, path: str) -> None:
        """No-op: Data is already persisted in PostgreSQL."""
        count = self.count()
        print(f"[VECTORSTORE] Data already persisted in PostgreSQL ({count} chunks). Skipping file save.")

    def load(self, path: str) -> None:
        """Load chunks from PostgreSQL (ignores file path)."""
        self._sync_local_cache()
        print(f"[VECTORSTORE] Loaded {len(self.chunks)} chunks from PostgreSQL")
