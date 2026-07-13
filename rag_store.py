"""
Vector store backed by pgvector for RAG chunks.
"""

import json
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

VECTOR_DIM = 512


def _get_dsn() -> str:
    host = os.getenv("DB_HOST", "demo-os-db")
    port = os.getenv("DB_PORT", "5432")
    user = os.getenv("DB_USER", "ai")
    password = os.getenv("DB_PASS", "ai")
    db = os.getenv("DB_DATABASE", "ai")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@contextmanager
def _get_conn():
    conn = psycopg.connect(_get_dsn(), row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def init_vector_store():
    """Ensure pgvector extension and chunks table exist."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id SERIAL PRIMARY KEY,
                    doc_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({VECTOR_DIM}),
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(doc_id, chunk_index)
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_doc_id ON knowledge_chunks(doc_id)"
            )
            # Replace any existing approximate index with HNSW for better recall.
            cur.execute("DROP INDEX IF EXISTS idx_knowledge_chunks_embedding")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        conn.commit()


def add_chunks(doc_id: int, chunks: List[str], embeddings: Optional[List[List[float]]] = None) -> int:
    """Replace existing chunks for a doc and insert new ones."""
    delete_chunks_for_doc(doc_id)
    if not chunks:
        return 0

    with _get_conn() as conn:
        with conn.cursor() as cur:
            for idx, content in enumerate(chunks):
                emb = embeddings[idx] if embeddings and idx < len(embeddings) else None
                if emb:
                    cur.execute(
                        "INSERT INTO knowledge_chunks (doc_id, chunk_index, content, embedding) VALUES (%s, %s, %s, %s)",
                        (doc_id, idx, content, emb),
                    )
                else:
                    cur.execute(
                        "INSERT INTO knowledge_chunks (doc_id, chunk_index, content) VALUES (%s, %s, %s)",
                        (doc_id, idx, content),
                    )
        conn.commit()
    return len(chunks)


def delete_chunks_for_doc(doc_id: int):
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_chunks WHERE doc_id = %s", (doc_id,))
        conn.commit()


def search_chunks(query_embedding: List[float], top_k: int = 3, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Semantic search using cosine similarity."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    kc.id,
                    kc.doc_id,
                    kc.chunk_index,
                    kc.content,
                    1 - (kc.embedding <=> %s::vector) AS score
                FROM knowledge_chunks kc
                WHERE kc.embedding IS NOT NULL
                ORDER BY kc.embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    results = []
    for row in rows:
        score = float(row["score"])
        if score >= threshold:
            results.append({
                "id": row["id"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "score": round(score, 4),
            })
    return results


def list_chunks_for_doc(doc_id: int) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, doc_id, chunk_index, content FROM knowledge_chunks WHERE doc_id = %s ORDER BY chunk_index",
                (doc_id,),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]
