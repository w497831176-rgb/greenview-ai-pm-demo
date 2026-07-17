"""
RAG indexing orchestrator: chunk + embed + store.
"""

import traceback
from typing import Optional

from db import property_db as db
import rag_chunking
import rag_embeddings
import rag_store


def index_document(doc_id: int, force: bool = False) -> bool:
    """Index a single knowledge document into the vector store."""
    doc = db.get_knowledge_doc(doc_id)
    if not doc:
        return False

    db.set_knowledge_doc_indexed(doc_id, "indexing", 0)

    try:
        content = doc.get("content", "") or ""
        if not content.strip():
            db.set_knowledge_doc_indexed(doc_id, "indexed", 0)
            if force:
                rag_store.delete_chunks_for_doc(doc_id)
            return True

        strategy = doc.get("split_strategy") or "auto"
        chunk_size = doc.get("chunk_size") or 512
        chunk_overlap = doc.get("chunk_overlap") or 64

        chunks = rag_chunking.split_text(
            content,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if force:
            rag_store.delete_chunks_for_doc(doc_id)

        if not chunks:
            db.set_knowledge_doc_indexed(doc_id, "indexed", 0)
            return True

        embeddings = rag_embeddings.embed_texts(chunks)
        # Ensure every chunk has a non-empty embedding; otherwise fall back to
        # deterministic hashes so retrieval still works.
        fallback = rag_embeddings._embed_fallback
        fixed_embeddings = []
        for idx, emb in enumerate(embeddings):
            if emb and len(emb) == rag_embeddings.VECTOR_DIM:
                fixed_embeddings.append(emb)
            else:
                fixed_embeddings.append(fallback(chunks[idx]))
        rag_store.add_chunks(doc_id, chunks, fixed_embeddings)
        db.set_knowledge_doc_indexed(doc_id, "indexed", len(chunks))
        return True
    except Exception:
        traceback.print_exc()
        db.set_knowledge_doc_indexed(doc_id, "failed", 0)
        return False


def reindex_document(doc_id: int) -> bool:
    return index_document(doc_id, force=True)


def _effective_threshold(requested: Optional[float]) -> float:
    """Use a lower threshold when the offline fallback embedding is active."""
    if requested is not None:
        return requested
    # Fallback embeddings are much less semantically dense than transformer models;
    # lower the bar so RAG still triggers in offline NAS environments.
    if rag_embeddings._should_use_fallback():
        return 0.25
    return 0.55


def semantic_search(query: str, top_k: int = 3, threshold: Optional[float] = None):
    """Search across indexed documents and return doc-shaped results.

    The frontend renders search results with the same card component used for
    listing documents, so we map vector matches back to document-level fields
    (id, title, category, is_indexed, chunk_count) and expose the top chunk
    score/content for display.
    """
    if not query.strip():
        return []
    threshold = _effective_threshold(threshold)
    query_embedding = rag_embeddings.embed_text(query)
    results = rag_store.search_chunks(query_embedding, top_k=top_k, threshold=threshold)

    enriched = []
    seen_doc_ids = set()
    for r in results:
        doc_id = r.get("doc_id")
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        doc = db.get_knowledge_doc(doc_id)
        if not doc:
            continue
        enriched.append({
            # Document-level fields for card rendering.
            "id": doc.get("id"),
            "title": doc.get("title"),
            "category": doc.get("category"),
            "is_indexed": bool(doc.get("is_indexed")),
            "chunk_count": doc.get("chunk_count") or 0,
            # Match-level fields for RAG context/citations.
            "doc_id": doc_id,
            "doc_title": doc.get("title"),
            "doc_category": doc.get("category"),
            "content": r.get("content", ""),
            "chunk_index": r.get("chunk_index"),
            "score": r.get("score", 0),
        })
    return enriched
