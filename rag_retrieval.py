"""
Advanced RAG retrieval pipeline.

Supports keyword search, semantic search, RRF fusion, optional cross-encoder
reranking, and score threshold filtering.
"""

import json
import math
import re
from typing import Any, Dict, List, Optional

from db import property_db as db
import rag_embeddings
import rag_indexer
import rag_store


def _tokenize(text: str) -> List[str]:
    """Tokenize Chinese/English text into searchable terms."""
    text = text.lower()
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", " ", text)
    tokens = []
    for word in text.split():
        if word:
            tokens.append(word)
            # Add character n-grams for Chinese words.
            if re.match(r"^[\u4e00-\u9fa5]+$", word):
                for n in (2, 3, 4):
                    if len(word) >= n:
                        for i in range(len(word) - n + 1):
                            tokens.append(word[i:i + n])
    return tokens


def _build_keyword_index() -> Dict[int, Dict[str, int]]:
    """Build in-memory TF index for knowledge docs."""
    docs = db.list_knowledge_docs()
    index: Dict[int, Dict[str, int]] = {}
    for doc in docs:
        if not doc.get("is_indexed"):
            continue
        text = f"{doc.get('title', '')} {doc.get('content', '')}"
        tokens = _tokenize(text)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        index[doc["id"]] = tf
    return index


def _keyword_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25-style keyword search over knowledge documents.

    Uses a simple TF/IDF scoring so it works without external FTS engines.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    index = _build_keyword_index()
    doc_list = db.list_knowledge_docs()
    doc_map = {d["id"]: d for d in doc_list}

    # Document frequency for each query token.
    df: Dict[str, int] = {}
    for tf in index.values():
        for t in query_tokens:
            if tf.get(t):
                df[t] = df.get(t, 0) + 1

    N = len(index) or 1
    scored = []
    for doc_id, tf in index.items():
        doc = doc_map.get(doc_id)
        if not doc:
            continue
        doc_len = sum(tf.values()) or 1
        score = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
            # Simplified BM25 term score.
            k1 = 1.5
            b = 0.75
            avg_len = sum(sum(t.values()) for t in index.values()) / N if N > 0 else doc_len
            denom = f + k1 * (1 - b + b * (doc_len / avg_len))
            score += idf * (f * (k1 + 1)) / denom
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, doc in scored[:top_k]:
        results.append({
            "id": doc["id"],
            "title": doc.get("title"),
            "category": doc.get("category"),
            "doc_id": doc["id"],
            "doc_title": doc.get("title"),
            "doc_category": doc.get("category"),
            "chunk_index": None,
            "content": (doc.get("content") or "")[:800],
            "score": round(score, 4),
            "source": "keyword",
        })
    return results


def _semantic_search(query: str, top_k: int = 10, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Semantic search wrapper returning doc-level results."""
    results = rag_indexer.semantic_search(query, top_k=top_k, threshold=threshold)
    for r in results:
        r["source"] = "semantic"
    return results


def _rrf_fusion(
    keyword_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: int = 60,
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion between keyword and semantic rankings."""
    scores: Dict[int, float] = {}
    details: Dict[int, Dict[str, Any]] = {}

    for rank, r in enumerate(keyword_results, start=1):
        doc_id = r["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + keyword_weight * (1.0 / (k + rank))
        details[doc_id] = r

    for rank, r in enumerate(semantic_results, start=1):
        doc_id = r["id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + semantic_weight * (1.0 / (k + rank))
        details[doc_id] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for doc_id, score in ranked:
        r = dict(details[doc_id])
        r["rrf_score"] = round(score, 4)
        r["source"] = "fusion"
        results.append(r)
    return results


def _rerank_results(query: str, results: List[Dict[str, Any]], model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Optional cross-encoder reranking.

    If a cross-encoder model is available locally, use it; otherwise fall back
    to a lightweight lexical overlap score so the pipeline still works offline.
    """
    if not results:
        return results

    cross_encoder = None
    if model_name:
        try:
            from sentence_transformers import CrossEncoder

            cross_encoder = CrossEncoder(model_name)
        except Exception:
            cross_encoder = None

    if cross_encoder is not None:
        pairs = [[query, r.get("content", "")] for r in results]
        scores = cross_encoder.predict(pairs)
        for r, score in zip(results, scores):
            r["rerank_score"] = round(float(score), 4)
        results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return results

    # Fallback: token overlap reranker.
    query_tokens = set(_tokenize(query))
    for r in results:
        content_tokens = set(_tokenize(r.get("content", "")))
        overlap = len(query_tokens & content_tokens)
        r["rerank_score"] = round(overlap / max(len(query_tokens), 1), 4)
    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return results


def advanced_search(
    query: str,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the full advanced RAG pipeline."""
    settings = settings or {}
    top_k = settings.get("top_k", 5)
    keyword_weight = settings.get("keyword_weight", 0.3)
    semantic_weight = settings.get("semantic_weight", 0.7)
    rrf_k = settings.get("rrf_k", 60)
    enable_rerank = settings.get("enable_rerank", False)
    rerank_model = settings.get("rerank_model")
    score_threshold = settings.get("score_threshold", 0.0)

    keyword_results = _keyword_search(query, top_k=top_k * 2)
    # Apply the configured similarity threshold to semantic matches before
    # fusing, so the threshold filters real semantic relevance rather than
    # the tiny RRF scores produced by rank fusion.
    semantic_results = _semantic_search(query, top_k=top_k * 2, threshold=score_threshold)

    fused = _rrf_fusion(
        keyword_results,
        semantic_results,
        k=rrf_k,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
    )

    if enable_rerank:
        fused = _rerank_results(query, fused, model_name=rerank_model)

    results = fused[:top_k]

    return {
        "query": query,
        "mode": "advanced",
        "settings": settings,
        "keyword_count": len(keyword_results),
        "semantic_count": len(semantic_results),
        "count": len(results),
        "results": results,
    }


def debug_search(
    query: str,
    top_k: int = 5,
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
    rrf_k: int = 60,
    enable_rerank: bool = False,
    rerank_model: Optional[str] = None,
    score_threshold: Optional[float] = None,
    expected_doc_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Debug endpoint exposing each retrieval stage."""
    settings = {
        "top_k": top_k,
        "keyword_weight": keyword_weight,
        "semantic_weight": semantic_weight,
        "rrf_k": rrf_k,
        "enable_rerank": enable_rerank,
        "rerank_model": rerank_model,
        "score_threshold": score_threshold or 0.0,
    }
    keyword_results = _keyword_search(query, top_k=top_k * 2)
    semantic_results = _semantic_search(query, top_k=top_k * 2, threshold=score_threshold)
    fused = _rrf_fusion(
        keyword_results,
        semantic_results,
        k=rrf_k,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
    )
    if enable_rerank:
        fused = _rerank_results(query, fused, model_name=rerank_model)

    results = fused[:top_k]

    titles = [r.get("title") for r in results]
    top1_hit = bool(expected_doc_title and titles and titles[0] == expected_doc_title)
    top3_hit = bool(expected_doc_title and expected_doc_title in titles)

    return {
        "query": query,
        "settings": settings,
        "keyword_results": keyword_results,
        "semantic_results": semantic_results,
        "fused_results": fused,
        "results": results,
        "count": len(results),
        "expected_doc_title": expected_doc_title,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
    }
