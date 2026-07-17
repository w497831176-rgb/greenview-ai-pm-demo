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
    """Tokenize Chinese/English/number text into searchable terms.

    The tokenizer splits on non-alphanumeric/CJK characters, and also inserts
    boundaries between Chinese characters and alphanumeric runs so queries like
    "17号充电区" match documents containing "17 号集中充电区".
    """
    text = text.lower()
    # Insert spaces between Chinese chars and alphanumeric runs.
    text = re.sub(
        r"(?<=[\u4e00-\u9fa5])(?=[a-z0-9])|(?<=[a-z0-9])(?=[\u4e00-\u9fa5])",
        " ",
        text,
    )
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


def _context_relevance_score(query: str, content: str) -> float:
    """Compute a query-centric lexical overlap score in [0, 1].

    Uses the same n-gram tokenizer as the keyword/rerank pipeline so the score
    reflects whether the content actually covers the query's phrases. A
    threshold of 0.2 filters one-off keyword matches while preserving docs that
    address the question.
    """
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    content_tokens = set(_tokenize(content))
    overlap = len(query_tokens & content_tokens)
    return round(overlap / len(query_tokens), 4)


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
    """BM25-style keyword search over knowledge chunks.

    Documents are first matched by their full title+content TF index, then
    expanded into their indexed chunks so every result points to an exact
    chunk.  This keeps citations chunk-accurate instead of falling back to
    the document's first chunk.
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
    scored_docs = []
    for doc_id, tf in index.items():
        doc = doc_map.get(doc_id)
        if not doc or not doc.get("is_indexed"):
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
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Expand top matching documents into their indexed chunks and re-rank by chunk.
    chunk_scores: List[tuple] = []
    all_chunks: List[Dict[str, Any]] = []
    for _, doc in scored_docs[:top_k]:
        chunks = rag_store.list_chunks_for_doc(doc["id"])
        if not chunks:
            continue
        title_tokens = _tokenize(doc.get("title") or "")
        for chunk in chunks:
            chunk_tokens = _tokenize(chunk.get("content", ""))
            # Inject title terms into each chunk so the document title still contributes.
            tokens = title_tokens + chunk_tokens
            tf_chunk: Dict[str, int] = {}
            for t in tokens:
                tf_chunk[t] = tf_chunk.get(t, 0) + 1
            all_chunks.append({"chunk": chunk, "doc": doc, "tf": tf_chunk, "len": len(tokens)})

    if not all_chunks:
        return []

    avg_chunk_len = sum(c["len"] for c in all_chunks) / len(all_chunks) or 1
    for item in all_chunks:
        tf_chunk = item["tf"]
        chunk_len = item["len"] or 1
        score = 0.0
        for t in query_tokens:
            f = tf_chunk.get(t, 0)
            if f == 0:
                continue
            idf = math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
            denom = f + 1.5 * (1 - 0.75 + 0.75 * (chunk_len / avg_chunk_len))
            score += idf * (f * 2.5) / denom
        if score > 0:
            chunk_scores.append((score, item))

    chunk_scores.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, item in chunk_scores[:top_k]:
        doc = item["doc"]
        chunk = item["chunk"]
        results.append({
            "id": doc["id"],
            "title": doc.get("title"),
            "category": doc.get("category"),
            "doc_id": doc["id"],
            "doc_title": doc.get("title"),
            "doc_category": doc.get("category"),
            "chunk_index": chunk.get("chunk_index"),
            "content": chunk.get("content", ""),
            "score": round(score, 4),
            "source": "keyword",
        })
    return results


def _semantic_search(query: str, top_k: int = 10, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Semantic search returning chunk-level results.

    Unlike the indexer wrapper, this does not deduplicate by document so that
    multiple relevant chunks from the same document can surface as independent
    evidence.
    """
    effective_threshold = rag_indexer._effective_threshold(threshold)
    query_embedding = rag_embeddings.embed_text(query)
    chunks = rag_store.search_chunks(query_embedding, top_k=top_k, threshold=effective_threshold)
    results = []
    for chunk in chunks:
        doc = db.get_knowledge_doc(chunk.get("doc_id"))
        if not doc or not doc.get("is_indexed"):
            continue
        results.append({
            "id": doc["id"],
            "title": doc.get("title"),
            "category": doc.get("category"),
            "doc_id": doc["id"],
            "doc_title": doc.get("title"),
            "doc_category": doc.get("category"),
            "chunk_index": chunk.get("chunk_index"),
            "content": chunk.get("content", ""),
            "score": chunk.get("score", 0),
            "is_indexed": bool(doc.get("is_indexed")),
            "source": "semantic",
        })
    return results


def _rrf_fusion(
    keyword_results: List[Dict[str, Any]],
    semantic_results: List[Dict[str, Any]],
    k: int = 60,
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion between keyword and semantic chunk rankings.

    Fusion key is (doc_id, chunk_index) so each chunk is an independent
    candidate and citations stay chunk-accurate.
    """
    scores: Dict[tuple, float] = {}
    details: Dict[tuple, Dict[str, Any]] = {}

    for rank, r in enumerate(keyword_results, start=1):
        key = (r.get("doc_id"), r.get("chunk_index"))
        scores[key] = scores.get(key, 0.0) + keyword_weight * (1.0 / (k + rank))
        details[key] = r

    for rank, r in enumerate(semantic_results, start=1):
        key = (r.get("doc_id"), r.get("chunk_index"))
        scores[key] = scores.get(key, 0.0) + semantic_weight * (1.0 / (k + rank))
        details[key] = r

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for key, score in ranked:
        r = dict(details[key])
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


def _extract_quoted_titles(query: str) -> List[str]:
    """Extract document titles wrapped in 《》 from the user query."""
    return re.findall(r'《([^《》]+)》', query)


def _extract_sub_queries(query: str) -> List[str]:
    """Split a long composite query into candidate sub-questions.

    Keeps whole lines and sentence-level segments ending with ？ or ? so that
    focused sub-questions (e.g. "家里漏水了怎么办？") can be retrieved
    independently.  Short fragments are ignored.
    """
    segments = []
    for line in query.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on Chinese or ASCII question marks while keeping the marker.
        parts = re.split(r'(?<=[？?])', line)
        for part in parts:
            part = part.strip()
            if len(part) >= 5:
                segments.append(part)
        # If the line had no question mark, keep it as a standalone sub-query.
        if not re.search(r'[？?]', line) and len(line) >= 5:
            segments.append(line)
    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for s in segments:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _single_query_search(
    query: str,
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run keyword + semantic + RRF + rerank + context filter for one query.

    The context filter is applied against this specific query so that a short
    sub-question ("家里漏水了怎么办？") is not penalised by tokens from the
    broader composite prompt.
    """
    top_k = settings.get("top_k", 5)
    keyword_weight = settings.get("keyword_weight", 0.3)
    semantic_weight = settings.get("semantic_weight", 0.7)
    rrf_k = settings.get("rrf_k", 60)
    enable_rerank = settings.get("enable_rerank", False)
    rerank_model = settings.get("rerank_model")
    score_threshold = settings.get("score_threshold", 0.0)
    context_threshold = settings.get("context_threshold", 0.2)

    keyword_results = _keyword_search(query, top_k=top_k * 2)
    semantic_results = _semantic_search(query, top_k=top_k * 2, threshold=score_threshold)
    semantic_results = [r for r in semantic_results if r.get("is_indexed")]

    fused = _rrf_fusion(
        keyword_results,
        semantic_results,
        k=rrf_k,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
    )

    if enable_rerank:
        fused = _rerank_results(query, fused, model_name=rerank_model)

    grounded = []
    for r in fused:
        if enable_rerank:
            ctx_score = r.get("rerank_score", 0.0)
            r["score"] = round(ctx_score, 4)
        else:
            ctx_text = f"{r.get('title', '')} {r.get('content', '')}"
            ctx_score = _context_relevance_score(query, ctx_text)
            r["score"] = round(ctx_score, 4)
        r["context_score"] = round(ctx_score, 4)
        if ctx_score >= context_threshold:
            grounded.append(r)

    return grounded[:top_k]


def _title_boosted_results(
    query: str,
    title: str,
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Retrieve chunks from documents whose titles are explicitly cited.

    Uses keyword search to locate the cited document(s), then scores every
    chunk of those documents against the full user query with the context
    relevance metric.  This prevents title-only chunks from drowning out
    content chunks (e.g. FAQ Q1) when the user asks a composite question.
    """
    top_k = settings.get("top_k", 5)
    context_threshold = settings.get("context_threshold", 0.2)

    # Find candidate docs by matching the cited title.
    title_hits = _keyword_search(title, top_k=top_k * 2)
    doc_ids = {r.get("doc_id") for r in title_hits if r.get("doc_id")}

    candidates = []
    for doc_id in doc_ids:
        doc = db.get_knowledge_doc(doc_id)
        if not doc:
            continue
        doc_title = doc.get("title") or ""
        chunks = rag_store.list_chunks_for_doc(doc_id)
        for c in chunks:
            content = c.get("content", "")
            ctx_text = f"{doc_title} {content}"
            ctx_score = _context_relevance_score(query, ctx_text)
            if ctx_score < context_threshold:
                continue
            candidates.append({
                "id": doc_id,
                "title": doc_title,
                "category": doc.get("category"),
                "doc_id": doc_id,
                "doc_title": doc_title,
                "doc_category": doc.get("category"),
                "chunk_index": c.get("chunk_index"),
                "content": content,
                "score": round(ctx_score, 4),
                "context_score": round(ctx_score, 4),
                "source": "title_boost",
            })

    candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
    # Keep a small number of high-confidence chunks per cited title.
    return candidates[:3]


def advanced_search(
    query: str,
    settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run composite-aware advanced RAG.

    In addition to the full user query, the pipeline:
      1. Searches for any document titles explicitly cited in 《》.
      2. Searches focused sub-questions (sentences ending with ？/?).
      3. Merges and deduplicates candidates by exact chunk.

    Only when none of the retrieval paths produce evidence does the caller see
    an empty result set (and may create an auto knowledge badcase).
    """
    settings = settings or {}
    top_k = settings.get("top_k", 5)

    merged: Dict[tuple, Dict[str, Any]] = {}

    def _add_results(results: List[Dict[str, Any]]) -> None:
        for r in results:
            key = (r.get("doc_id"), r.get("chunk_index"))
            existing = merged.get(key)
            if existing is None or r.get("score", 0) > existing.get("score", 0):
                merged[key] = r

    # Primary retrieval path: the full user query.
    original_results = _single_query_search(query, settings)
    _add_results(original_results)

    # Secondary path: exact titles cited in 《》 (e.g. 《常见维修问题 FAQ》).
    for title in _extract_quoted_titles(query):
        _add_results(_title_boosted_results(query, title, settings))

    # Tertiary path: focused sub-questions inside the composite prompt.
    for sub_query in _extract_sub_queries(query):
        if sub_query == query:
            continue
        _add_results(_single_query_search(sub_query, settings))

    results = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # Keep simple diagnostics based on the original query so callers can still
    # compare the raw full-query pipeline against the composite result set.
    keyword_results = _keyword_search(query, top_k=top_k * 2)
    semantic_results = [r for r in _semantic_search(query, top_k=top_k * 2, threshold=settings.get("score_threshold", 0.0)) if r.get("is_indexed")]

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
    context_threshold: Optional[float] = None,
    expected_doc_title: Optional[str] = None,
) -> Dict[str, Any]:
    """Debug endpoint exposing each retrieval stage."""
    score_threshold = score_threshold or 0.0
    context_threshold = context_threshold if context_threshold is not None else 0.2
    settings = {
        "top_k": top_k,
        "keyword_weight": keyword_weight,
        "semantic_weight": semantic_weight,
        "rrf_k": rrf_k,
        "enable_rerank": enable_rerank,
        "rerank_model": rerank_model,
        "score_threshold": score_threshold,
        "context_threshold": context_threshold,
    }
    keyword_results = _keyword_search(query, top_k=top_k * 2)
    semantic_results = _semantic_search(query, top_k=top_k * 2, threshold=score_threshold)
    # Expose unfiltered semantic candidates in debug, but flag those that would
    # be removed by the business-index gate.
    semantic_results_debug = []
    for r in semantic_results:
        r_debug = dict(r)
        r_debug["_filtered_not_indexed"] = not r.get("is_indexed")
        semantic_results_debug.append(r_debug)
    semantic_results_indexed = [r for r in semantic_results if r.get("is_indexed")]

    fused = _rrf_fusion(
        keyword_results,
        semantic_results_indexed,
        k=rrf_k,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
    )
    if enable_rerank:
        fused = _rerank_results(query, fused, model_name=rerank_model)

    # Apply the same grounded-evidence filter used in production.
    grounded = []
    for r in fused:
        if enable_rerank:
            ctx_score = r.get("rerank_score", 0.0)
        else:
            ctx_text = f"{r.get('title', '')} {r.get('content', '')}"
            ctx_score = _context_relevance_score(query, ctx_text)
        r["context_score"] = ctx_score
        if ctx_score >= context_threshold:
            grounded.append(r)

    # The original-query stages are exposed for diagnostics, but the final
    # production result set comes from the composite-aware advanced_search so
    # that title/sub-question evidence is preserved.
    composite = advanced_search(query, settings=settings)
    results = composite.get("results", [])

    titles = [r.get("title") for r in results]
    top1_hit = bool(expected_doc_title and titles and titles[0] == expected_doc_title)
    top3_hit = bool(expected_doc_title and expected_doc_title in titles)

    return {
        "query": query,
        "settings": settings,
        "keyword_results": keyword_results,
        "semantic_results": semantic_results_debug,
        "fused_results": fused,
        "results": results,
        "count": len(results),
        "expected_doc_title": expected_doc_title,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "composite_debug": {
            "quoted_titles": _extract_quoted_titles(query),
            "sub_queries": _extract_sub_queries(query),
            "merged_count": composite.get("count", 0),
        },
    }
