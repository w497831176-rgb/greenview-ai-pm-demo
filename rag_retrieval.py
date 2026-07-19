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


def _is_structural_chunk(content: str, doc_title: str = "") -> bool:
    """Return True for title-only and heading-only chunks.

    These fragments may help index navigation, but they cannot support an
    owner-facing factual answer or a clickable citation on their own.
    """
    normalized = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", (content or "").lower())
    title_normalized = re.sub(r"[^\u4e00-\u9fa5a-z0-9]", "", (doc_title or "").lower())
    if not normalized:
        return True
    if title_normalized and (
        normalized == title_normalized
        or (normalized.endswith(title_normalized) and len(normalized) - len(title_normalized) <= 8)
        or (title_normalized.endswith(normalized) and len(title_normalized) - len(normalized) <= 8)
    ):
        return True
    return bool(re.fullmatch(r"第[0-9一二三四五六七八九十百]+[章节].{0,24}|[总附]则", normalized))


def _explicit_evidence_priority(query: str, content: str) -> float:
    """Rank supporting chunks for an explicitly named source.

    It is a transparent property-domain evidence policy, not a document-name
    rule: owner symptoms and service-time intents must outrank a generic
    complaint or title sentence when the owner asks for repair guidance.
    """
    q = query or ""
    text = content or ""
    score = _context_relevance_score(q, text)
    water_query = any(term in q for term in ("漏水", "滴水", "渗水"))
    water_evidence = any(term in text for term in ("漏水", "滴水", "渗水", "水管", "水阀"))
    timing_query = any(term in q for term in ("时效", "响应", "多久", "到场", "上门", "紧急"))
    timing_evidence = any(term in text for term in ("紧急维修", "一般维修", "到场", "上门", "派单", "受理"))
    if water_query and water_evidence:
        score += 0.30
    if timing_query and timing_evidence:
        score += 0.18
    if "投诉" in text and not any(term in q for term in ("投诉", "纠纷", "12345")):
        score -= 0.50
    return round(max(0.0, min(1.0, score)), 4)


def _build_keyword_index(
    allowed_document_ids: Optional[set[int]] = None,
) -> Dict[int, Dict[str, int]]:
    """Build in-memory TF index for knowledge docs."""
    docs = db.list_knowledge_docs()
    index: Dict[int, Dict[str, int]] = {}
    for doc in docs:
        if not doc.get("is_indexed"):
            continue
        if allowed_document_ids is not None and int(doc["id"]) not in allowed_document_ids:
            continue
        text = f"{doc.get('title', '')} {doc.get('content', '')}"
        tokens = _tokenize(text)
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        index[doc["id"]] = tf
    return index


def _keyword_search(
    query: str,
    top_k: int = 10,
    threshold: Optional[float] = None,
    allowed_document_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """BM25-style keyword search over knowledge chunks.

    Documents are first matched by their full title+content TF index, then
    expanded into their indexed chunks so every result points to an exact
    chunk.  This keeps citations chunk-accurate instead of falling back to
    the document's first chunk.

    ``threshold`` is accepted for API compatibility but currently ignored;
    keyword matches always require a positive score.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    index = _build_keyword_index(allowed_document_ids=allowed_document_ids)
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
            if _is_structural_chunk(chunk.get("content", ""), doc.get("title") or ""):
                continue
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


def _semantic_search(
    query: str,
    top_k: int = 10,
    threshold: Optional[float] = None,
    allowed_document_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """Semantic search returning chunk-level results.

    Unlike the indexer wrapper, this does not deduplicate by document so that
    multiple relevant chunks from the same document can surface as independent
    evidence.
    """
    effective_threshold = rag_indexer._effective_threshold(threshold)
    query_embedding = rag_embeddings.embed_text(query)
    chunks = rag_store.search_chunks(
        query_embedding,
        top_k=top_k,
        threshold=effective_threshold,
        allowed_document_ids=allowed_document_ids,
    )
    results = []
    for chunk in chunks:
        doc = db.get_knowledge_doc(chunk.get("doc_id"))
        if not doc or not doc.get("is_indexed"):
            continue
        if _is_structural_chunk(chunk.get("content", ""), doc.get("title") or ""):
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
    """Fuse exact evidence chunks and preserve why each candidate survived.

    A result is keyed by (doc_id, chunk_index). Unlike document-level fusion,
    this keeps a clickable citation tied to the exact supporting text. The
    returned provenance is retained for the debug page and Trace.
    """
    scores: Dict[tuple, float] = {}
    details: Dict[tuple, Dict[str, Any]] = {}

    def add(result: Dict[str, Any], rank: int, channel: str, weight: float) -> None:
        key = (result.get("doc_id"), result.get("chunk_index"))
        scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank))
        item = details.get(key)
        if item is None:
            item = dict(result)
            item["retrieval_sources"] = []
            details[key] = item
        if channel not in item["retrieval_sources"]:
            item["retrieval_sources"].append(channel)
        item[f"{channel}_rank"] = rank
        item[f"{channel}_score"] = result.get("score")

    for rank, result in enumerate(keyword_results, start=1):
        add(result, rank, "keyword", keyword_weight)
    for rank, result in enumerate(semantic_results, start=1):
        add(result, rank, "semantic", semantic_weight)

    results = []
    for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        item = dict(details[key])
        item["rrf_score"] = round(score, 6)
        item["source"] = "fusion"
        results.append(item)
    return results

def _rerank_results(query: str, results: List[Dict[str, Any]], model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Optionally rerank and explicitly disclose any offline fallback."""
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
        pairs = [[query, result.get("content", "")] for result in results]
        scores = cross_encoder.predict(pairs)
        for result, score in zip(results, scores):
            result["rerank_score"] = round(float(score), 4)
            result["rerank_mode"] = "cross_encoder"
        results.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
        return results

    query_tokens = set(_tokenize(query))
    for result in results:
        content_tokens = set(_tokenize(result.get("content", "")))
        overlap = len(query_tokens & content_tokens)
        result["rerank_score"] = round(overlap / max(len(query_tokens), 1), 4)
        result["rerank_mode"] = "lexical_fallback"
    results.sort(key=lambda item: item.get("rerank_score", 0), reverse=True)
    return results

def _extract_quoted_titles(query: str) -> List[str]:
    """Extract document titles wrapped in 《》 from the user query."""
    return re.findall(r'《([^《》]+)》', query)


def _extract_sub_queries(query: str) -> List[str]:
    """Extract only independently retrievable evidence sub-questions.

    Composite owner requests often include operational steps plus one or more
    explicitly grounded questions.  We retain the full query, and add numbered
    fragments only when they name a document or ask for knowledge evidence.
    This improves cited-chunk precision without multiplying every tool/query
    step into an expensive embedding search.
    """
    segments: List[str] = []
    for line in query.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) >= 5:
            segments.append(line)
        for part in re.split(r"(?:^|[\n：；;])\s*\d{1,2}[.、]\s*", line):
            part = part.strip()
            if len(part) < 5 or part == line:
                continue
            if "《" in part or "知识库" in part or "依据" in part:
                segments.append(part)
        for part in re.split(r'(?<=[？?])', line):
            part = part.strip()
            if len(part) >= 5:
                segments.append(part)
    seen = set()
    return [segment for segment in segments if not (segment in seen or seen.add(segment))]


DEFAULT_RETRIEVAL_SETTINGS = {
    "top_k": 5,
    "keyword_weight": 0.3,
    "semantic_weight": 0.7,
    "rrf_k": 60,
    "enable_rerank": False,
    "rerank_model": None,
    "score_threshold": 0.0,
    "context_threshold": 0.2,
}


def normalize_retrieval_settings(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Clamp and normalise the shared retrieval settings contract.

    The owner chat, search API and debug console call this helper so the values
    in the platform console are not silently replaced by different defaults.
    """
    result = dict(DEFAULT_RETRIEVAL_SETTINGS)
    result.update(settings or {})
    try:
        result["top_k"] = max(1, min(10, int(result["top_k"])))
    except (TypeError, ValueError):
        result["top_k"] = DEFAULT_RETRIEVAL_SETTINGS["top_k"]
    try:
        result["rrf_k"] = max(1, min(1000, int(result["rrf_k"])))
    except (TypeError, ValueError):
        result["rrf_k"] = DEFAULT_RETRIEVAL_SETTINGS["rrf_k"]
    for key in ("keyword_weight", "semantic_weight"):
        try:
            result[key] = max(0.0, float(result[key]))
        except (TypeError, ValueError):
            result[key] = DEFAULT_RETRIEVAL_SETTINGS[key]
    weight_sum = result["keyword_weight"] + result["semantic_weight"]
    if weight_sum <= 0:
        result["keyword_weight"] = DEFAULT_RETRIEVAL_SETTINGS["keyword_weight"]
        result["semantic_weight"] = DEFAULT_RETRIEVAL_SETTINGS["semantic_weight"]
    else:
        result["keyword_weight"] = round(result["keyword_weight"] / weight_sum, 4)
        result["semantic_weight"] = round(result["semantic_weight"] / weight_sum, 4)
    for key in ("score_threshold", "context_threshold"):
        try:
            result[key] = max(0.0, min(1.0, float(result[key])))
        except (TypeError, ValueError):
            result[key] = DEFAULT_RETRIEVAL_SETTINGS[key]
    result["enable_rerank"] = bool(result.get("enable_rerank"))
    return result


def _single_query_search(
    query: str,
    settings: Dict[str, Any],
    allowed_document_ids: Optional[set[int]] = None,
) -> List[Dict[str, Any]]:
    """Run hybrid retrieval and the same evidence gate used in production."""
    settings = normalize_retrieval_settings(settings)
    top_k = settings["top_k"]
    keyword_results = _keyword_search(
        query,
        top_k=top_k * 2,
        allowed_document_ids=allowed_document_ids,
    )
    semantic_results = _semantic_search(
        query,
        top_k=top_k * 2,
        threshold=settings["score_threshold"],
        allowed_document_ids=allowed_document_ids,
    )
    semantic_results = [result for result in semantic_results if result.get("is_indexed")]
    fused = _rrf_fusion(
        keyword_results,
        semantic_results,
        k=settings["rrf_k"],
        keyword_weight=settings["keyword_weight"],
        semantic_weight=settings["semantic_weight"],
    )
    if settings["enable_rerank"]:
        fused = _rerank_results(query, fused, model_name=settings["rerank_model"])

    grounded = []
    for result in fused:
        if settings["enable_rerank"]:
            context_score = result.get("rerank_score", 0.0)
            result["score"] = round(context_score, 4)
        else:
            context_score = _context_relevance_score(
                query, f"{result.get('title', '')} {result.get('content', '')}"
            )
            result["score"] = round(context_score, 4)
        result["context_score"] = round(context_score, 4)
        result["evidence_status"] = (
            "accepted"
            if context_score >= settings["context_threshold"]
            else "filtered_low_relevance"
        )
        if result["evidence_status"] == "accepted":
            grounded.append(result)
    return grounded[:top_k]

def _title_boosted_results(
    query: str,
    title: str,
    settings: Dict[str, Any],
    allowed_document_ids: Optional[set[int]] = None,
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
    title_hits = _keyword_search(
        title,
        top_k=top_k * 2,
        allowed_document_ids=allowed_document_ids,
    )
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
            if _is_structural_chunk(content, doc_title):
                continue
            ctx_score = _explicit_evidence_priority(query, content)
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
    allowed_document_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run composite-aware RAG inside an Agent's published document scope.

    Scope is applied in keyword indexing and vector SQL before either channel
    selects Top-K.  ``None`` preserves the global platform debug behavior;
    an empty list means the Agent has no RAG capability.
    """
    settings = normalize_retrieval_settings(settings)
    scope = None if allowed_document_ids is None else {int(item) for item in allowed_document_ids}
    if scope is not None and not scope:
        return {
            "query": query,
            "mode": "advanced",
            "settings": settings,
            "keyword_count": 0,
            "semantic_count": 0,
            "count": 0,
            "results": [],
            "embedding_runtime": rag_embeddings.get_runtime_info(),
            "scope": {"mode": "agent_bound", "allowed_document_ids": []},
            "evidence_policy": {
                "score_threshold": settings["score_threshold"],
                "context_threshold": settings["context_threshold"],
                "top_k": settings["top_k"],
            },
        }
    top_k = settings["top_k"]
    merged: Dict[tuple, Dict[str, Any]] = {}

    def add_results(results: List[Dict[str, Any]], path: str) -> None:
        for result in results:
            item = dict(result)
            item["retrieval_paths"] = list(
                dict.fromkeys(item.get("retrieval_paths", []) + [path])
            )
            key = (item.get("doc_id"), item.get("chunk_index"))
            current = merged.get(key)
            if current is None:
                merged[key] = item
                continue
            paths = list(
                dict.fromkeys(current.get("retrieval_paths", []) + item["retrieval_paths"])
            )
            sources = list(
                dict.fromkeys(current.get("retrieval_sources", []) + item.get("retrieval_sources", []))
            )
            if item.get("score", 0) > current.get("score", 0):
                item["retrieval_paths"] = paths
                if sources:
                    item["retrieval_sources"] = sources
                merged[key] = item
            else:
                current["retrieval_paths"] = paths
                if sources:
                    current["retrieval_sources"] = sources

    add_results(
        _single_query_search(query, settings, allowed_document_ids=scope),
        "full_query",
    )
    for title in _extract_quoted_titles(query):
        add_results(
            _title_boosted_results(
                query,
                title,
                settings,
                allowed_document_ids=scope,
            ),
            "quoted_title",
        )
    for sub_query in _extract_sub_queries(query):
        if sub_query != query:
            add_results(
                _single_query_search(
                    sub_query,
                    settings,
                    allowed_document_ids=scope,
                ),
                "sub_query",
            )

    results = sorted(merged.values(), key=lambda item: item.get("score", 0), reverse=True)[:top_k]
    keyword_results = _keyword_search(
        query,
        top_k=top_k * 2,
        allowed_document_ids=scope,
    )
    semantic_results = [
        result
        for result in _semantic_search(
            query,
            top_k=top_k * 2,
            threshold=settings["score_threshold"],
            allowed_document_ids=scope,
        )
        if result.get("is_indexed")
    ]
    return {
        "query": query,
        "mode": "advanced",
        "settings": settings,
        "keyword_count": len(keyword_results),
        "semantic_count": len(semantic_results),
        "count": len(results),
        "results": results,
        "embedding_runtime": rag_embeddings.get_runtime_info(),
        "scope": {
            "mode": "global_debug" if scope is None else "agent_bound",
            "allowed_document_ids": None if scope is None else sorted(scope),
        },
        "evidence_policy": {
            "score_threshold": settings["score_threshold"],
            "context_threshold": settings["context_threshold"],
            "top_k": settings["top_k"],
        },
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
    """Expose every retrieval stage without making candidates look like evidence."""
    settings = normalize_retrieval_settings({
        "top_k": top_k, "keyword_weight": keyword_weight, "semantic_weight": semantic_weight,
        "rrf_k": rrf_k, "enable_rerank": enable_rerank, "rerank_model": rerank_model,
        "score_threshold": 0.0 if score_threshold is None else score_threshold,
        "context_threshold": 0.2 if context_threshold is None else context_threshold,
    })
    keyword_results = _keyword_search(query, top_k=settings["top_k"] * 2)
    semantic_results = _semantic_search(query, top_k=settings["top_k"] * 2, threshold=settings["score_threshold"])
    semantic_debug = []
    for result in semantic_results:
        item = dict(result)
        item["_filtered_not_indexed"] = not item.get("is_indexed")
        semantic_debug.append(item)
    semantic_indexed = [result for result in semantic_results if result.get("is_indexed")]
    fused = _rrf_fusion(keyword_results, semantic_indexed, k=settings["rrf_k"],
                         keyword_weight=settings["keyword_weight"], semantic_weight=settings["semantic_weight"])
    reranked_results: List[Dict[str, Any]] = []
    if settings["enable_rerank"]:
        fused = _rerank_results(query, fused, model_name=settings["rerank_model"])
        reranked_results = [dict(result) for result in fused]
    primary_accepted = []
    for result in fused:
        context_score = result.get("rerank_score", 0.0) if settings["enable_rerank"] else _context_relevance_score(
            query, f"{result.get('title', '')} {result.get('content', '')}")
        result["context_score"] = round(context_score, 4)
        result["evidence_status"] = "accepted" if context_score >= settings["context_threshold"] else "filtered_low_relevance"
        if result["evidence_status"] == "accepted":
            primary_accepted.append(result)
    composite = advanced_search(query, settings=settings)
    results = composite.get("results", [])
    titles = [result.get("title") for result in results]
    return {
        "query": query, "settings": settings, "keyword_results": keyword_results,
        "semantic_results": semantic_debug, "fused_results": fused,
        "reranked_results": reranked_results, "results": results, "count": len(results),
        "expected_doc_title": expected_doc_title,
        "top1_hit": bool(expected_doc_title and titles and titles[0] == expected_doc_title),
        "top3_hit": bool(expected_doc_title and expected_doc_title in titles),
        "embedding_runtime": rag_embeddings.get_runtime_info(),
        "filter_summary": {"context_threshold": settings["context_threshold"],
                            "accepted_from_primary_fusion": len(primary_accepted),
                            "final_composite_evidence": len(results)},
        "composite_debug": {"quoted_titles": _extract_quoted_titles(query),
                            "sub_queries": _extract_sub_queries(query),
                            "merged_count": composite.get("count", 0)},
    }
