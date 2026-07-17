"""
Knowledge Base API
==================

REST endpoints for knowledge docs and badcases (platform management views).
Includes keyword search, semantic RAG search, chunk management and evaluation.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from db.property_db import (
    create_badcase as db_create_badcase,
    create_knowledge_doc as db_create_knowledge_doc,
    delete_badcase as db_delete_badcase,
    delete_knowledge_doc as db_delete_knowledge_doc,
    get_badcase as db_get_badcase,
    get_knowledge_doc as db_get_knowledge_doc,
    get_knowledge_draft as db_get_knowledge_draft,
    get_retrieval_settings as db_get_retrieval_settings,
    list_badcases as db_list_badcases,
    list_knowledge_docs as db_list_knowledge_docs,
    search_knowledge as db_search_knowledge,
    set_knowledge_doc_indexed as db_set_knowledge_doc_indexed,
    set_knowledge_doc_indexed_flag as db_set_knowledge_doc_indexed_flag,
    update_badcase as db_update_badcase,
    update_knowledge_doc as db_update_knowledge_doc,
    update_knowledge_draft as db_update_knowledge_draft,
    update_retrieval_settings as db_update_retrieval_settings,
)
import rag_indexer
import rag_retrieval
import rag_store

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# Standalone retrieval router so the same endpoints can also be mounted at /api/retrieval.
retrieval_router = APIRouter(tags=["retrieval"])


class KnowledgeDocCreate(BaseModel):
    title: str
    content: str
    category: str = "未分类"
    chunk_size: int = 512
    chunk_overlap: int = 64
    split_strategy: str = "auto"


class KnowledgeDocUpdate(BaseModel):
    title: str
    content: str
    category: str = "未分类"
    chunk_size: int = 512
    chunk_overlap: int = 64
    split_strategy: str = "auto"


class ChunkConfigUpdate(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    split_strategy: str = "auto"


class IndexedFlagUpdate(BaseModel):
    is_indexed: bool


class DebugSearchRequest(BaseModel):
    query: str
    top_k: int = Field(3, ge=1, le=10)
    threshold: Optional[float] = None
    expected_doc_title: Optional[str] = None


class BadcaseCreate(BaseModel):
    title: str
    description: str
    category: str = "未分类"
    status: str = "待处理"


class BadcaseUpdate(BaseModel):
    title: str
    description: str
    category: str = "未分类"
    status: str = "待处理"


class RetrievalSettingsUpdate(BaseModel):
    top_k: int = Field(5, ge=1, le=10)
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    rrf_k: int = 60
    enable_rerank: bool = False
    rerank_model: Optional[str] = None
    score_threshold: float = 0.0
    context_threshold: float = 0.2


class RetrievalDebugRequest(BaseModel):
    query: str
    top_k: int = Field(5, ge=1, le=10)
    keyword_weight: float = 0.3
    semantic_weight: float = 0.7
    rrf_k: int = 60
    enable_rerank: bool = False
    rerank_model: Optional[str] = None
    score_threshold: Optional[float] = None
    context_threshold: Optional[float] = None
    expected_doc_title: Optional[str] = None


@retrieval_router.get("/settings")
async def get_retrieval_settings_top():
    """Get advanced RAG retrieval settings (mounted at /api/retrieval)."""
    settings = db_get_retrieval_settings("default")
    if not settings:
        settings = db_update_retrieval_settings("default")
    return {"retrieval_settings": settings}


@retrieval_router.post("/settings")
async def update_retrieval_settings_top(request: RetrievalSettingsUpdate):
    """Update advanced RAG retrieval settings (mounted at /api/retrieval)."""
    settings = db_update_retrieval_settings(
        name="default",
        top_k=request.top_k,
        keyword_weight=request.keyword_weight,
        semantic_weight=request.semantic_weight,
        rrf_k=request.rrf_k,
        enable_rerank=request.enable_rerank,
        rerank_model=request.rerank_model,
        score_threshold=request.score_threshold,
        context_threshold=request.context_threshold,
    )
    return {"retrieval_settings": settings}


@retrieval_router.post("/debug")
async def debug_retrieval_top(request: RetrievalDebugRequest):
    """Debug advanced RAG pipeline (mounted at /api/retrieval)."""
    result = rag_retrieval.debug_search(
        query=request.query,
        top_k=request.top_k,
        keyword_weight=request.keyword_weight,
        semantic_weight=request.semantic_weight,
        rrf_k=request.rrf_k,
        enable_rerank=request.enable_rerank,
        rerank_model=request.rerank_model,
        score_threshold=request.score_threshold,
        context_threshold=request.context_threshold,
        expected_doc_title=request.expected_doc_title,
    )
    return result


@router.get("/retrieval/settings")
async def get_retrieval_settings():
    """Get advanced RAG retrieval settings."""
    settings = db_get_retrieval_settings("default")
    if not settings:
        settings = db_update_retrieval_settings("default")
    return {"retrieval_settings": settings}


@router.post("/retrieval/settings")
async def update_retrieval_settings(request: RetrievalSettingsUpdate):
    """Update advanced RAG retrieval settings."""
    settings = db_update_retrieval_settings(
        name="default",
        top_k=request.top_k,
        keyword_weight=request.keyword_weight,
        semantic_weight=request.semantic_weight,
        rrf_k=request.rrf_k,
        enable_rerank=request.enable_rerank,
        rerank_model=request.rerank_model,
        score_threshold=request.score_threshold,
        context_threshold=request.context_threshold,
    )
    return {"retrieval_settings": settings}


@router.post("/retrieval/debug")
async def debug_retrieval(request: RetrievalDebugRequest):
    """Debug advanced RAG pipeline with full stage visibility."""
    result = rag_retrieval.debug_search(
        query=request.query,
        top_k=request.top_k,
        keyword_weight=request.keyword_weight,
        semantic_weight=request.semantic_weight,
        rrf_k=request.rrf_k,
        enable_rerank=request.enable_rerank,
        rerank_model=request.rerank_model,
        score_threshold=request.score_threshold,
        context_threshold=request.context_threshold,
        expected_doc_title=request.expected_doc_title,
    )
    return result


@router.get("/retrieval-settings")
async def get_retrieval_settings_alias():
    """Frontend alias for /retrieval/settings."""
    settings = db_get_retrieval_settings("default")
    if not settings:
        settings = db_update_retrieval_settings("default")
    return {"retrieval_settings": settings}


@router.post("/retrieval-settings")
async def update_retrieval_settings_alias(request: RetrievalSettingsUpdate):
    """Frontend alias for /retrieval/settings."""
    settings = db_update_retrieval_settings(
        name="default",
        top_k=request.top_k,
        keyword_weight=request.keyword_weight,
        semantic_weight=request.semantic_weight,
        rrf_k=request.rrf_k,
        enable_rerank=request.enable_rerank,
        rerank_model=request.rerank_model,
        score_threshold=request.score_threshold,
        context_threshold=request.context_threshold,
    )
    return {"retrieval_settings": settings}


@router.post("/retrieval-debug")
async def debug_retrieval_alias(request: RetrievalDebugRequest):
    """Frontend alias for /retrieval/debug."""
    return await debug_retrieval(request)


@router.post("/drafts/{draft_id}/approve")
async def approve_knowledge_draft(draft_id: int):
    """Approve a knowledge draft and publish it to the knowledge base.

    Mirrors the behaviour of /badcases/{id}/publish-draft/{draft_id}.
    """
    draft = db_get_knowledge_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=404, detail="draft not found")

    doc = db_create_knowledge_doc(
        title=draft["title"],
        content=draft["content"],
        category=draft.get("category", "未分类"),
    )
    db_update_knowledge_draft(draft_id, status="published")

    # If this draft belongs to a badcase in fixing state, move it to verifying.
    badcase_id = draft.get("badcase_id")
    if badcase_id:
        bc = db_get_badcase(badcase_id)
        if bc and bc.get("status") == "fixing":
            db_update_badcase(badcase_id, status="verifying", fix_plan="knowledge published")

    return {"knowledge_doc": doc}


@router.get("/docs")
async def list_knowledge_docs():
    """List all knowledge documents."""
    docs = db_list_knowledge_docs()
    return {"knowledge_docs": docs, "count": len(docs)}


@router.get("/docs/{doc_id}")
async def get_knowledge_doc(doc_id: int):
    """Get a single knowledge document."""
    doc = db_get_knowledge_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="not found")
    return {"knowledge_doc": doc}


@router.post("/docs")
async def create_knowledge_doc(request: KnowledgeDocCreate):
    """Create a new knowledge document and trigger indexing."""
    doc = db_create_knowledge_doc(
        title=request.title,
        content=request.content,
        category=request.category,
        index_status="pending",
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        split_strategy=request.split_strategy,
    )
    if doc:
        try:
            rag_indexer.index_document(doc["id"])
        except Exception:
            pass
        doc = db_get_knowledge_doc(doc["id"])
    return {"knowledge_doc": doc}


@router.put("/docs/{doc_id}")
async def update_knowledge_doc(doc_id: int, request: KnowledgeDocUpdate):
    """Update a knowledge document and reindex if content changed."""
    old_doc = db_get_knowledge_doc(doc_id)
    if not old_doc:
        raise HTTPException(status_code=404, detail="not found")

    doc = db_update_knowledge_doc(
        doc_id=doc_id,
        title=request.title,
        content=request.content,
        category=request.category,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        split_strategy=request.split_strategy,
        index_status="pending" if old_doc.get("content") != request.content else None,
    )
    if doc and doc.get("is_indexed"):
        rag_indexer.reindex_document(doc_id)
        doc = db_get_knowledge_doc(doc_id)
    return {"knowledge_doc": doc}


@router.delete("/docs/{doc_id}")
async def delete_knowledge_doc(doc_id: int):
    """Delete a knowledge document and its vectors."""
    deleted = db_delete_knowledge_doc(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    rag_store.delete_chunks_for_doc(doc_id)
    return {"ok": True, "deleted_id": doc_id}


@router.get("/docs/{doc_id}/chunks")
async def list_doc_chunks(doc_id: int):
    """List vector chunks for a document."""
    doc = db_get_knowledge_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="not found")
    chunks = rag_store.list_chunks_for_doc(doc_id)
    return {"knowledge_doc": doc, "chunks": chunks, "count": len(chunks)}


@router.post("/docs/{doc_id}/reindex")
async def reindex_doc(doc_id: int):
    """Manually re-chunk and reindex a document."""
    doc = db_get_knowledge_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="not found")
    ok = rag_indexer.reindex_document(doc_id)
    if not ok:
        raise HTTPException(status_code=500, detail="indexing failed")
    return {"knowledge_doc": db_get_knowledge_doc(doc_id)}


@router.patch("/docs/{doc_id}/indexed")
async def toggle_doc_indexed(doc_id: int, request: IndexedFlagUpdate):
    """Enable or disable a document from retrieval."""
    doc = db_get_knowledge_doc(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="not found")
    db_set_knowledge_doc_indexed_flag(doc_id, request.is_indexed)
    return {"knowledge_doc": db_get_knowledge_doc(doc_id)}


@router.get("/search")
async def search_docs(
    query: str = Query(...),
    top_k: int = Query(3, ge=1, le=10),
    mode: str = Query("keyword"),
    threshold: Optional[float] = Query(None),
):
    """Search knowledge documents by keyword, semantic similarity, or advanced RAG fusion."""
    if mode == "semantic":
        results = rag_indexer.semantic_search(query, top_k=top_k, threshold=threshold)
    elif mode == "advanced":
        settings = db_get_retrieval_settings("default") or {}
        results = rag_retrieval.advanced_search(
            query,
            settings={
                "top_k": top_k,
                "keyword_weight": settings.get("keyword_weight", 0.3),
                "semantic_weight": settings.get("semantic_weight", 0.7),
                "rrf_k": settings.get("rrf_k", 60),
                "enable_rerank": settings.get("enable_rerank", False),
                "rerank_model": settings.get("rerank_model"),
                "score_threshold": threshold if threshold is not None else settings.get("score_threshold", 0.0),
                "context_threshold": settings.get("context_threshold", 0.2),
            },
        )
        return {"results": results["results"], "count": results["count"], "mode": mode, "details": results}
    else:
        results = db_search_knowledge(query, top_k=top_k)
    return {"results": results, "count": len(results), "mode": mode}


@router.post("/debug")
async def debug_search(request: DebugSearchRequest):
    """Interactive retrieval debugger: run semantic search with explicit parameters.

    Returns the raw retrieved chunks, the effective threshold used by the indexer,
    and optional top1/top3 hit metrics when an expected document title is provided.
    """
    results = rag_indexer.semantic_search(
        request.query, top_k=request.top_k, threshold=request.threshold
    )
    titles = [r.get("doc_title") for r in results]
    expected = request.expected_doc_title
    top1_hit = bool(expected and titles and titles[0] == expected)
    top3_hit = bool(expected and expected in titles)
    return {
        "query": request.query,
        "top_k": request.top_k,
        "threshold": request.threshold,
        "threshold_used": rag_indexer._effective_threshold(request.threshold),
        "count": len(results),
        "results": results,
        "expected_doc_title": expected,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
    }


@router.get("/badcases")
async def list_badcases():
    """List all badcases."""
    cases = db_list_badcases()
    return {"badcases": cases, "count": len(cases)}


@router.get("/badcases/{case_id}")
async def get_badcase(case_id: int):
    """Get a single badcase."""
    case = db_get_badcase(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    return {"badcase": case}


@router.post("/badcases")
async def create_badcase(request: BadcaseCreate):
    """Create a new badcase."""
    case = db_create_badcase(
        title=request.title,
        description=request.description,
        category=request.category,
        status=request.status,
    )
    return {"badcase": case}


@router.put("/badcases/{case_id}")
async def update_badcase(case_id: int, request: BadcaseUpdate):
    """Update a badcase."""
    case = db_update_badcase(
        case_id=case_id,
        title=request.title,
        description=request.description,
        category=request.category,
        status=request.status,
    )
    if not case:
        raise HTTPException(status_code=404, detail="not found")
    return {"badcase": case}


@router.delete("/badcases/{case_id}")
async def delete_badcase(case_id: int):
    """Delete a badcase."""
    deleted = db_delete_badcase(case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="not found")
    return {"ok": True, "deleted_id": case_id}
