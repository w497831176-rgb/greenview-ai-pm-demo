"""Runtime release, snapshot, policy and evidence APIs."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.runtime.release_compiler import (
    compile_runtime_release,
    publish_compiled_release,
)
from app.runtime.acceptance import ACCEPTANCE_CASES
from app.runtime.snapshot_resolver import resolve_snapshot
from db.property_db import (
    get_agent_by_agent_id,
    get_current_runtime_release,
    get_evidence_ledger,
    get_runtime_release,
    list_runtime_releases,
    list_runtime_acceptance_runs,
    list_tool_policies,
    publish_runtime_release,
    rollback_runtime_release,
    save_runtime_acceptance_run,
    set_agent_knowledge_bindings,
)


router = APIRouter(prefix="/api/runtime", tags=["runtime-v18"])


class PublishRequest(BaseModel):
    created_by: str = "platform-operator"


class RollbackRequest(BaseModel):
    release_id: str


class KnowledgeBindingRequest(BaseModel):
    knowledge_doc_ids: List[int] = Field(default_factory=list)
    publish: bool = False
    created_by: str = "platform-operator"


class ContractAcceptanceRequest(BaseModel):
    case_key: str
    session_id: str
    expected_agent_id: Optional[str] = None
    expected_skill_ids: List[int] = Field(default_factory=list)
    expected_mcp_servers: List[str] = Field(default_factory=list)
    expected_knowledge_doc_ids: List[int] = Field(default_factory=list)


class RetrievalCostPreviewRequest(BaseModel):
    query: str = Field(min_length=1)
    agent_id: str
    top_k: int = Field(ge=1, le=10)


@router.get("/acceptance/cases")
async def acceptance_cases():
    return {"cases": ACCEPTANCE_CASES, "count": len(ACCEPTANCE_CASES)}


@router.get("/acceptance/runs")
async def acceptance_runs(limit: int = 50):
    items = list_runtime_acceptance_runs(limit)
    return {"runs": items, "count": len(items)}


def _redact_release(release: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not release:
        return None
    item = json.loads(json.dumps(release, ensure_ascii=False, default=str))
    config = item.get("config") or {}
    for server in config.get("mcp_servers") or []:
        if server.get("env"):
            server["env"] = {
                key: "***configured***" for key in (server.get("env") or {})
            }
    model_policy = config.get("model_policy") or {}
    for model in [model_policy.get("default"), *(model_policy.get("available") or [])]:
        if isinstance(model, dict) and model.get("api_key"):
            model["api_key"] = "***configured***"
    return item


@router.get("/releases/current")
async def current_release():
    release = _redact_release(get_current_runtime_release())
    if not release:
        raise HTTPException(status_code=503, detail="no published RuntimeRelease")
    return {"release": release}


@router.get("/releases")
async def releases(limit: int = 50):
    items = [_redact_release(item) for item in list_runtime_releases(limit)]
    return {"releases": items, "count": len(items)}


@router.get("/releases/{release_id}")
async def release_detail(release_id: str):
    release = _redact_release(get_runtime_release(release_id))
    if not release:
        raise HTTPException(status_code=404, detail="runtime release not found")
    return {
        "release": release,
        "tool_policies": list_tool_policies(release_id),
    }


@router.post("/releases/compile")
async def compile_release(request: PublishRequest):
    release = compile_runtime_release(created_by=request.created_by)
    return {
        "release": _redact_release(release),
        "published": False,
        "next_step": (
            "publish"
            if (release.get("validation") or {}).get("valid")
            else "fix_validation_errors"
        ),
    }


@router.post("/releases/publish-current-config")
async def publish_current_config(request: PublishRequest):
    release = publish_compiled_release(created_by=request.created_by)
    return {
        "release": _redact_release(release),
        "published": release.get("status") == "published",
        "effective_on": "new_session",
        "existing_sessions": "keep_pinned_snapshot",
    }


@router.post("/releases/{release_id}/publish")
async def publish_existing_release(release_id: str):
    try:
        release = publish_runtime_release(release_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "release": _redact_release(release),
        "published": True,
        "effective_on": "new_session",
    }


@router.post("/releases/rollback")
async def rollback_release(request: RollbackRequest):
    try:
        release = rollback_runtime_release(request.release_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "release": _redact_release(release),
        "rollback": True,
        "effective_on": "new_session",
        "existing_sessions": "keep_pinned_snapshot",
    }


@router.get("/sessions/{session_id}/snapshot")
async def session_snapshot(session_id: str):
    snapshot = resolve_snapshot(session_id)
    payload = snapshot.model_dump(mode="json")
    payload["config"] = (_redact_release({"config": payload["config"]}) or {}).get(
        "config", {}
    )
    return {"snapshot": payload}


@router.get("/traces/{trace_id}/evidence")
async def trace_evidence(trace_id: str):
    ledger = get_evidence_ledger(trace_id)
    if not ledger:
        raise HTTPException(status_code=404, detail="evidence ledger not found")
    return {"evidence": ledger, "ledger": ledger.get("ledger") or {}}


@router.post("/cost-preview/retrieval")
async def retrieval_cost_preview(request: RetrievalCostPreviewRequest):
    """Preview the exact published retrieval boundary without calling a model.

    This uses the current immutable RuntimeRelease, the selected vertical
    Agent's explicit RAG bindings, live retrieval verification and the same
    deterministic snapshot fallback as owner chat.  ``top_k`` is a simulation
    override only: it never mutates draft settings or publishes a release.
    """
    release = get_current_runtime_release()
    if not release:
        raise HTTPException(status_code=503, detail="no published RuntimeRelease")
    config = release.get("config") or {}
    agent = next(
        (
            item
            for item in config.get("agents") or []
            if item.get("agent_id") == request.agent_id
            and item.get("enabled")
            and item.get("category") not in {"router", "orchestration"}
        ),
        None,
    )
    if not agent:
        raise HTTPException(status_code=404, detail="vertical agent not found")

    allowed_document_ids = {
        int(item) for item in (agent.get("knowledge_doc_ids") or [])
    }
    knowledge_versions = {
        int(item["knowledge_doc_id"]): item
        for item in (config.get("knowledge") or [])
        if int(item.get("knowledge_doc_id") or 0) in allowed_document_ids
    }
    policy = dict(config.get("retrieval_policy") or {})
    policy["top_k"] = request.top_k
    live_results: List[Dict[str, Any]] = []
    live_status = "completed"
    try:
        import rag_retrieval

        retrieval = await asyncio.to_thread(
            rag_retrieval.advanced_search,
            request.query,
            policy,
        )
        live_results = list((retrieval or {}).get("results") or [])
    except Exception as exc:
        live_status = f"failed:{type(exc).__name__}"

    from app.runtime.coordinator import _results_from_snapshot

    results, used_snapshot_fallback = _results_from_snapshot(
        request.query,
        live_results,
        knowledge_versions,
        allowed_document_ids,
        request.top_k,
    )
    preview_results = [
        {
            "document_id": item.get("doc_id", item.get("document_id")),
            "document_title": item.get("doc_title") or item.get("title") or "",
            "chunk_index": item.get("chunk_index"),
            "content": item.get("content") or item.get("chunk_text") or "",
            "score": item.get("score"),
            "retrieval_sources": item.get("retrieval_sources") or [],
        }
        for item in results
    ]
    return {
        "preview": {
            "release_id": release.get("release_id"),
            "agent_id": request.agent_id,
            "bound_document_ids": sorted(allowed_document_ids),
            "simulated_top_k": request.top_k,
            "retrieval_status": live_status,
            "used_snapshot_fallback": used_snapshot_fallback,
            "results": preview_results,
            "evidence_count": len(preview_results),
            "context_characters": sum(
                len(item["content"]) for item in preview_results
            ),
            "provider_usage": None,
            "estimated_cost": None,
            "claim_policy": (
                "无模型预估不等于 Provider Token 或成本；必须发布候选 Release，"
                "用同题真实 Trace 通过质量门槛后才能宣称收益。"
            ),
        },
        "configuration_mutated": False,
        "model_called": False,
    }


@router.put("/agents/{agent_id}/knowledge-bindings")
async def bind_agent_knowledge(agent_id: str, request: KnowledgeBindingRequest):
    if not get_agent_by_agent_id(agent_id):
        raise HTTPException(status_code=404, detail="agent not found")
    set_agent_knowledge_bindings(agent_id, request.knowledge_doc_ids)
    release = None
    if request.publish:
        release = publish_compiled_release(created_by=request.created_by)
    return {
        "agent_id": agent_id,
        "knowledge_doc_ids": sorted(set(request.knowledge_doc_ids)),
        "release": _redact_release(release),
        "effective_on": "new_session" if release else "after_publish",
    }


@router.post("/acceptance/contract")
async def contract_acceptance(request: ContractAcceptanceRequest):
    """No-model proof that a new session sees one coherent capability graph."""
    snapshot = resolve_snapshot(request.session_id)
    config = snapshot.config
    agent = next(
        (
            item
            for item in config.get("agents") or []
            if item.get("agent_id") == request.expected_agent_id
        ),
        None,
    )
    assertions: List[Dict[str, Any]] = []
    if request.expected_agent_id:
        assertions.append(
            {
                "name": "agent_in_snapshot",
                "passed": bool(agent and agent.get("enabled")),
                "expected": request.expected_agent_id,
            }
        )
    if agent:
        assertions.extend(
            [
                {
                    "name": "skill_bindings",
                    "passed": set(request.expected_skill_ids).issubset(
                        set(agent.get("skill_ids") or [])
                    ),
                    "expected": request.expected_skill_ids,
                    "actual": agent.get("skill_ids") or [],
                },
                {
                    "name": "mcp_bindings",
                    "passed": set(request.expected_mcp_servers).issubset(
                        set(agent.get("mcp_server_names") or [])
                    ),
                    "expected": request.expected_mcp_servers,
                    "actual": agent.get("mcp_server_names") or [],
                },
                {
                    "name": "knowledge_bindings",
                    "passed": set(request.expected_knowledge_doc_ids).issubset(
                        set(agent.get("knowledge_doc_ids") or [])
                    ),
                    "expected": request.expected_knowledge_doc_ids,
                    "actual": agent.get("knowledge_doc_ids") or [],
                },
            ]
        )
    passed = bool(assertions) and all(item["passed"] for item in assertions)
    result = {
        "case_key": request.case_key,
        "passed": passed,
        "release_id": snapshot.release_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "assertions": assertions,
        "model_called": False,
        "writes_business_data": False,
    }
    saved = save_runtime_acceptance_run(
        acceptance_run_id=f"accept_{uuid.uuid4().hex}",
        case_key=request.case_key,
        release_id=snapshot.release_id,
        status="passed" if passed else "failed",
        evidence=result,
        cleanup={"required": False, "reason": "no-model snapshot contract"},
    )
    result["acceptance_run_id"] = saved.get("acceptance_run_id")
    return result
