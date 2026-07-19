"""Runtime release, snapshot, policy and evidence APIs."""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.runtime.agent_factory import vertical_agent_cards
from app.runtime.contracts import RuntimePath, ToolEffect
from app.runtime.release_compiler import (
    compile_runtime_release,
    publish_compiled_release,
)
from app.runtime.acceptance import ACCEPTANCE_CASES
from app.runtime.snapshot_resolver import resolve_snapshot
from app.runtime.tool_planner import plan_tools
from app.skill_runtime import select_skills
from agents.router import _capability_fallback
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
    baseline_session_id: Optional[str] = None


class RetrievalCostPreviewRequest(BaseModel):
    query: str = Field(min_length=1)
    agent_id: str
    top_k: int = Field(ge=1, le=10)


class ExtensionAcceptanceRequest(BaseModel):
    """No-model proof for a newly published, off-domain capability package."""

    case_key: str = "EXT-OFFDOMAIN-01"
    session_id: str
    query: str = Field(min_length=1)
    expected_agent_id: str
    expected_skill_ids: List[int] = Field(default_factory=list)
    expected_mcp_tools: List[str] = Field(default_factory=list)
    expected_knowledge_doc_ids: List[int] = Field(default_factory=list)
    run_scoped_retrieval: bool = False
    baseline_session_id: Optional[str] = None
    expected_baseline_snapshot_hash: Optional[str] = None


class TraceAcceptanceRequest(BaseModel):
    """Evaluate one real chat Trace without sending another model request."""

    case_key: str = "HIT-TRACE-01"
    trace_id: str
    expected_runtime_path: Optional[str] = None
    expected_agent_id: Optional[str] = None
    expected_skill_ids: List[int] = Field(default_factory=list)
    expected_mcp_tools: List[str] = Field(default_factory=list)
    expected_knowledge_doc_ids: List[int] = Field(default_factory=list)
    require_skill: bool = False
    require_rag: bool = False
    require_mcp: bool = False
    require_handoff: bool = False
    require_action_receipt: bool = False
    require_badcase: bool = False
    require_evaluation: bool = True
    require_cost: bool = True
    forbid_business_write: bool = True


class CompositeAcceptanceRequest(BaseModel):
    """Parent Acceptance Run that links independently safe child Traces."""

    case_key: str = "HIT-COMPOSITE-01"
    child_cases: List[TraceAcceptanceRequest] = Field(min_length=1)
    expected_child_case_keys: List[str] = Field(default_factory=list)


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
            allowed_document_ids=sorted(allowed_document_ids),
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
    baseline_snapshot = (
        resolve_snapshot(request.baseline_session_id)
        if request.baseline_session_id
        else None
    )
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


def _acceptance_assertion(
    name: str,
    passed: bool,
    expected: Any = None,
    actual: Any = None,
    detail: str = "",
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


@router.post("/acceptance/extension")
async def extension_acceptance(request: ExtensionAcceptanceRequest):
    """Exercise Router/Skill/ToolPlan/RAG scope from one immutable snapshot.

    This endpoint never calls an LLM and never invokes an MCP write.  Optional
    retrieval uses the local embedding/index pipeline inside the Agent-bound
    document scope.
    """

    snapshot = resolve_snapshot(request.session_id)
    baseline_snapshot = (
        resolve_snapshot(request.baseline_session_id)
        if request.baseline_session_id
        else None
    )
    cards = vertical_agent_cards(snapshot.config)
    routed_agent_id, route_reason, route_scores = _capability_fallback(
        request.query,
        cards,
    )
    agent = next(
        (
            item
            for item in snapshot.config.get("agents") or []
            if item.get("agent_id") == request.expected_agent_id
            and item.get("enabled")
            and item.get("category") not in {"router", "orchestration"}
        ),
        None,
    )
    assertions: List[Dict[str, Any]] = [
        _acceptance_assertion(
            "dynamic_router",
            routed_agent_id == request.expected_agent_id,
            request.expected_agent_id,
            routed_agent_id,
            route_reason,
        ),
        _acceptance_assertion(
            "agent_in_snapshot",
            bool(agent),
            request.expected_agent_id,
            agent.get("agent_id") if agent else None,
        ),
    ]
    if baseline_snapshot:
        baseline_agent_ids = {
            str(item.get("agent_id"))
            for item in baseline_snapshot.config.get("agents") or []
        }
        assertions.append(
            _acceptance_assertion(
                "old_session_does_not_hot_load_new_agent",
                request.expected_agent_id not in baseline_agent_ids
                and baseline_snapshot.snapshot_hash != snapshot.snapshot_hash,
                {
                    "new_agent_absent": request.expected_agent_id,
                    "different_snapshot": True,
                },
                {
                    "baseline_snapshot_hash": baseline_snapshot.snapshot_hash,
                    "new_snapshot_hash": snapshot.snapshot_hash,
                    "baseline_agent_present": (
                        request.expected_agent_id in baseline_agent_ids
                    ),
                },
            )
        )
        if request.expected_baseline_snapshot_hash:
            assertions.append(
                _acceptance_assertion(
                    "old_session_snapshot_hash_unchanged",
                    baseline_snapshot.snapshot_hash
                    == request.expected_baseline_snapshot_hash,
                    request.expected_baseline_snapshot_hash,
                    baseline_snapshot.snapshot_hash,
                )
            )
    selected_skill_ids: List[int] = []
    read_plans = []
    write_plans = []
    retrieval_evidence: List[Dict[str, Any]] = []
    retrieval_scope: Optional[Dict[str, Any]] = None
    if agent:
        skills_by_id = {
            int(item["skill_id"]): item
            for item in snapshot.config.get("skills") or []
        }
        skill_candidates = [
            {
                "id": item["skill_id"],
                "name": item.get("name"),
                "description": item.get("description"),
                "instructions": item.get("instructions_fallback"),
                "enabled": item.get("enabled"),
                "trigger_condition": item.get("trigger_condition"),
                "skill_metadata": item.get("metadata") or {},
            }
            for skill_id in agent.get("skill_ids") or []
            for item in [skills_by_id.get(int(skill_id))]
            if item and item.get("enabled")
        ]
        selected_skills, _ = select_skills(skill_candidates, request.query)
        selected_skill_ids = [
            int(item["skill_id"]) for item in selected_skills
        ]
        read_plans = plan_tools(
            snapshot.config,
            request.expected_agent_id,
            request.query,
            RuntimePath.CONSULTATION,
            effects=[ToolEffect.READ],
        )
        write_plans = plan_tools(
            snapshot.config,
            request.expected_agent_id,
            request.query,
            RuntimePath.CONTROLLED_ACTION,
            effects=[ToolEffect.CREATE, ToolEffect.UPDATE],
            execution_modes=["proposal"],
        )
        assertions.extend(
            [
                _acceptance_assertion(
                    "skill_runtime_selection",
                    set(request.expected_skill_ids).issubset(
                        set(selected_skill_ids)
                    ),
                    request.expected_skill_ids,
                    selected_skill_ids,
                ),
                _acceptance_assertion(
                    "mcp_binding",
                    {
                        key.split(":", 1)[0]
                        for key in request.expected_mcp_tools
                    }.issubset(set(agent.get("mcp_server_names") or [])),
                    request.expected_mcp_tools,
                    agent.get("mcp_server_names") or [],
                ),
                _acceptance_assertion(
                    "rag_binding",
                    set(request.expected_knowledge_doc_ids).issubset(
                        set(agent.get("knowledge_doc_ids") or [])
                    ),
                    request.expected_knowledge_doc_ids,
                    agent.get("knowledge_doc_ids") or [],
                ),
            ]
        )
    planned_keys = {
        f"{item.server_name}:{item.tool_name}"
        for item in [*read_plans, *write_plans]
    }
    assertions.append(
        _acceptance_assertion(
            "configuration_driven_tool_plan",
            set(request.expected_mcp_tools).issubset(planned_keys),
            request.expected_mcp_tools,
            sorted(planned_keys),
        )
    )
    if agent and request.run_scoped_retrieval:
        allowed_document_ids = sorted(
            {int(item) for item in agent.get("knowledge_doc_ids") or []}
        )
        try:
            import rag_retrieval

            retrieval = await asyncio.to_thread(
                rag_retrieval.advanced_search,
                request.query,
                snapshot.config.get("retrieval_policy") or {},
                allowed_document_ids=allowed_document_ids,
            )
            retrieval_evidence = list((retrieval or {}).get("results") or [])
            retrieval_scope = (retrieval or {}).get("scope")
            actual_document_ids = {
                int(item.get("doc_id", item.get("document_id")))
                for item in retrieval_evidence
                if item.get("doc_id", item.get("document_id")) is not None
            }
            assertions.append(
                _acceptance_assertion(
                    "rag_scope_before_top_k",
                    actual_document_ids.issubset(set(allowed_document_ids))
                    and (retrieval_scope or {}).get("mode") == "agent_bound",
                    allowed_document_ids,
                    {
                        "retrieved_document_ids": sorted(actual_document_ids),
                        "scope": retrieval_scope,
                    },
                )
            )
        except Exception as exc:
            assertions.append(
                _acceptance_assertion(
                    "rag_scope_before_top_k",
                    False,
                    request.expected_knowledge_doc_ids,
                    type(exc).__name__,
                    str(exc)[:300],
                )
            )
    passed = bool(assertions) and all(item["passed"] for item in assertions)
    result = {
        "case_key": request.case_key,
        "passed": passed,
        "release_id": snapshot.release_id,
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_diff": (
            {
                "baseline_session_id": request.baseline_session_id,
                "baseline_release_id": baseline_snapshot.release_id,
                "baseline_snapshot_id": baseline_snapshot.snapshot_id,
                "baseline_snapshot_hash": baseline_snapshot.snapshot_hash,
                "new_session_id": request.session_id,
                "new_release_id": snapshot.release_id,
                "new_snapshot_id": snapshot.snapshot_id,
                "new_snapshot_hash": snapshot.snapshot_hash,
            }
            if baseline_snapshot
            else None
        ),
        "route": {
            "selected_agent_id": routed_agent_id,
            "reason": route_reason,
            "scores": route_scores,
        },
        "selected_skill_ids": selected_skill_ids,
        "tool_plans": [
            item.model_dump(mode="json") for item in [*read_plans, *write_plans]
        ],
        "retrieval_scope": retrieval_scope,
        "retrieval_evidence": retrieval_evidence,
        "assertions": assertions,
        "model_called": False,
        "mcp_invoked": False,
        "writes_business_data": False,
    }
    saved = save_runtime_acceptance_run(
        acceptance_run_id=f"accept_{uuid.uuid4().hex}",
        case_key=request.case_key,
        release_id=snapshot.release_id,
        status="passed" if passed else "failed",
        evidence=result,
        cleanup={
            "required": True,
            "strategy": "rollback_test_release_and_remove_test_configuration",
        },
    )
    result["acceptance_run_id"] = saved.get("acceptance_run_id")
    return result


@router.post("/acceptance/trace")
async def trace_acceptance(request: TraceAcceptanceRequest):
    """Evaluate evidence already produced by one real owner chat Trace."""

    evidence_row = get_evidence_ledger(request.trace_id)
    if not evidence_row:
        raise HTTPException(status_code=404, detail="evidence ledger not found")
    ledger = evidence_row.get("ledger") or {}
    route = ledger.get("route_decision") or {}
    skills = ledger.get("activated_skills") or []
    invocations = ledger.get("tool_invocations") or []
    retrieval = ledger.get("retrieval_evidence") or []
    citations = ledger.get("citation_links") or []
    evaluations = ledger.get("evaluation_results") or []
    costs = ledger.get("cost_entries") or []
    model_calls = ledger.get("model_calls") or []
    proposals = ledger.get("action_proposals") or []
    receipts = ledger.get("action_receipts") or []
    badcase_links = ledger.get("badcase_links") or []
    violations = ledger.get("contract_violations") or []
    actual_skill_ids = {
        int(item["skill_id"])
        for item in skills
        if item.get("skill_id") is not None
    }
    actual_mcp_keys = {
        f"{item.get('server_name')}:{item.get('tool_name')}"
        for item in invocations
        if item.get("invocation_status") == "success"
    }
    actual_doc_ids = {
        int(item.get("document_id", item.get("knowledge_id")))
        for item in retrieval
        if item.get("document_id", item.get("knowledge_id")) is not None
        and str(item.get("document_id", item.get("knowledge_id"))).isdigit()
    }
    evidence_ids = {
        str(item.get("evidence_id"))
        for item in retrieval
        if item.get("evidence_id")
    }
    citation_evidence_ids = {
        str(item.get("evidence_id"))
        for item in citations
        if item.get("evidence_id")
    }
    assertions: List[Dict[str, Any]] = []
    if request.expected_runtime_path:
        assertions.append(
            _acceptance_assertion(
                "runtime_path",
                evidence_row.get("runtime_path") == request.expected_runtime_path,
                request.expected_runtime_path,
                evidence_row.get("runtime_path"),
            )
        )
    if request.expected_agent_id:
        assertions.append(
            _acceptance_assertion(
                "route_agent",
                route.get("selected_agent_id") == request.expected_agent_id,
                request.expected_agent_id,
                route.get("selected_agent_id"),
                str(route.get("reason") or ""),
            )
        )
    assertions.extend(
        [
            _acceptance_assertion(
                "skill_hit",
                (not request.require_skill and not request.expected_skill_ids)
                or (
                    bool(actual_skill_ids)
                    and set(request.expected_skill_ids).issubset(actual_skill_ids)
                ),
                request.expected_skill_ids or ("at_least_one" if request.require_skill else []),
                sorted(actual_skill_ids),
            ),
            _acceptance_assertion(
                "mcp_hit",
                (not request.require_mcp and not request.expected_mcp_tools)
                or (
                    bool(actual_mcp_keys)
                    and set(request.expected_mcp_tools).issubset(actual_mcp_keys)
                ),
                request.expected_mcp_tools or ("at_least_one" if request.require_mcp else []),
                sorted(actual_mcp_keys),
            ),
            _acceptance_assertion(
                "rag_hit",
                (not request.require_rag and not request.expected_knowledge_doc_ids)
                or (
                    bool(retrieval)
                    and set(request.expected_knowledge_doc_ids).issubset(
                        actual_doc_ids
                    )
                ),
                request.expected_knowledge_doc_ids or ("at_least_one" if request.require_rag else []),
                sorted(actual_doc_ids),
            ),
            _acceptance_assertion(
                "citation_same_evidence_set",
                (not request.require_rag and not citations)
                or (
                    bool(citations)
                    and citation_evidence_ids.issubset(evidence_ids)
                ),
                "citation evidence_id subset of retrieval evidence_id",
                sorted(citation_evidence_ids),
            ),
            _acceptance_assertion(
                "evaluation_present",
                (not request.require_evaluation) or bool(evaluations),
                request.require_evaluation,
                evaluations,
            ),
            _acceptance_assertion(
                "cost_chain_present",
                (not request.require_cost)
                or (
                    bool(costs)
                    and bool(model_calls)
                    and all(item.get("usage_source") for item in costs)
                ),
                request.require_cost,
                {
                    "cost_stages": [item.get("stage") for item in costs],
                    "model_stages": [item.get("stage") for item in model_calls],
                },
            ),
            _acceptance_assertion(
                "no_unauthorized_business_write",
                (not request.forbid_business_write)
                or (not proposals and not receipts),
                request.forbid_business_write,
                {
                    "proposal_count": len(proposals),
                    "receipt_count": len(receipts),
                },
            ),
            _acceptance_assertion(
                "no_contract_violation",
                not violations,
                [],
                violations,
            ),
        ]
    )
    if request.require_handoff:
        handoff_evaluations = [
            item for item in evaluations if item.get("case") == "handoff_policy"
        ]
        assertions.append(
            _acceptance_assertion(
                "handoff_hit",
                bool(handoff_evaluations),
                True,
                handoff_evaluations,
            )
        )
    if request.require_action_receipt:
        committed_receipts = [
            item
            for item in receipts
            if item.get("status") == "committed" and item.get("resource_id")
        ]
        assertions.append(
            _acceptance_assertion(
                "committed_action_receipt",
                bool(committed_receipts),
                True,
                committed_receipts,
            )
        )
    if request.require_badcase:
        assertions.append(
            _acceptance_assertion(
                "badcase_link",
                bool(badcase_links),
                True,
                badcase_links,
            )
        )
    passed = bool(assertions) and all(item["passed"] for item in assertions)
    result = {
        "case_key": request.case_key,
        "trace_id": request.trace_id,
        "passed": passed,
        "release_id": evidence_row.get("release_id"),
        "runtime_path": evidence_row.get("runtime_path"),
        "assertions": assertions,
        "evidence_summary": {
            "route": route,
            "skills": len(skills),
            "rag_evidence": len(retrieval),
            "citations": len(citations),
            "mcp_invocations": len(invocations),
            "evaluations": len(evaluations),
            "cost_entries": len(costs),
            "badcase_links": badcase_links,
        },
        "model_called": False,
        "writes_business_data": False,
    }
    saved = save_runtime_acceptance_run(
        acceptance_run_id=f"accept_{uuid.uuid4().hex}",
        case_key=request.case_key,
        release_id=evidence_row.get("release_id"),
        status="passed" if passed else "failed",
        evidence=result,
        cleanup={"required": False, "reason": "trace evidence is retained"},
    )
    result["acceptance_run_id"] = saved.get("acceptance_run_id")
    return result


@router.post("/acceptance/composite")
async def composite_acceptance(request: CompositeAcceptanceRequest):
    """Create one parent acceptance record over independently governed Traces."""

    child_results = [
        await trace_acceptance(child_case)
        for child_case in request.child_cases
    ]
    actual_case_keys = [item.get("case_key") for item in child_results]
    assertions = [
        _acceptance_assertion(
            "all_child_traces_passed",
            all(item.get("passed") for item in child_results),
            True,
            [
                {
                    "case_key": item.get("case_key"),
                    "trace_id": item.get("trace_id"),
                    "passed": item.get("passed"),
                }
                for item in child_results
            ],
        ),
        _acceptance_assertion(
            "required_child_cases_present",
            set(request.expected_child_case_keys).issubset(
                set(actual_case_keys)
            ),
            request.expected_child_case_keys,
            actual_case_keys,
        ),
        _acceptance_assertion(
            "child_traces_are_distinct",
            len({item.get("trace_id") for item in child_results})
            == len(child_results),
            "one governed Trace per scenario",
            [item.get("trace_id") for item in child_results],
        ),
    ]
    passed = all(item["passed"] for item in assertions)
    release_ids = sorted(
        {
            str(item.get("release_id"))
            for item in child_results
            if item.get("release_id")
        }
    )
    result = {
        "case_key": request.case_key,
        "passed": passed,
        "release_id": release_ids[0] if len(release_ids) == 1 else None,
        "release_ids": release_ids,
        "parent_trace_id": None,
        "child_trace_ids": [
            item.get("trace_id") for item in child_results
        ],
        "child_acceptance_run_ids": [
            item.get("acceptance_run_id") for item in child_results
        ],
        "child_results": child_results,
        "assertions": assertions,
        "model_called": False,
        "writes_business_data": False,
        "note": (
            "父 Acceptance Run 只聚合证据；每个子 Trace 仍由自己的"
            " consultation / handoff / controlled-action 权限边界执行。"
        ),
    }
    saved = save_runtime_acceptance_run(
        acceptance_run_id=f"accept_{uuid.uuid4().hex}",
        case_key=request.case_key,
        release_id=result["release_id"],
        status="passed" if passed else "failed",
        evidence=result,
        cleanup={
            "required": any(
                not child.forbid_business_write
                for child in request.child_cases
            ),
            "strategy": "follow_each_child_case_cleanup_contract",
        },
    )
    result["acceptance_run_id"] = saved.get("acceptance_run_id")
    return result
