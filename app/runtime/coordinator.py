"""V1.8 RuntimeCoordinator: one authority over the three runtime paths."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from app.handoff_policy import evaluate_handoff_policy
from app.runtime.agent_factory import build_agent_from_snapshot, vertical_agent_cards
from app.runtime.citation_renderer import (
    build_evidence_set,
    prompt_evidence_allowlist,
    render_citations,
)
from app.runtime.contracts import (
    ActionProposal,
    ActionReceipt,
    ApprovalEvent,
    RiskLevel,
    RouteDecision,
    RunState,
    RunStatus,
    RuntimePath,
    ToolEffect,
    ToolInvocation,
    content_hash,
)
from app.runtime.cost_ledger import build_cost_entry
from app.runtime.evidence_ledger import EvidenceLedger
from app.runtime.mcp_executor import (
    build_model_native_read_tools,
    preinvoke_read_tools,
)
from app.runtime.snapshot_resolver import resolve_snapshot
from app.runtime.tool_gateway import ToolGateway
from app.settings import MODEL_ID, USE_THINKING, build_model
from app.work_order_workflow import (
    action_gateway,
    advance_work_order_workflow,
    is_cancel_request,
    is_confirmation,
    is_explicit_work_order_request,
)
from db.property_db import (
    create_chat_trace,
    ensure_chat_session,
    get_chat_session,
    get_action_proposal,
    get_latest_action_proposal,
    get_pending_action_proposal,
    get_work_order_draft,
    list_action_approvals,
    now_cn,
    record_mcp_call_audit,
    record_model_call,
    record_trace_event,
    request_handoff,
    resume_handoff_after_owner_message,
    save_chat_message,
    update_chat_trace,
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {_json(payload)}\n\n"


def _extract_tool_calls(value: Any) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    candidate = getattr(value, "run_response", None) or value
    raw_calls = getattr(candidate, "tool_calls", None) or getattr(candidate, "tools", None) or []
    for raw in raw_calls:
        if isinstance(raw, dict):
            name = raw.get("tool_name") or raw.get("name") or raw.get("tool") or ""
            arguments = raw.get("arguments") or raw.get("args") or {}
        else:
            name = getattr(raw, "tool", None) or getattr(raw, "name", None) or ""
            arguments = getattr(raw, "arguments", None) or getattr(raw, "args", None) or {}
        if hasattr(arguments, "model_dump"):
            arguments = arguments.model_dump()
        elif not isinstance(arguments, dict):
            arguments = {"value": str(arguments)}
        item = {"tool_name": str(name), "arguments": arguments}
        if item not in calls:
            calls.append(item)
    return calls


def _metrics_dict(value: Any) -> Dict[str, Optional[int]]:
    metrics = getattr(value, "metrics", None)
    if not metrics:
        return {}
    result: Dict[str, Optional[int]] = {}
    for source, target in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
        ("cached_tokens", "cached_tokens"),
        ("cached_input_tokens", "cached_input_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        raw = (
            metrics.get(source)
            if isinstance(metrics, dict)
            else getattr(metrics, source, None)
        )
        if raw is not None:
            try:
                result[target] = int(raw)
            except (TypeError, ValueError):
                pass
    return result


def _estimate_tokens(text: str) -> Optional[int]:
    if not text:
        return None
    # Deliberately labelled local estimate; never used as provider usage or
    # multiplied by a price to fabricate an actual amount.
    return max(1, len(text) // 4)


def _claims_business_success(text: str) -> bool:
    normalized = text or ""
    return bool(
        re.search(
            r"(?:已|已经|正式).{0,6}(?:创建|提交|写入|更新|操作).{0,8}(?:成功|完成)|"
            r"(?:创建|提交|写入|更新|操作).{0,8}(?:成功|已完成)|"
            r"(?:资源\s*ID|工单号)[：:]\s*(?:WO|TICKET|ORDER)[-_][A-Za-z0-9_-]+",
            normalized,
            flags=re.IGNORECASE,
        )
    )


def _record_citation_violations(
    ledger: EvidenceLedger,
    violations: List[Dict[str, Any]],
) -> None:
    for violation in violations:
        code = str(violation.get("code") or "citation_violation")
        metadata = {
            key: value
            for key, value in violation.items()
            if key not in {"code", "detail"}
        }
        ledger.violation(
            code,
            str(
                violation.get("detail")
                or "Model citation was not present in the immutable EvidenceSet."
            ),
            **metadata,
        )


def _price_for_snapshot(snapshot_config: Dict[str, Any], model_id: str) -> Optional[Dict[str, Any]]:
    candidates = [
        item
        for item in snapshot_config.get("price_snapshots") or []
        if item.get("model_id") == model_id and item.get("enabled", True)
    ]
    if candidates:
        candidates.sort(key=lambda item: str(item.get("effective_date") or ""), reverse=True)
        return candidates[0]
    return None


def _model_config_for_snapshot(
    snapshot_config: Dict[str, Any],
    model_id: str,
) -> Dict[str, Any]:
    policy = snapshot_config.get("model_policy") or {}
    for item in [policy.get("default"), *(policy.get("available") or [])]:
        if isinstance(item, dict) and item.get("model_id") == model_id:
            return item
    return {}


def _build_model_from_snapshot(
    snapshot_config: Dict[str, Any],
    model_id: str,
) -> Any:
    config = _model_config_for_snapshot(snapshot_config, model_id)
    params = config.get("model_params") or {}
    overrides: Dict[str, Any] = {}
    if config.get("base_url"):
        overrides["base_url"] = config["base_url"]
    if "use_thinking" in params:
        overrides["use_thinking"] = bool(params["use_thinking"])
    return build_model(model_id, **overrides)


def _model_provider(snapshot_config: Dict[str, Any], model_id: str) -> str:
    return str(
        _model_config_for_snapshot(snapshot_config, model_id).get("provider")
        or "unknown"
    )


def _usage_for_observability(cost: Any) -> Dict[str, Any]:
    complete = cost.usage_source.value == "provider_reported_complete"
    return {
        "uncached_input_tokens": (
            max(0, int(cost.input_tokens or 0) - int(cost.cached_input_tokens or 0))
            if complete
            else None
        ),
        "cached_input_tokens": cost.cached_input_tokens if complete else None,
        "output_tokens": cost.output_tokens if complete else None,
        "reasoning_tokens": cost.reasoning_tokens,
        "total_tokens": cost.total_tokens,
        "usage_split_unavailable": not complete,
        "cost_contract": cost.model_dump(mode="json"),
    }


def _lexical_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_-]+", (text or "").lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for size in (2, 3, 4):
                for index in range(max(0, len(token) - size + 1)):
                    terms.add(token[index : index + size])
        elif len(token) >= 2:
            terms.add(token)
    return terms


def _results_from_snapshot(
    query: str,
    live_results: List[Dict[str, Any]],
    knowledge_versions: Dict[int, Dict[str, Any]],
    allowed_document_ids: set[int],
    top_k: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    published_chunks: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for doc_id in allowed_document_ids:
        document = knowledge_versions.get(doc_id) or {}
        for chunk in document.get("chunk_snapshots") or []:
            published_chunks[(doc_id, int(chunk.get("chunk_index") or 0))] = {
                **chunk,
                "document": document,
            }

    verified: List[Dict[str, Any]] = []
    seen: set[Tuple[int, int]] = set()
    for result in live_results:
        try:
            doc_id = int(result.get("doc_id", result.get("document_id")))
            chunk_index = int(result.get("chunk_index") or 0)
        except (TypeError, ValueError):
            continue
        snapshot_chunk = published_chunks.get((doc_id, chunk_index))
        content = str(result.get("content") or result.get("chunk_text") or "")
        if (
            not snapshot_chunk
            or content_hash(content) != snapshot_chunk.get("chunk_hash")
        ):
            continue
        document = snapshot_chunk["document"]
        verified.append(
            {
                **result,
                "doc_id": doc_id,
                "doc_title": document.get("title") or result.get("doc_title"),
                "content": snapshot_chunk.get("content") or content,
                "chunk_hash": snapshot_chunk.get("chunk_hash"),
                "document_hash": document.get("document_hash"),
                "document_version": document.get("document_version"),
            }
        )
        seen.add((doc_id, chunk_index))
    if verified:
        return verified[:top_k], False

    query_terms = _lexical_terms(query)
    fallback: List[Dict[str, Any]] = []
    for (doc_id, chunk_index), snapshot_chunk in published_chunks.items():
        document = snapshot_chunk["document"]
        content = str(snapshot_chunk.get("content") or "")
        overlap = query_terms & _lexical_terms(
            f"{document.get('title') or ''} {content}"
        )
        if not overlap:
            continue
        fallback.append(
            {
                "doc_id": doc_id,
                "doc_title": document.get("title") or "",
                "chunk_index": chunk_index,
                "content": content,
                "chunk_hash": snapshot_chunk.get("chunk_hash"),
                "document_hash": document.get("document_hash"),
                "document_version": document.get("document_version"),
                "score": round(len(overlap) / max(1, len(query_terms)), 6),
                "retrieval_sources": ["runtime_release_snapshot_lexical"],
            }
        )
    fallback.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
    return fallback[:top_k], True


class RuntimeCoordinator:
    """Resolve, authorize and record every owner chat run."""

    async def stream(
        self,
        message: str,
        session_id: str,
        user_id: str,
        image_analysis_ids: Optional[List[str]] = None,
    ) -> AsyncIterator[str]:
        del image_analysis_ids  # V1.8 multimodal stays on its dedicated endpoint.
        trace_id = uuid.uuid4().hex[:16]
        run_id = f"run_{uuid.uuid4().hex}"
        started = time.time()
        ensure_chat_session(session_id)
        snapshot = resolve_snapshot(session_id)
        path = self._select_path(session_id, message, snapshot.config)
        state = RunState(
            run_id=run_id,
            trace_id=trace_id,
            session_id=session_id,
            snapshot_id=snapshot.snapshot_id,
            path=path,
            status=RunStatus.RUNNING,
            next_step="resolve_snapshot",
        )
        create_chat_trace(
            trace_id=trace_id,
            session_id=session_id,
            user_message=message,
            risk_level="L2" if path == RuntimePath.CONTROLLED_ACTION else "L0",
            version_snapshot=snapshot.snapshot_hash,
        )
        save_chat_message(
            session_id=session_id,
            role="user",
            content=message,
            trace_id=trace_id,
            status="success",
            usage_source="not_applicable",
        )
        ledger = EvidenceLedger(
            trace_id=trace_id,
            session_id=session_id,
            config_snapshot={
                "snapshot_id": snapshot.snapshot_id,
                "release_id": snapshot.release_id,
                "snapshot_hash": snapshot.snapshot_hash,
            },
            release_id=snapshot.release_id,
            config_hash=snapshot.snapshot_hash,
            runtime_path=path.value,
        )
        record_trace_event(
            trace_id,
            "resolve_snapshot",
            "success",
            output_summary=f"{snapshot.release_id}/{snapshot.snapshot_hash[:12]}",
            metadata={
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "snapshot_hash": snapshot.snapshot_hash,
            },
        )
        yield _sse(
            "start",
            {
                "session_id": session_id,
                "trace_id": trace_id,
                "run_id": run_id,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "runtime_path": path.value,
            },
        )
        try:
            handoff = await self._maybe_handoff(
                message, session_id, trace_id, snapshot.release_id
            )
            if handoff:
                reply, handoff_state = handoff
                state.status = RunStatus.COMPLETED
                state.next_step = None
                ledger.capture_state(state)
                ledger.append(
                    "evaluation_results",
                    {"case": "handoff_policy", "passed": True, "state": handoff_state},
                )
                ledger.persist("complete")
                saved = save_chat_message(
                    session_id=session_id,
                    role="assistant",
                    content=reply,
                    trace_id=trace_id,
                    current_agent="人工协同控制器",
                    current_agent_id="human_copilot",
                    status="success",
                    usage_source="not_applicable",
                )
                update_chat_trace(
                    trace_id,
                    intent="handoff",
                    agent_name="人工协同控制器",
                    agent_id="human_copilot",
                    status="complete",
                )
                yield _sse("delta", {"content": reply})
                yield _sse(
                    "done",
                    {
                        "status": "complete",
                        "message_id": saved.get("id"),
                        "trace_id": trace_id,
                        "handoff": True,
                        "handoff_state": handoff_state,
                        "release_id": snapshot.release_id,
                        "snapshot_id": snapshot.snapshot_id,
                        "usage_source": "not_applicable",
                    },
                )
                return

            if path == RuntimePath.CONTROLLED_ACTION:
                async for event in self._stream_controlled_action(
                    message, session_id, trace_id, snapshot, state, ledger, started
                ):
                    yield event
                return

            async for event in self._stream_consultation(
                message,
                session_id,
                user_id,
                trace_id,
                snapshot,
                state,
                ledger,
                started,
            ):
                yield event
        except Exception as exc:
            state.status = RunStatus.FAILED
            state.next_step = None
            ledger.violation("runtime_failure", str(exc))
            ledger.capture_state(state)
            ledger.persist("failed")
            update_chat_trace(trace_id, status="failed")
            record_trace_event(
                trace_id,
                "runtime_failure",
                "failed",
                latency_ms=int((time.time() - started) * 1000),
                output_summary=str(exc)[:240],
            )
            yield _sse(
                "error",
                {
                    "error": str(exc),
                    "trace_id": trace_id,
                    "release_id": snapshot.release_id,
                    "snapshot_id": snapshot.snapshot_id,
                },
            )

    @staticmethod
    def _select_path(
        session_id: str,
        message: str,
        snapshot_config: Dict[str, Any],
    ) -> RuntimePath:
        if (
            get_work_order_draft(session_id)
            or get_pending_action_proposal(session_id)
            or is_explicit_work_order_request(message)
            or (
                is_confirmation(message)
                and get_latest_action_proposal(session_id, "work_order.create")
            )
        ):
            return RuntimePath.CONTROLLED_ACTION
        if RuntimeCoordinator._match_write_tool(snapshot_config, message):
            return RuntimePath.CONTROLLED_ACTION
        return RuntimePath.CONSULTATION

    @staticmethod
    def _match_write_tool(
        snapshot_config: Dict[str, Any],
        message: str,
    ) -> Optional[Dict[str, Any]]:
        normalized = (message or "").lower()
        candidates: List[Dict[str, Any]] = []
        gateway = ToolGateway(snapshot_config)
        for agent in snapshot_config.get("agents") or []:
            if not agent.get("enabled") or agent.get("category") in {"router", "orchestration"}:
                continue
            agent_id = str(agent.get("agent_id") or "")
            bound = set(agent.get("mcp_server_names") or [])
            for server in snapshot_config.get("mcp_servers") or []:
                server_name = str(server.get("name") or "")
                if not server.get("enabled") or server_name not in bound:
                    continue
                for tool in server.get("tools") or []:
                    tool_name = str(tool.get("name") or "")
                    if not tool_name or tool_name.lower() not in normalized:
                        continue
                    try:
                        policy = gateway.write_policy(
                            server_name, tool_name, agent_id=agent_id
                        )
                    except Exception:
                        continue
                    candidates.append(
                        {
                            "agent_id": agent_id,
                            "agent_name": str(agent.get("name") or agent_id),
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "input_schema": tool.get("input_schema") or {},
                            "policy": policy,
                        }
                    )
        return candidates[0] if len(candidates) == 1 else None

    async def _maybe_handoff(
        self,
        message: str,
        session_id: str,
        trace_id: str,
        release_id: str,
    ) -> Optional[Tuple[str, str]]:
        policy = evaluate_handoff_policy(message)
        if policy.get("should_request_handoff"):
            session = request_handoff(
                session_id,
                str(policy.get("reason") or "需要人工协同"),
                risk_level=str(policy.get("level") or "L3"),
                reason_code=str(policy.get("reason_code") or "owner_requested"),
                queue=policy.get("queue"),
                handoff_package={
                    "trace_id": trace_id,
                    "release_id": release_id,
                    "trigger_message": message,
                    "policy": policy,
                },
            )
            return (
                "已为您发起人工协同处理。AI 不会代替工作人员作出后续处理决定；"
                "您可以继续补充信息，内容会进入接管包。",
                str(session.get("handoff_status") or "requested"),
            )
        current = get_chat_session(session_id) or {}
        status = str(current.get("handoff_status") or "none")
        if status == "waiting_user":
            resumed = resume_handoff_after_owner_message(session_id)
            return "已将补充信息同步给接管工作人员，人工处理已恢复。", str(resumed.get("handoff_status") or "active")
        if status in {"requested", "active"}:
            return "当前会话正在人工协同处理中。补充信息已保存，AI 不会重复作出处理决定。", status
        return None

    async def _stream_controlled_action(
        self,
        message: str,
        session_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        state.next_step = "collect_or_resume_action"
        pending = get_pending_action_proposal(session_id)
        use_work_order = bool(
            get_work_order_draft(session_id)
            or (pending and pending.get("action_type") == "work_order.create")
            or is_explicit_work_order_request(message)
        )
        if use_work_order:
            result = advance_work_order_workflow(
                session_id,
                message,
                trace_id=trace_id,
                release_id=snapshot.release_id,
            )
        else:
            result = await self._advance_dynamic_mcp_action(
                message=message,
                session_id=session_id,
                trace_id=trace_id,
                snapshot=snapshot,
            )
        if result is None:
            raise RuntimeError("controlled action path produced no workflow result")
        selected_agent_id = str(result.get("agent_id") or "maintenance")
        selected_agent_name = str(result.get("agent_name") or "维修 Agent")
        action_type = str(result.get("action_type") or "work_order.create")
        route = RouteDecision(
            candidates=[selected_agent_id],
            selected_agent_id=selected_agent_id,
            reason=str(result.get("route_reason") or "受控维修工单流程"),
            confidence=1.0,
            required_capability_types=["action", "hitl"],
        )
        state.route_decision = route
        state.selected_agent = {
            "agent_id": selected_agent_id,
            "name": selected_agent_name,
        }
        proposal_id = result.get("proposal_id")
        proposal_row: Optional[Dict[str, Any]] = None
        if proposal_id:
            proposal_row = get_action_proposal(str(proposal_id))
            if proposal_row:
                state.pending_actions.append(
                    ActionProposal(
                        proposal_id=proposal_row["proposal_id"],
                        session_id=proposal_row["session_id"],
                        trace_id=proposal_row.get("trace_id"),
                        release_id=proposal_row.get("release_id"),
                        action_type=proposal_row["action_type"],
                        risk_level=proposal_row["risk_level"],
                        payload=proposal_row.get("payload") or {},
                        parameter_hash=content_hash(proposal_row.get("payload") or {}),
                        idempotency_key=proposal_row["idempotency_key"],
                        status=proposal_row["status"],
                    )
                )
                for approval in list_action_approvals(str(proposal_id)):
                    state.approval_events.append(
                        ApprovalEvent(
                            proposal_id=str(proposal_id),
                            decision=str(approval["decision"]),
                            actor=str(approval["actor"]),
                            parameter_hash=content_hash(proposal_row.get("payload") or {}),
                            comment=approval.get("comment"),
                            decided_at=str(approval["decided_at"]),
                        )
                    )
        receipt_data = result.get("receipt")
        if receipt_data:
            receipt = ActionReceipt.model_validate(receipt_data)
            state.action_receipts.append(receipt)
            receipt_result = receipt.result or {}
            if action_type.startswith("mcp."):
                proposal_payload = (proposal_row or {}).get("payload") or {}
                invocation = ToolInvocation(
                    server_name=str(
                        receipt_result.get("server_name")
                        or proposal_payload.get("server_name")
                        or ""
                    ),
                    tool_name=str(
                        receipt_result.get("tool_name")
                        or proposal_payload.get("tool_name")
                        or ""
                    ),
                    effect=ToolEffect(
                        str(receipt_result.get("effect") or "create")
                    ),
                    arguments=(
                        receipt_result.get("arguments")
                        or proposal_payload.get("arguments")
                        or {}
                    ),
                    discovery_status="success",
                    transport_status=(
                        "success" if receipt.may_claim_success else "failed"
                    ),
                    invocation_status=(
                        "success" if receipt.may_claim_success else "failed"
                    ),
                    business_status=str(
                        receipt_result.get("business_status")
                        or ("success" if receipt.may_claim_success else "unknown")
                    ),
                    latency_ms=receipt_result.get("latency_ms"),
                    result_summary=receipt_result.get("result_summary"),
                    error_summary=receipt.error_summary,
                    receipt_id=receipt.receipt_id,
                )
                state.tool_invocations.append(invocation)
                record_mcp_call_audit(
                    trace_id=trace_id,
                    server_name=invocation.server_name,
                    tool_name=invocation.tool_name,
                    arguments=invocation.arguments,
                    status=(
                        invocation.business_status
                        if invocation.invocation_status == "success"
                        else invocation.invocation_status
                    ),
                    result_summary=invocation.result_summary,
                    error_summary=invocation.error_summary,
                    latency_ms=invocation.latency_ms,
                    invocation_mode="confirmed_action",
                )
        if result.get("action") in {
            "awaiting_confirmation",
            "awaiting_parameters",
            "draft_updated",
            "confirmation_blocked",
        }:
            state.status = RunStatus.PAUSED
            state.next_step = "await_user_confirmation"
        elif result.get("action") == "failed":
            state.status = RunStatus.FAILED
            state.next_step = "retry_or_handoff"
        else:
            state.status = RunStatus.COMPLETED
            state.next_step = None

        reply = str(result.get("reply") or "")
        tool_call = {
            "tool_name": "action_gateway",
            "arguments": {
                "action_type": action_type,
                "proposal_id": proposal_id,
                "phase": result.get("action"),
            },
            "status": (
                "committed"
                if state.action_receipts
                and state.action_receipts[-1].may_claim_success
                else result.get("action")
            ),
            "receipt_id": (
                state.action_receipts[-1].receipt_id if state.action_receipts else None
            ),
            "resource_id": (
                state.action_receipts[-1].resource_id if state.action_receipts else None
            ),
        }
        controlled_mcp_payload = []
        for invocation in state.tool_invocations:
            payload = invocation.model_dump(mode="json")
            payload["status"] = (
                invocation.business_status
                if invocation.invocation_status == "success"
                else invocation.invocation_status
            )
            payload["invocation_mode"] = "confirmed_action"
            controlled_mcp_payload.append(payload)
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {
                "case": "action_receipt_contract",
                "passed": (
                    not _claims_business_success(reply)
                    or bool(
                        state.action_receipts
                        and state.action_receipts[-1].may_claim_success
                    )
                ),
            },
        )
        ledger.persist(
            "paused"
            if state.status == RunStatus.PAUSED
            else ("failed" if state.status == RunStatus.FAILED else "complete")
        )
        update_chat_trace(
            trace_id,
            intent=selected_agent_id,
            agent_name=selected_agent_name,
            agent_id=selected_agent_id,
            status=(
                "failed" if state.status == RunStatus.FAILED else "complete"
            ),
        )
        record_trace_event(
            trace_id,
            "action_gateway",
            "failed" if state.status == RunStatus.FAILED else "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=reply[:240],
            metadata={
                "proposal_id": proposal_id,
                "receipt_id": tool_call.get("receipt_id"),
                "resource_id": tool_call.get("resource_id"),
                "workflow_status": state.status.value,
            },
        )
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=reply,
            token_count=0,
            round_token_count=0,
            token_detail={
                "input_tokens": None,
                "output_tokens": None,
                "cached_tokens": None,
                "reasoning_tokens": None,
                "total_tokens": None,
            },
            citations=[],
            activated_skills=[],
            route_intent=selected_agent_id,
            route_reason=route.reason,
            current_agent=selected_agent_name,
            current_agent_id=selected_agent_id,
            tool_calls=[tool_call],
            model_id=None,
            thinking_enabled=False,
            model_selection_reason="controlled_action_workflow",
            trace_id=trace_id,
            status=state.status.value,
            latency_ms=int((time.time() - started) * 1000),
            mcp_calls=controlled_mcp_payload or None,
            usage_source="not_applicable",
        )
        yield _sse(
            "route",
            {
                "intent": selected_agent_id,
                "reason": route.reason,
                "current_agent": selected_agent_name,
                "current_agent_id": selected_agent_id,
                "trace_id": trace_id,
            },
        )
        yield _sse("delta", {"content": reply})
        yield _sse("tool_calls", {"tool_calls": [tool_call]})
        yield _sse(
            "done",
            {
                "status": state.status.value,
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "runtime_path": RuntimePath.CONTROLLED_ACTION.value,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "proposal_id": proposal_id,
                "action_receipts": [
                    item.model_dump(mode="json") for item in state.action_receipts
                ],
                "tool_calls": [tool_call],
                "mcp_calls": controlled_mcp_payload,
                "citations": [],
                "activated_skills": [],
                "usage_source": "not_applicable",
            },
        )

    async def _advance_dynamic_mcp_action(
        self,
        message: str,
        session_id: str,
        trace_id: str,
        snapshot: Any,
    ) -> Dict[str, Any]:
        pending = get_pending_action_proposal(session_id)
        if pending and str(pending.get("action_type") or "").startswith("mcp."):
            payload = pending.get("payload") or {}
            base = {
                "handled": True,
                "proposal_id": pending["proposal_id"],
                "action_type": pending["action_type"],
                "agent_id": payload.get("agent_id"),
                "agent_name": payload.get("agent_name") or payload.get("agent_id"),
                "route_reason": "发布快照中的写 MCP 进入受控确认路径。",
            }
            if is_cancel_request(message):
                action_gateway.reject(
                    pending["proposal_id"],
                    actor=f"owner:{session_id}",
                    comment="用户拒绝动态 MCP 写操作",
                )
                return {
                    **base,
                    "action": "rejected",
                    "reply": "已拒绝本次待确认操作；MCP 未执行，业务数据未写入。",
                }
            if not is_confirmation(message):
                return {
                    **base,
                    "action": "awaiting_confirmation",
                    "reply": (
                        f"操作 {payload.get('server_name')}/{payload.get('tool_name')} "
                        "仍在等待确认。请回复“确认提交”，或回复“拒绝”。"
                    ),
                }
            proposal = action_gateway.approve(
                pending["proposal_id"],
                actor=f"owner:{session_id}",
                comment="用户明确确认动态 MCP 写操作",
            )
            receipt = await action_gateway.execute_async(proposal.proposal_id)
            if not receipt.may_claim_success:
                return {
                    **base,
                    "action": "failed",
                    "reply": (
                        "操作未提交成功：后端没有签发包含真实资源 ID 的 committed "
                        "Receipt。不会把 MCP 调用失败包装成业务成功。"
                    ),
                    "receipt": receipt.model_dump(mode="json"),
                    "error_summary": receipt.error_summary,
                }
            return {
                **base,
                "action": "committed",
                "reply": (
                    f"操作已真实提交成功，资源 ID：{receipt.resource_id}。"
                    f"Receipt：{receipt.receipt_id}。"
                ),
                "receipt": receipt.model_dump(mode="json"),
            }

        match = self._match_write_tool(snapshot.config, message)
        if not match:
            return {
                "handled": True,
                "action": "failed",
                "action_type": "mcp.unknown",
                "agent_id": "runtime_governor",
                "agent_name": "运行时治理器",
                "route_reason": "写工具未能唯一匹配，默认拒绝。",
                "reply": "未能在当前发布快照中唯一匹配允许的写工具，操作已拒绝。",
            }

        arguments: Dict[str, Any] = {}
        json_start = message.find("{")
        if json_start >= 0:
            try:
                arguments, _ = json.JSONDecoder().raw_decode(message[json_start:])
                if not isinstance(arguments, dict):
                    raise ValueError("MCP arguments must be a JSON object")
            except Exception:
                return {
                    "handled": True,
                    "action": "awaiting_parameters",
                    "action_type": f"mcp.{match['server_name']}.{match['tool_name']}",
                    "agent_id": match["agent_id"],
                    "agent_name": match["agent_name"],
                    "route_reason": "动态写 MCP 参数收集。",
                    "reply": "检测到写工具，但 JSON 参数无法解析。请按工具输入 Schema 提供合法 JSON。",
                }
        required = list((match.get("input_schema") or {}).get("required") or [])
        missing = [key for key in required if key not in arguments]
        if missing:
            return {
                "handled": True,
                "action": "awaiting_parameters",
                "action_type": f"mcp.{match['server_name']}.{match['tool_name']}",
                "agent_id": match["agent_id"],
                "agent_name": match["agent_name"],
                "route_reason": "动态写 MCP 参数收集。",
                "reply": (
                    "该写操作尚未生成 Proposal，缺少参数："
                    + "、".join(missing)
                    + "。请附带 JSON 参数重新提交。"
                ),
            }
        action_type = f"mcp.{match['server_name']}.{match['tool_name']}"
        proposal = action_gateway.propose(
            session_id=session_id,
            action_type=action_type,
            payload={
                "agent_id": match["agent_id"],
                "agent_name": match["agent_name"],
                "server_name": match["server_name"],
                "tool_name": match["tool_name"],
                "arguments": arguments,
            },
            trace_id=trace_id,
            release_id=snapshot.release_id,
            risk_level=RiskLevel.L2,
        )
        return {
            "handled": True,
            "action": "awaiting_confirmation",
            "proposal_id": proposal.proposal_id,
            "action_type": action_type,
            "agent_id": match["agent_id"],
            "agent_name": match["agent_name"],
            "route_reason": "发布快照中的写 MCP 生成 Proposal，等待用户确认。",
            "reply": (
                f"已生成待确认 Proposal，尚未执行 {match['server_name']}/"
                f"{match['tool_name']}。\n\n参数：{_json(arguments)}\n\n"
                "确认无误请回复“确认提交”；取消请回复“拒绝”。"
            ),
        }

    async def _stream_consultation(
        self,
        message: str,
        session_id: str,
        user_id: str,
        trace_id: str,
        snapshot: Any,
        state: RunState,
        ledger: EvidenceLedger,
        started: float,
    ) -> AsyncIterator[str]:
        state.next_step = "route"
        cards = vertical_agent_cards(snapshot.config)
        if not cards:
            raise RuntimeError("published RuntimeRelease has no enabled vertical agent")

        router_started = time.time()
        from agents.router import classify_intent

        router_config = next(
            (
                item
                for item in snapshot.config.get("agents") or []
                if item.get("agent_id") == "router"
                or item.get("category") in {"router", "orchestration"}
            ),
            {},
        )
        default_model = (snapshot.config.get("model_policy") or {}).get("default") or {}
        router_model_id = str(
            router_config.get("model_id")
            or default_model.get("model_id")
            or MODEL_ID
        )
        route_result = await classify_intent(
            message,
            vertical_agents=cards,
            user_id=user_id,
            session_id=session_id,
            published_instructions=str(router_config.get("instructions") or ""),
            model=_build_model_from_snapshot(snapshot.config, router_model_id),
        )
        candidates = [str(item["agent_id"]) for item in cards]
        selected = str(
            route_result.get("target_agent_id")
            or route_result.get("intent")
            or "customer_service"
        )
        if selected not in candidates:
            selected = "customer_service" if "customer_service" in candidates else candidates[0]
        route = RouteDecision(
            candidates=candidates,
            selected_agent_id=selected,
            reason=str(route_result.get("reason") or "Router selected published agent"),
            confidence=(
                float(route_result["confidence"])
                if route_result.get("confidence") is not None
                else None
            ),
            required_capability_types=["agent", "skill", "rag", "readonly_tool"],
        )
        state.route_decision = route
        state.selected_agent = next(
            item for item in snapshot.config["agents"] if item.get("agent_id") == selected
        )
        yield _sse(
            "route",
            {
                "intent": selected,
                "reason": route.reason,
                "current_agent": state.selected_agent.get("name"),
                "current_agent_id": selected,
                "trace_id": trace_id,
            },
        )

        router_usage = route_result.get("metrics") or {}
        router_cost = build_cost_entry(
            stage="router",
            provider=_model_provider(snapshot.config, router_model_id),
            requested_model=router_model_id,
            response_model=router_model_id,
            model_policy_version=str(
                (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
            ),
            provider_usage=router_usage if router_usage else None,
            price_row=_price_for_snapshot(snapshot.config, router_model_id),
            local_estimate_tokens=_estimate_tokens(message),
        )
        state.cost_entries.append(router_cost)
        state.model_calls.append(
            {
                "stage": "router",
                "model_id": router_model_id,
                "latency_ms": int((time.time() - router_started) * 1000),
                "usage": router_usage,
                "usage_source": router_cost.usage_source.value,
            }
        )
        record_model_call(
            trace_id=trace_id,
            stage="router",
            model_id=router_model_id,
            model_selection_reason=f"published snapshot route to {selected}",
            latency_ms=int((time.time() - router_started) * 1000),
            input_tokens=router_cost.input_tokens,
            output_tokens=router_cost.output_tokens,
            reasoning_tokens=router_cost.reasoning_tokens,
            cached_tokens=router_cost.cached_input_tokens,
            total_tokens=router_cost.total_tokens,
            usage_source=router_cost.usage_source.value,
            price_snapshot=(
                router_cost.price_snapshot.model_dump(mode="json")
                if router_cost.price_snapshot
                else None
            ),
            estimated_cost_cny=router_cost.amount,
            usage_normalized=_usage_for_observability(router_cost),
        )

        state.next_step = "retrieve"
        retrieval_started = time.time()
        allowed_doc_ids = {
            int(item) for item in state.selected_agent.get("knowledge_doc_ids") or []
        }
        knowledge_versions = {
            int(item["knowledge_doc_id"]): item
            for item in snapshot.config.get("knowledge") or []
        }
        results: List[Dict[str, Any]] = []
        retrieval_status = "not_requested"
        if allowed_doc_ids:
            try:
                import rag_retrieval

                retrieval = await asyncio.to_thread(
                    rag_retrieval.advanced_search,
                    message,
                    snapshot.config.get("retrieval_policy") or {},
                )
                results = list((retrieval or {}).get("results") or [])
                results, used_snapshot_fallback = _results_from_snapshot(
                    message,
                    results,
                    knowledge_versions,
                    allowed_doc_ids,
                    int(
                        (snapshot.config.get("retrieval_policy") or {}).get("top_k")
                        or 5
                    ),
                )
                retrieval_status = (
                    "completed_snapshot_fallback"
                    if used_snapshot_fallback
                    else "completed"
                )
            except Exception as exc:
                results, _ = _results_from_snapshot(
                    message,
                    [],
                    knowledge_versions,
                    allowed_doc_ids,
                    int(
                        (snapshot.config.get("retrieval_policy") or {}).get("top_k")
                        or 5
                    ),
                )
                retrieval_status = (
                    "completed_snapshot_fallback" if results else "failed"
                )
                ledger.violation(
                    "live_retrieval_failed",
                    str(exc),
                    snapshot_fallback_count=len(results),
                )
        evidence = build_evidence_set(
            message,
            results,
            knowledge_versions=knowledge_versions,
            allowed_document_ids=allowed_doc_ids,
            retrieval_status=retrieval_status,
        )
        state.retrieval_evidence = evidence
        record_trace_event(
            trace_id,
            "retrieval",
            "failed" if retrieval_status == "failed" else "success",
            latency_ms=int((time.time() - retrieval_started) * 1000),
            output_summary=f"{len(evidence.items)} evidence items",
            metadata={
                "snapshot_id": snapshot.snapshot_id,
                "allowed_document_ids": sorted(allowed_doc_ids),
                "evidence_ids": [item.evidence_id for item in evidence.items],
            },
        )

        state.next_step = "readonly_mcp"
        mcp_context, invocations = await preinvoke_read_tools(
            snapshot.config, selected, message
        )
        preinvoked_servers = {
            invocation.server_name
            for invocation in invocations
            if invocation.tool_name != "discovery"
        }
        model_native_toolkits = build_model_native_read_tools(
            snapshot.config,
            selected,
            excluded_servers=preinvoked_servers,
        )
        state.tool_invocations = list(invocations)
        for invocation in invocations:
            record_trace_event(
                trace_id,
                f"mcp.{invocation.server_name}.{invocation.tool_name}",
                (
                    "success"
                    if invocation.invocation_status == "success"
                    and invocation.business_status == "success"
                    else "failed"
                ),
                output_summary=invocation.result_summary or invocation.error_summary,
                metadata=invocation.model_dump(mode="json"),
            )
            audit_status = (
                invocation.business_status
                if invocation.invocation_status == "success"
                else (
                    invocation.transport_status
                    if invocation.transport_status in {"timeout", "failed"}
                    else invocation.invocation_status
                )
            )
            record_mcp_call_audit(
                trace_id=trace_id,
                server_name=invocation.server_name,
                tool_name=invocation.tool_name,
                arguments=invocation.arguments,
                status=audit_status,
                result_summary=invocation.result_summary,
                error_summary=invocation.error_summary,
                latency_ms=invocation.latency_ms,
                invocation_mode="policy_preinvoke",
            )

        evidence_prompt = prompt_evidence_allowlist(evidence)
        build = build_agent_from_snapshot(
            snapshot,
            selected,
            message,
            tools=model_native_toolkits,
            evidence_prompt=evidence_prompt + mcp_context,
        )
        state.activated_skills = build.activated_skills
        for call in build.skill_tool_calls:
            record_trace_event(
                trace_id,
                f"skill.{call['skill_id']}.get_skill_instructions",
                "success",
                output_summary=(
                    f"loaded Skill {call['skill_id']} "
                    f"version={call['skill_version']}"
                ),
                metadata=call,
            )
        state.next_step = "answer"
        agent_started = time.time()
        contextual_message = (
            "[运行边界] 本轮是只读技术栈咨询路径，不得创建工单、草稿或任何待确认 Action。\n"
            + message
        )
        full_content = ""
        tool_calls: List[Dict[str, Any]] = list(build.skill_tool_calls)
        final_metrics: Dict[str, Optional[int]] = {}
        async for chunk in build.agent.arun(
            contextual_message,
            user_id=user_id,
            session_id=session_id,
            stream=True,
        ):
            content = getattr(chunk, "content", None) or getattr(chunk, "delta", None)
            if content:
                full_content += str(content)
            for call in _extract_tool_calls(chunk):
                if call not in tool_calls:
                    tool_calls.append(call)
            metrics = _metrics_dict(chunk)
            if metrics:
                final_metrics.update(metrics)

        model_native_invocations = []
        for toolkit in model_native_toolkits:
            model_native_invocations.extend(
                list(getattr(toolkit, "recorded_invocations", []) or [])
            )
            if hasattr(toolkit, "close"):
                try:
                    await asyncio.wait_for(toolkit.close(), timeout=3)
                except Exception:
                    pass
        state.tool_invocations.extend(model_native_invocations)
        for invocation in model_native_invocations:
            record_trace_event(
                trace_id,
                f"mcp.{invocation.server_name}.{invocation.tool_name}",
                (
                    "success"
                    if invocation.invocation_status == "success"
                    and invocation.business_status == "success"
                    else "failed"
                ),
                latency_ms=invocation.latency_ms,
                output_summary=invocation.result_summary or invocation.error_summary,
                metadata=invocation.model_dump(mode="json"),
            )
            record_mcp_call_audit(
                trace_id=trace_id,
                server_name=invocation.server_name,
                tool_name=invocation.tool_name,
                arguments=invocation.arguments,
                status=(
                    invocation.business_status
                    if invocation.invocation_status == "success"
                    else invocation.invocation_status
                ),
                result_summary=invocation.result_summary,
                error_summary=invocation.error_summary,
                latency_ms=invocation.latency_ms,
                invocation_mode="model_native",
            )

        loaded_skill_tool = any(
            call.get("tool_name") == "get_skill_instructions" for call in tool_calls
        )
        if build.activated_skills and not loaded_skill_tool:
            ledger.violation(
                "skill_selected_not_loaded",
                "Skill trigger matched, but Agno get_skill_instructions was not observed.",
                selected_skill_ids=[
                    item.skill_id for item in build.activated_skills
                ],
            )

        rendered, citations, citation_violations = render_citations(
            full_content, evidence
        )
        state.citations = citations
        _record_citation_violations(ledger, citation_violations)

        model_id = str(
            state.selected_agent.get("model_id")
            or (
                (snapshot.config.get("model_policy") or {}).get("default") or {}
            ).get("model_id")
            or MODEL_ID
        )
        vertical_cost = build_cost_entry(
            stage="vertical_agent",
            provider=_model_provider(snapshot.config, model_id),
            requested_model=model_id,
            response_model=model_id,
            model_policy_version=str(
                (snapshot.config.get("model_policy") or {}).get("version") or "v1.8"
            ),
            provider_usage=final_metrics if final_metrics else None,
            price_row=_price_for_snapshot(snapshot.config, model_id),
            local_estimate_tokens=_estimate_tokens(contextual_message + rendered),
        )
        state.cost_entries.append(vertical_cost)
        state.model_calls.append(
            {
                "stage": "vertical_agent",
                "model_id": model_id,
                "latency_ms": int((time.time() - agent_started) * 1000),
                "usage": final_metrics,
                "usage_source": vertical_cost.usage_source.value,
            }
        )
        record_model_call(
            trace_id=trace_id,
            stage="vertical_agent",
            model_id=model_id,
            model_selection_reason=f"agent model from snapshot:{selected}",
            latency_ms=int((time.time() - agent_started) * 1000),
            input_tokens=vertical_cost.input_tokens,
            output_tokens=vertical_cost.output_tokens,
            reasoning_tokens=vertical_cost.reasoning_tokens,
            cached_tokens=vertical_cost.cached_input_tokens,
            total_tokens=vertical_cost.total_tokens,
            usage_source=vertical_cost.usage_source.value,
            price_snapshot=(
                vertical_cost.price_snapshot.model_dump(mode="json")
                if vertical_cost.price_snapshot
                else None
            ),
            estimated_cost_cny=vertical_cost.amount,
            usage_normalized=_usage_for_observability(vertical_cost),
        )

        state.status = RunStatus.COMPLETED
        state.next_step = None
        ledger.capture_state(state)
        ledger.append(
            "evaluation_results",
            {
                "case": "consultation_no_write",
                "passed": not state.pending_actions and not state.action_receipts,
            },
        )
        ledger.append(
            "evaluation_results",
            {
                "case": "citation_allowlist",
                "passed": not citation_violations,
                "violations": citation_violations,
            },
        )
        ledger.persist("complete")
        update_chat_trace(
            trace_id,
            intent=selected,
            agent_name=str(state.selected_agent.get("name") or selected),
            agent_id=selected,
            status="complete",
        )
        record_trace_event(
            trace_id,
            "final_response",
            "success",
            latency_ms=int((time.time() - started) * 1000),
            output_summary=rendered[:240],
            metadata={
                "snapshot_id": snapshot.snapshot_id,
                "activated_skill_ids": [
                    item.skill_id for item in state.activated_skills
                ],
                "evidence_ids": [item.evidence_id for item in evidence.items],
                "citation_evidence_ids": [
                    item.evidence_id for item in citations
                ],
                "mcp_invocation_ids": [
                    item.invocation_id for item in state.tool_invocations
                ],
            },
        )

        citations_payload = []
        for item in citations:
            payload = item.model_dump(mode="json")
            payload.update(
                {
                    "doc_id": item.document_id,
                    "doc_title": item.title,
                    "content": item.content_snapshot,
                    "used_in_answer": True,
                }
            )
            citations_payload.append(payload)
        skills_payload = [
            item.model_dump(mode="json") for item in state.activated_skills
        ]
        mcp_payload = []
        for item in state.tool_invocations:
            payload = item.model_dump(mode="json")
            payload["status"] = (
                item.business_status
                if item.invocation_status == "success"
                else (
                    item.transport_status
                    if item.transport_status in {"timeout", "failed"}
                    else item.invocation_status
                )
            )
            payload["invocation_mode"] = (
                "model_native"
                if item in model_native_invocations
                else "policy_preinvoke"
            )
            mcp_payload.append(payload)
        token_count = vertical_cost.total_tokens or 0
        saved = save_chat_message(
            session_id=session_id,
            role="assistant",
            content=rendered,
            token_count=token_count,
            round_token_count=(router_cost.total_tokens or 0) + token_count,
            token_detail={
                "input_tokens": vertical_cost.input_tokens,
                "output_tokens": vertical_cost.output_tokens,
                "reasoning_tokens": vertical_cost.reasoning_tokens,
                "cached_tokens": vertical_cost.cached_input_tokens,
                "total_tokens": vertical_cost.total_tokens,
                "local_estimate_tokens": vertical_cost.local_estimate_tokens,
            },
            citations=citations_payload,
            activated_skills=skills_payload,
            route_intent=selected,
            route_reason=route.reason,
            current_agent=str(state.selected_agent.get("name") or selected),
            current_agent_id=selected,
            tool_calls=tool_calls or None,
            model_id=model_id,
            thinking_enabled=USE_THINKING,
            model_selection_reason=f"published snapshot:{snapshot.release_id}",
            trace_id=trace_id,
            status="success",
            latency_ms=int((time.time() - agent_started) * 1000),
            mcp_calls=mcp_payload or None,
            usage_source=vertical_cost.usage_source.value,
        )

        # Buffering until citation validation is intentional: the answer text,
        # final citations and clickable snapshots are emitted from one structure.
        yield _sse(
            "delta",
            {
                "content": rendered,
                "current_agent": state.selected_agent.get("name"),
                "current_agent_id": selected,
            },
        )
        if tool_calls:
            yield _sse("tool_calls", {"tool_calls": tool_calls})
        yield _sse(
            "done",
            {
                "status": "complete",
                "message_id": saved.get("id"),
                "trace_id": trace_id,
                "runtime_path": RuntimePath.CONSULTATION.value,
                "release_id": snapshot.release_id,
                "snapshot_id": snapshot.snapshot_id,
                "current_agent": state.selected_agent.get("name"),
                "current_agent_id": selected,
                "route_intent": selected,
                "route_reason": route.reason,
                "citations": citations_payload,
                "activated_skills": skills_payload,
                "tool_calls": tool_calls,
                "mcp_calls": mcp_payload,
                "cost_entries": [
                    item.model_dump(mode="json") for item in state.cost_entries
                ],
                "usage_source": vertical_cost.usage_source.value,
                "token_count": token_count,
                "round_token_count": (router_cost.total_tokens or 0) + token_count,
                "auto_badcase_id": None,
            },
        )
