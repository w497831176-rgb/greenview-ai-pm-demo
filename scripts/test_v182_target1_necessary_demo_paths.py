"""Offline contracts for target 1: the necessary A/B/C demo paths.

This script uses a fresh temporary SQLite database, never opens the network,
and never invokes a model Provider. Existing focused tests cover A and C; this
file closes the B-RAG, B-Tool, and controlled-write behavior gaps.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any


TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-target1-demo-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
for key in (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
):
    os.environ[key] = ""


from db.property_db import (  # noqa: E402
    _get_conn,
    ensure_chat_session,
    get_action_proposal,
    get_chat_session,
    get_work_order,
    init_db,
)


# app.settings constructs the default model object during import and reads the
# model configuration table. Initialize only this fresh temporary database
# before importing any app runtime module.
init_db()


from app.runtime import mcp_executor  # noqa: E402
from app.runtime.agent_factory import build_agent_from_snapshot  # noqa: E402
from app.runtime.contracts import (  # noqa: E402
    LaneDecision,
    RunConfigSnapshot,
    RuntimeLane,
    RuntimePath,
    ToolEffect,
    content_hash,
)
from app.runtime.coordinator import (  # noqa: E402
    _answer_contract_for,
    _requires_rag_citation,
    _results_from_snapshot,
)
from app.runtime.tool_planner import plan_tools  # noqa: E402
from app.work_order_workflow import advance_work_order_workflow  # noqa: E402


RAG_QUERY = "紧急维修的登记和到场时限是什么？"
TOOL_QUERY = "查询我最近的维修工单。"


def _scalar(sql: str) -> int:
    conn = _get_conn()
    try:
        return int(conn.execute(sql).fetchone()[0])
    finally:
        conn.close()


def _provider_attempt_count() -> int:
    return _scalar(
        "SELECT COUNT(*) FROM model_calls WHERE record_kind = 'provider_attempt'"
    )


def _snapshot(session_id: str):
    knowledge_content = (
        "第二章 响应时效承诺\n"
        "一、紧急维修：接到报修后，物业客服中心在 5 分钟内完成工单登记并通知工程人员，"
        "工程人员 30 分钟内到场处置。"
    )
    tool_policy = {
        "server_id": 7,
        "server_name": "workorder-server",
        "tool_name": "get_my_recent_work_orders",
        "effect": "read",
        "risk_level": "L0",
        "allowed_paths": ["consultation"],
        "requires_confirmation": False,
        "enabled": True,
        "policy_reason": "target1 offline fixture",
    }
    return RunConfigSnapshot(
        snapshot_id=f"snapshot-{session_id}",
        release_id="rr-target1-offline",
        snapshot_hash="target1-offline-snapshot",
        session_id=session_id,
        created_at="2026-08-07T00:00:00+08:00",
        config={
            "agents": [
                {
                    "agent_id": "maintenance",
                    "name": "维修 Agent",
                    "enabled": True,
                    "category": "maintenance",
                    "domain_scope": "property",
                    "instructions": "处理维修咨询和只读工单查询。",
                    "skill_ids": [8],
                    "mcp_server_names": ["workorder-server"],
                    "knowledge_doc_ids": [1],
                }
            ],
            "skills": [
                {
                    "skill_id": 8,
                    "name": "维修工单处理",
                    "description": "维修咨询与工单处理规则",
                    "version": "1.0.0",
                    "enabled": True,
                    "trigger_condition": "维修,报修,工单",
                    "metadata": {
                        "positive_triggers": ["维修", "报修", "工单"]
                    },
                    "content_hash": "skill-eight-target1",
                    "reference_snapshots": [],
                    "instructions_fallback": "先核实维修事项；只读查询不得写入业务数据。",
                }
            ],
            "knowledge": [
                {
                    "knowledge_doc_id": 1,
                    "title": "物业维修服务承诺",
                    "document_hash": "doc-one-target1",
                    "document_version": "v1",
                    "chunk_snapshots": [
                        {
                            "chunk_index": 1,
                            "content": knowledge_content,
                            "chunk_hash": content_hash(knowledge_content),
                        }
                    ],
                }
            ],
            "mcp_servers": [
                {
                    "id": 7,
                    "name": "workorder-server",
                    "enabled": True,
                    "command": "fake-workorder",
                    "args": [],
                    "tools": [
                        {
                            "name": "get_my_recent_work_orders",
                            "description": "查询我的最近维修工单",
                            "input_schema": {
                                "type": "object",
                                "properties": {"limit": {"type": "integer"}},
                            },
                            "policy": tool_policy,
                        }
                    ],
                }
            ],
            "retrieval_policy": {"top_k": 5, "context_threshold": 0.2},
            "model_policy": {
                "default": {
                    "model_id": "deepseek-v4-flash",
                    "provider": "deepseek",
                    "model_params": {"use_thinking": True},
                },
                "available": [],
            },
        },
    )


def test_b_rag_uses_bound_snapshot_and_skill() -> None:
    snapshot = _snapshot("target1-b-rag")
    maintenance = next(
        item
        for item in snapshot.config.get("agents") or []
        if item.get("agent_id") == "maintenance" and item.get("enabled")
    )
    allowed_document_ids = {
        int(item) for item in maintenance.get("knowledge_doc_ids") or []
    }
    knowledge_versions = {
        int(item["knowledge_doc_id"]): item
        for item in snapshot.config.get("knowledge") or []
        if int(item.get("knowledge_doc_id") or 0) in allowed_document_ids
    }
    results, used_snapshot = _results_from_snapshot(
        RAG_QUERY,
        [],
        knowledge_versions,
        allowed_document_ids,
        int((snapshot.config.get("retrieval_policy") or {}).get("top_k") or 5),
        float(
            (snapshot.config.get("retrieval_policy") or {}).get(
                "context_threshold"
            )
            or 0.2
        ),
    )
    matching = [
        item
        for item in results
        if int(item.get("doc_id") or 0) == 1
        and int(item.get("chunk_index") or -1) == 1
    ]
    assert used_snapshot
    assert matching, results
    compact = str(matching[0].get("content") or "").replace(" ", "")
    assert "5分钟" in compact and "30分钟" in compact

    build = build_agent_from_snapshot(snapshot, "maintenance", RAG_QUERY)
    assert [item.skill_id for item in build.activated_skills] == [8], {
        "activated": [item.skill_id for item in build.activated_skills],
        "decisions": build.skill_decisions,
        "bound_skill_ids": maintenance.get("skill_ids") or [],
    }
    assert [item.get("skill_id") for item in build.skill_tool_calls] == [8]
    assert all(item.get("status") == "success" for item in build.skill_tool_calls)
    # The Skill is already loaded into immutable instructions, so its access
    # tool is intentionally hidden from the model after pre-invocation.
    assert build.agent.skills is None


class _FakeFunction:
    async def entrypoint(self, **arguments: Any) -> dict[str, Any]:
        assert arguments == {"limit": 5}
        return {
            "status": "success",
            "data": [
                {
                    "work_order_id": "WO-TARGET1-001",
                    "status": "处理中",
                }
            ],
        }


class _FakeMCPTools:
    def __init__(self, **_: Any):
        self.functions: dict[str, Any] = {}

    async def __aenter__(self):
        self.functions = {"get_my_recent_work_orders": _FakeFunction()}
        return self

    async def close(self) -> None:
        return None


def test_b_tool_is_successful_evidence_without_unrelated_rag() -> None:
    snapshot = _snapshot("target1-b-tool")
    plans = plan_tools(
        snapshot.config,
        "maintenance",
        TOOL_QUERY,
        RuntimePath.CONSULTATION,
        effects=[ToolEffect.READ],
        execution_modes=["auto_preinvoke"],
    )
    assert [(item.server_name, item.tool_name) for item in plans] == [
        ("workorder-server", "get_my_recent_work_orders")
    ]
    assert plans[0].arguments == {"limit": 5}

    original = mcp_executor.MCPTools
    mcp_executor.MCPTools = _FakeMCPTools
    try:
        context, invocations = asyncio.run(
            mcp_executor.preinvoke_read_tools(
                snapshot.config,
                "maintenance",
                TOOL_QUERY,
            )
        )
    finally:
        mcp_executor.MCPTools = original

    assert len(invocations) == 1
    invocation = invocations[0]
    assert invocation.invocation_status == "success"
    assert invocation.business_status == "success"
    assert "WO-TARGET1-001" in context

    contract = _answer_contract_for(
        LaneDecision(
            lane=RuntimeLane.PROPERTY_GOVERNED,
            business_intent="read_recent_work_orders",
        )
    )
    assert not _requires_rag_citation(
        contract,
        evidence_count=4,
        linked_skill_evidence_count=0,
        successful_tool_evidence_count=1,
    )
    assert _requires_rag_citation(
        contract,
        evidence_count=1,
        linked_skill_evidence_count=0,
        successful_tool_evidence_count=0,
    )


def test_controlled_write_requires_confirmation_and_is_idempotent() -> None:
    session_id = "target1-controlled-write"
    ensure_chat_session(session_id)
    before_orders = _scalar("SELECT COUNT(*) FROM work_orders")
    first = advance_work_order_workflow(
        session_id,
        (
            "请创建维修工单：房号 3-2-1201，厨房水槽持续漏水，紧急，"
            "联系人测试业主，电话 13800138000，尽快上门。"
        ),
        trace_id="trace-target1-proposal",
        release_id="rr-target1-test",
    )
    assert first and first["action"] == "awaiting_confirmation"
    assert _scalar("SELECT COUNT(*) FROM work_orders") == before_orders
    proposal = get_action_proposal(first["proposal_id"])
    assert proposal and proposal["status"] == "pending_confirmation"
    assert get_chat_session(session_id)["handoff_status"] == "none"

    committed = advance_work_order_workflow(
        session_id,
        "确认创建",
        trace_id="trace-target1-confirm",
        release_id="rr-target1-test",
    )
    assert committed and committed["action"] == "committed"
    receipt = committed.get("receipt") or {}
    assert receipt.get("status") == "committed"
    assert receipt.get("receipt_id") and receipt.get("resource_id")
    assert get_work_order(receipt["resource_id"])
    assert _scalar("SELECT COUNT(*) FROM work_orders") == before_orders + 1

    replay = advance_work_order_workflow(session_id, "确认创建")
    assert replay and replay["action"] == "idempotent_replay"
    assert replay["work_order_id"] == committed["work_order_id"]
    assert (replay.get("receipt") or {}).get("receipt_id") == receipt["receipt_id"]
    assert _scalar("SELECT COUNT(*) FROM work_orders") == before_orders + 1
    assert get_chat_session(session_id)["handoff_status"] == "none"


def main() -> None:
    before_attempts = _provider_attempt_count()
    tests = [
        test_b_rag_uses_bound_snapshot_and_skill,
        test_b_tool_is_successful_evidence_without_unrelated_rag,
        test_controlled_write_requires_confirmation_and_is_idempotent,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    assert _provider_attempt_count() == before_attempts == 0
    print("Target 1 necessary demo paths: PASS (Provider attempts: 0)")


if __name__ == "__main__":
    try:
        main()
    finally:
        TEMP_DIR.cleanup()
