"""No-model V1.8 runtime convergence contract checks."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-v180-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
os.environ["RUNTIME_ENGINE"] = "v18"

from app.runtime.acceptance import ACCEPTANCE_CASES
from app.runtime.action_gateway import ActionGateway
from app.runtime.citation_renderer import build_evidence_set, render_citations
from app.runtime.contracts import ToolEffect
from app.runtime.cost_ledger import build_cost_entry
from app.runtime.release_compiler import (
    compile_tool_policy,
    ensure_bootstrap_release,
    publish_compiled_release,
)
from app.runtime.snapshot_resolver import resolve_snapshot
from db.property_db import get_work_order, init_db


def test_release_and_snapshot_immutability():
    init_db()
    release_1 = ensure_bootstrap_release()
    snapshot_1 = resolve_snapshot("session-old")
    release_2 = publish_compiled_release(created_by="contract-test")
    snapshot_old_again = resolve_snapshot("session-old")
    snapshot_new = resolve_snapshot("session-new")
    assert release_1["status"] in {"published", "superseded"}
    assert release_2["status"] == "published", release_2.get("validation")
    assert snapshot_old_again.snapshot_hash == snapshot_1.snapshot_hash
    assert snapshot_old_again.release_id == snapshot_1.release_id
    assert snapshot_new.release_id == release_2["release_id"]


def test_tool_policy_default_deny():
    server = {"id": 1, "name": "demo", "enabled": True}
    read = compile_tool_policy(server, {"name": "get_status", "tool_metadata": {}})
    write = compile_tool_policy(server, {"name": "create_record", "tool_metadata": {}})
    delete = compile_tool_policy(server, {"name": "delete_record", "tool_metadata": {}})
    unknown = compile_tool_policy(server, {"name": "do_magic", "tool_metadata": {}})
    assert read.effect == ToolEffect.READ and read.enabled
    assert write.effect == ToolEffect.CREATE and write.requires_confirmation
    assert delete.effect == ToolEffect.DELETE and not delete.enabled
    assert unknown.effect == ToolEffect.UNKNOWN and not unknown.enabled


def test_citation_single_source_contract():
    evidence = build_evidence_set(
        "滑梯安全",
        [
            {
                "doc_id": 7,
                "doc_title": "儿童游乐区安全制度",
                "chunk_index": 2,
                "content": "发现儿童受伤应立即停止设施使用并联系工作人员。",
                "score": 0.91,
                "retrieval_sources": ["keyword", "semantic"],
            }
        ],
        allowed_document_ids={7},
    )
    evidence_id = evidence.items[0].evidence_id
    rendered, citations, violations = render_citations(
        f"应立即停用设施并联系工作人员 [[evidence:{evidence_id}]]。",
        evidence,
    )
    assert rendered.endswith("【引用1】。")
    assert citations[0].content_snapshot == evidence.items[0].content_snapshot
    assert not violations
    rendered_bad, bad_citations, bad_violations = render_citations(
        "错误引用 [[evidence:ev_not_allowed]]", evidence
    )
    assert "ev_not_allowed" not in rendered_bad
    assert not bad_citations
    assert bad_violations[0]["code"] == "invalid_evidence_id"
    rendered_malformed, malformed_citations, malformed_violations = (
        render_citations(
            "天气工具结果 [[evidence:MCP weather-server]]",
            evidence,
        )
    )
    assert "[[evidence:" not in rendered_malformed
    assert not malformed_citations
    assert malformed_violations == [
        {
            "code": "invalid_evidence_id",
            "evidence_id": "MCP weather-server",
        }
    ]
    unsupported, unsupported_citations, unsupported_violations = (
        render_citations(
            (
                "杭州明天天气阴，温度30度，适合户外作业 "
                f"[[evidence:{evidence_id}]]"
            ),
            evidence,
        )
    )
    assert "[[evidence:" not in unsupported
    assert not unsupported_citations
    assert unsupported_violations[0]["code"] == (
        "unsupported_evidence_citation"
    )
    assert unsupported_violations[0]["evidence_id"] == evidence_id
    section_supported, section_citations, section_violations = render_citations(
        (
            "#### 紧急维修\n"
            f"- 依据 **[[evidence:{evidence_id}]]**\n"
            "- 发现设施松动时应立即停用并联系工作人员。"
        ),
        evidence,
    )
    assert "【引用1】" in section_supported
    assert len(section_citations) == 1
    assert not section_violations
    unstructured, unstructured_citations, unstructured_violations = (
        render_citations("天气条件适合维修 [[MCP天气建议]]", evidence)
    )
    assert "[[" not in unstructured
    assert not unstructured_citations
    assert unstructured_violations == [
        {
            "code": "unstructured_reference_marker",
            "marker": "MCP天气建议",
        }
    ]


def test_cost_availability_contract():
    price = {
        "model_id": "demo-model",
        "currency": "CNY",
        "effective_date": "2026-07-18",
        "input_price_per_1m": 1.0,
        "cached_input_price_per_1m": 0.2,
        "output_price_per_1m": 2.0,
    }
    total_only = build_cost_entry(
        "router",
        "demo",
        "demo-model",
        "demo-model",
        "v1",
        {"total_tokens": 100},
        price,
        90,
    )
    assert total_only.usage_source.value == "provider_reported_total_only"
    assert total_only.amount is None and total_only.formula is None
    estimated = build_cost_entry(
        "agent", "demo", "demo-model", "demo-model", "v1", None, price, 120
    )
    assert estimated.usage_source.value == "local_estimate"
    assert estimated.amount is None
    complete = build_cost_entry(
        "agent",
        "demo",
        "demo-model",
        "demo-model",
        "v1",
        {
            "input_tokens": 100,
            "cached_tokens": 20,
            "output_tokens": 50,
            "total_tokens": 150,
        },
        price,
    )
    assert complete.usage_source.value == "provider_reported_complete"
    assert complete.amount is not None and complete.formula


def test_action_gateway_receipt_and_idempotency():
    gateway = ActionGateway()
    payload = {
        "room_id": "3-2-1201",
        "issue_type": "水电",
        "issue_desc": "V1.8 契约测试漏水",
        "urgency": "中",
        "contact_name": "测试业主",
        "contact_phone": "13800138000",
        "appointment_time": "明天下午",
    }
    rejected = gateway.propose("reject-session", "work_order.create", payload)
    gateway.reject(rejected.proposal_id, "contract-test")
    assert get_work_order("should-not-exist") is None

    proposal = gateway.propose("confirm-session", "work_order.create", payload)
    gateway.approve(proposal.proposal_id, "contract-test")
    receipt_1 = gateway.execute(proposal.proposal_id)
    receipt_2 = gateway.execute(proposal.proposal_id)
    assert receipt_1.may_claim_success
    assert receipt_1.receipt_id == receipt_2.receipt_id
    assert receipt_1.resource_id == receipt_2.resource_id
    assert get_work_order(receipt_1.resource_id) is not None


def test_action_success_claim_contract():
    from app.runtime.coordinator import _claims_business_success

    assert not _claims_business_success(
        "维修工单草稿已完整，尚未创建正式工单。"
        "问题描述：请帮我创建维修工单：3-2-1201卫生间漏水。"
    )
    assert not _claims_business_success("已拒绝本次操作，业务数据未写入。")
    assert _claims_business_success(
        "正式维修工单已创建成功，工单号：WO-20260719-DEMO。"
    )


def test_mcp_preinvoke_initializes_session():
    from app.runtime import mcp_executor

    class FakeFunction:
        async def entrypoint(self, **arguments):
            return {
                "status": "success",
                "arguments": arguments,
                "weather": "sunny",
            }

    class FakeMCPTools:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.functions = {}
            self.entered = False
            self.closed = False
            self.__class__.instances.append(self)

        async def __aenter__(self):
            self.entered = True
            self.functions = {
                "get_current_weather": FakeFunction(),
                "get_weather_advice": FakeFunction(),
            }
            return self

        async def close(self):
            self.closed = True

    def policy(tool_name):
        return {
            "server_id": 1,
            "server_name": "weather-server",
            "tool_name": tool_name,
            "effect": "read",
            "risk_level": "L0",
            "allowed_paths": ["consultation"],
            "requires_confirmation": False,
            "enabled": True,
            "policy_reason": "contract fixture",
        }

    config = {
        "agents": [
            {
                "agent_id": "maintenance",
                "enabled": True,
                "mcp_server_names": ["weather-server"],
            }
        ],
        "mcp_servers": [
            {
                "id": 1,
                "name": "weather-server",
                "enabled": True,
                "command": "fake-weather",
                "args": [],
                "tools": [
                    {
                        "name": "get_current_weather",
                        "policy": policy("get_current_weather"),
                    },
                    {
                        "name": "get_weather_advice",
                        "policy": policy("get_weather_advice"),
                    },
                ],
            }
        ],
    }
    original = mcp_executor.MCPTools
    mcp_executor.MCPTools = FakeMCPTools
    try:
        context, invocations = asyncio.run(
            mcp_executor.preinvoke_read_tools(
                config,
                "maintenance",
                "请查询本小区明天天气并给出风险建议。",
            )
        )
    finally:
        mcp_executor.MCPTools = original

    assert FakeMCPTools.instances[0].entered
    assert FakeMCPTools.instances[0].closed
    assert len(invocations) == 2
    assert all(item.invocation_status == "success" for item in invocations)
    assert "weather-server/get_current_weather" in context
    assert "weather-server/get_weather_advice" in context


def test_mcp_business_result_unwrap_contract():
    from app.runtime.mcp_executor import _business_status, _structured_result

    class FakeAgnoResult:
        def __init__(self):
            self.content = (
                '{"status":"success","data":{"city":"杭州","condition":"阴"}}'
            )
            self.metadata = {
                "structured_content": {
                    "result": (
                        '{"status":"success","data":{"city":"杭州",'
                        '"condition":"阴"}}'
                    )
                }
            }

    parsed = _structured_result(FakeAgnoResult())
    status, summary = _business_status(FakeAgnoResult())
    assert parsed == {
        "status": "success",
        "data": {"city": "杭州", "condition": "阴"},
    }
    assert status == "success"
    assert '"status": "success"' in summary


def test_citation_violation_recording_contract():
    from app.runtime.coordinator import _record_citation_violations

    class FakeLedger:
        def __init__(self):
            self.calls = []

        def violation(self, code, detail, **metadata):
            self.calls.append(
                {"code": code, "detail": detail, "metadata": metadata}
            )

    ledger = FakeLedger()
    _record_citation_violations(
        ledger,
        [{"code": "invalid_evidence_id", "evidence_id": "ev_unknown"}],
    )
    assert ledger.calls == [
        {
            "code": "invalid_evidence_id",
            "detail": (
                "Model citation was not present in the immutable EvidenceSet."
            ),
            "metadata": {"evidence_id": "ev_unknown"},
        }
    ]


def test_v18_runtime_auto_badcase_contract():
    from app.runtime.badcase_capture import (
        capture_runtime_badcase,
        runtime_badcase_trigger,
    )
    from app.runtime.contracts import RunEvidenceLedger
    from db.property_db import (
        delete_badcase,
        get_badcase,
        list_badcase_actions,
    )

    ledger = RunEvidenceLedger(
        trace_id="contract-auto-badcase-trace",
        session_id="contract-auto-badcase-session",
        config_snapshot={"snapshot_id": "snap-contract"},
        evaluation_results=[
            {
                "case": "citation_allowlist",
                "passed": False,
            }
        ],
        contract_violations=[
            {
                "code": "invalid_evidence_id",
                "detail": "unknown evidence",
                "metadata": {"evidence_id": "ev_unknown"},
            }
        ],
    )
    trigger = runtime_badcase_trigger(ledger)
    assert trigger and trigger["source"] == "runtime_contract"
    created = capture_runtime_badcase(
        ledger=ledger,
        original_query="契约测试问题",
        ai_response="契约测试回答",
    )
    assert created and created["trace_id"] == ledger.trace_id
    duplicate = capture_runtime_badcase(
        ledger=ledger,
        original_query="契约测试问题",
        ai_response="契约测试回答",
    )
    assert duplicate["id"] == created["id"]
    stored = get_badcase(int(created["id"]))
    assert stored["source"] == "runtime_contract"
    actions = list_badcase_actions(int(created["id"]))
    assert any(item["action_type"] == "auto-capture" for item in actions)
    assert delete_badcase(int(created["id"]))


def test_static_conflict_removal():
    repo = Path(__file__).resolve().parents[1]
    thin_chat = (repo / "app" / "chat.py").read_text(encoding="utf-8")
    agent_factory = (repo / "app" / "runtime" / "agent_factory.py").read_text(
        encoding="utf-8"
    )
    coordinator = (repo / "app" / "runtime" / "coordinator.py").read_text(
        encoding="utf-8"
    )
    assert len(thin_chat.splitlines()) < 30
    assert "KnowledgeTools" not in agent_factory
    assert "WorkOrderTools" not in agent_factory
    assert "create_badcase" not in coordinator
    assert '"used_in_answer": False' not in coordinator


def test_fixed_acceptance_matrix():
    keys = [case["case_key"] for case in ACCEPTANCE_CASES]
    assert keys == [
        "HIT-01",
        "HANDOFF-01",
        "ACTION-01",
        "EXT-ASR-01",
        "EXT-MCP-01",
        "BADCASE-01",
        "COST-01",
        "FAIL-01",
    ]
    required = {
        "input",
        "business_write",
        "expected_route",
        "expected_skill",
        "expected_mcp",
        "expected_rag",
        "trace",
        "cost",
        "cleanup",
    }
    assert all(required.issubset(case) for case in ACCEPTANCE_CASES)


def main():
    tests = [
        test_release_and_snapshot_immutability,
        test_tool_policy_default_deny,
        test_citation_single_source_contract,
        test_cost_availability_contract,
        test_action_gateway_receipt_and_idempotency,
        test_action_success_claim_contract,
        test_mcp_preinvoke_initializes_session,
        test_mcp_business_result_unwrap_contract,
        test_citation_violation_recording_contract,
        test_v18_runtime_auto_badcase_contract,
        test_static_conflict_removal,
        test_fixed_acceptance_matrix,
    ]
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
    finally:
        TEMP_DIR.cleanup()
    print("V1.8 runtime convergence no-model contracts passed.")


if __name__ == "__main__":
    main()
