"""No-model V1.8 runtime convergence contract checks."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


TEMP_DIR = tempfile.TemporaryDirectory(prefix="yiai-v180-")
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


def test_fixed_acceptance_matrix():
    keys = [case["case_key"] for case in ACCEPTANCE_CASES]
    assert keys == [
        "E2E-00",
        "READ-01",
        "MCP-R-01",
        "ACTION-01",
        "EXT-ASR-01",
        "EXT-MCP-01",
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
