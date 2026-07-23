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

from app.runtime.acceptance import ACCEPTANCE_CASES
from app.runtime.action_gateway import ActionGateway
from app.runtime.citation_renderer import build_evidence_set, render_citations
from app.runtime.contracts import RuntimePath, ToolEffect
from app.runtime.cost_ledger import build_cost_entry
from app.runtime.contracts import ToolInvocation
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


def test_new_vertical_agent_has_explicit_rag_scope():
    from app.agents import AgentCreate, create_agent

    created = asyncio.run(
        create_agent(
            AgentCreate(
                agent_id="contract-explicit-rag-agent",
                name="Contract Explicit RAG Agent",
                instructions="Contract-only vertical Agent.",
            )
        )
    )
    agent = created["agent"]
    assert agent["category"] == "vertical"
    assert agent["instructions"] == "Contract-only vertical Agent."
    assert agent["knowledge_scope_mode"] == "explicit"
    assert agent["knowledge_doc_ids"] == []

    release = publish_compiled_release(created_by="explicit-rag-contract")
    node = next(
        item
        for item in release["config"]["agents"]
        if item["agent_id"] == agent["agent_id"]
    )
    assert node["knowledge_doc_ids"] == []


def test_badcase_ai_context_keeps_contract_failures():
    from app.badcases import _focused_badcase_model_context

    focused = _focused_badcase_model_context(
        {
            "trace_id": "trace-contract-failure",
            "retrieval_evidence": [
                {
                    "evidence_id": f"ev-{index}",
                    "document_id": index,
                    "chunk_id": f"chunk-{index}",
                    "title": f"doc-{index}",
                    "content_snapshot": "large content " * 500,
                }
                for index in range(12)
            ],
            "evaluation_results": [
                {
                    "case": "citation_allowlist",
                    "passed": False,
                    "violations": [{"code": "unstructured_reference_marker"}],
                }
            ],
            "contract_violations": [
                {
                    "code": "unstructured_reference_marker",
                    "detail": "Model marker was outside EvidenceSet.",
                }
            ],
        }
    )
    assert focused["trace_id"] == "trace-contract-failure"
    assert focused["contract_violations"][0]["code"] == (
        "unstructured_reference_marker"
    )
    assert focused["evaluation_results"][0]["passed"] is False
    assert focused["retrieval_evidence_count"] == 12
    assert len(focused["retrieval_evidence_summary"]) == 8
    assert "content_snapshot" not in focused["retrieval_evidence_summary"][0]


def test_badcase_retest_uses_public_chat_adapter():
    from app.chat import stream_chat_response

    assert callable(stream_chat_response)


def test_tool_policy_default_deny():
    server = {"id": 1, "name": "demo", "enabled": True}
    read = compile_tool_policy(
        server,
        {
            "name": "opaque_read_name",
            "tool_metadata": {
                "effect": "read",
                "effect_source": "operator_declared",
            },
        },
    )
    write = compile_tool_policy(
        server,
        {
            "name": "opaque_write_name",
            "tool_metadata": {
                "effect": "create",
                "effect_source": "operator_declared",
            },
        },
    )
    delete = compile_tool_policy(
        server,
        {
            "name": "opaque_delete_name",
            "tool_metadata": {
                "effect": "delete",
                "effect_source": "operator_declared",
            },
        },
    )
    unknown = compile_tool_policy(server, {"name": "do_magic", "tool_metadata": {}})
    misleading_name = compile_tool_policy(
        server,
        {"name": "create_record", "tool_metadata": {}},
    )
    assert read.effect == ToolEffect.READ and read.enabled
    assert write.effect == ToolEffect.CREATE and write.requires_confirmation
    assert delete.effect == ToolEffect.DELETE and not delete.enabled
    assert unknown.effect == ToolEffect.UNKNOWN and not unknown.enabled
    assert misleading_name.effect == ToolEffect.UNKNOWN
    assert not misleading_name.enabled


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
    rendered_suffix, suffix_citations, suffix_violations = render_citations(
        (
            "发现儿童受伤应立即停止设施使用并联系工作人员 "
            f"[[evidence:{evidence_id.removeprefix('ev_')}]]。"
        ),
        evidence,
    )
    assert rendered_suffix.endswith("【引用1】。")
    assert suffix_citations[0].evidence_id == evidence_id
    assert not suffix_violations
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
    source_line, source_line_citations, source_line_violations = render_citations(
        (
            f"> 引用来源：儿童游乐区安全制度 [[evidence:{evidence_id}]]\n\n"
            "1. 发现儿童受伤应立即停止设施使用\n"
            "2. 立即联系工作人员处理"
        ),
        evidence,
    )
    assert "【引用1】" in source_line
    assert len(source_line_citations) == 1
    assert not source_line_violations
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
    assert total_only.price_snapshot is not None
    estimated = build_cost_entry(
        "agent", "demo", "demo-model", "demo-model", "v1", None, price, 120
    )
    assert estimated.usage_source.value == "local_estimate"
    assert estimated.amount is None
    assert estimated.price_snapshot is not None
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


def test_agno_completed_metrics_are_extractable():
    from app.runtime.coordinator import _metrics_dict

    class FakeRunMetrics:
        input_tokens = 120
        cached_tokens = 20
        output_tokens = 40
        reasoning_tokens = 5
        total_tokens = 160

    class FakeRunCompleted:
        event = "RunCompleted"
        content = "full answer"
        metrics = FakeRunMetrics()

    assert _metrics_dict(FakeRunCompleted()) == {
        "input_tokens": 120,
        "output_tokens": 40,
        "reasoning_tokens": 5,
        "cached_tokens": 20,
        "total_tokens": 160,
    }


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
    stored = get_work_order(receipt_1.resource_id)
    assert stored is not None
    assert stored["status"] == "待派单"


def test_bare_work_order_command_collects_real_issue_description():
    from app.work_order_workflow import advance_work_order_workflow

    session_id = "contract-bare-work-order"
    first = advance_work_order_workflow(session_id, "帮我创建工单")
    assert first is not None
    assert first["action"] == "draft_updated"
    assert first["draft"]["issue_desc"] == ""
    assert "维修问题描述" in first["missing_fields"]

    second = advance_work_order_workflow(
        session_id,
        "卫生间天花板持续滴水",
    )
    assert second is not None
    assert second["draft"]["issue_desc"] == "卫生间天花板持续滴水"
    assert second["draft"]["issue_type"] == "水电"


def test_natural_work_order_cancellation_rejects_without_write():
    from app.work_order_workflow import (
        advance_work_order_workflow,
        is_cancel_request,
    )

    assert is_cancel_request("取消，不要提交这个工单。")
    assert is_cancel_request("别创建了")
    assert not is_cancel_request("不要取消这个工单")

    session_id = "contract-natural-work-order-cancel"
    drafted = advance_work_order_workflow(
        session_id,
        (
            "请创建维修工单：房号 3-2-1201，卫生间天花板持续滴水，"
            "紧急，联系人测试业主，电话 13800138000，明天上午上门。"
        ),
    )
    assert drafted and drafted["action"] == "awaiting_confirmation"
    rejected = advance_work_order_workflow(
        session_id,
        "取消，不要提交这个工单。",
    )
    assert rejected and rejected["action"] == "rejected"
    assert "未创建正式工单" in rejected["reply"]


def test_repeated_work_order_confirmation_replays_same_receipt():
    from app.runtime.coordinator import RuntimeCoordinator
    from app.work_order_workflow import advance_work_order_workflow

    session_id = "contract-work-order-replay"
    draft = advance_work_order_workflow(
        session_id,
        (
            "帮我创建维修工单：房号 3-2-1201，厨房水槽持续漏水，"
            "紧急，联系人测试业主，电话 13800138000，尽快上门。"
        ),
    )
    assert draft and draft["action"] == "awaiting_confirmation"
    committed = advance_work_order_workflow(session_id, "确认提交")
    replayed = advance_work_order_workflow(session_id, "确认提交")
    assert committed and committed["action"] == "committed"
    assert replayed and replayed["action"] == "idempotent_replay"
    assert committed["work_order_id"] == replayed["work_order_id"]
    assert RuntimeCoordinator._is_work_order_action_context(
        session_id,
        "确认提交",
    )


def test_pending_dynamic_mcp_action_keeps_controlled_path():
    from app.runtime.coordinator import RuntimeCoordinator

    session_id = "contract-dynamic-mcp-pending"
    proposal = ActionGateway().propose(
        session_id=session_id,
        action_type="mcp.offdomain.create_record",
        payload={
            "agent_id": "offdomain",
            "server_name": "offdomain-server",
            "tool_name": "create_record",
            "arguments": {"value": "contract"},
        },
    )
    assert proposal.status == "pending_confirmation"
    assert RuntimeCoordinator._select_path(
        session_id,
        "拒绝",
        {},
    ) == RuntimePath.CONTROLLED_ACTION
    assert not RuntimeCoordinator._is_work_order_action_context(
        session_id,
        "拒绝",
    )


def test_committed_dynamic_mcp_confirmation_replays_receipt():
    from types import SimpleNamespace

    from app.runtime.coordinator import RuntimeCoordinator
    from db.property_db import save_action_receipt

    session_id = "contract-dynamic-mcp-replay"
    proposal = ActionGateway().propose(
        session_id=session_id,
        action_type="mcp.offdomain.create_record",
        payload={
            "agent_id": "offdomain",
            "agent_name": "Off-domain Agent",
            "server_name": "offdomain-server",
            "tool_name": "create_record",
            "arguments": {"value": "contract"},
        },
    )
    saved = save_action_receipt(
        receipt_id="receipt-contract-dynamic-replay",
        proposal_id=proposal.proposal_id,
        idempotency_key=proposal.idempotency_key,
        status="committed",
        resource_type="mcp:offdomain-server:create_record",
        resource_id="RESOURCE-CONTRACT-1",
        result={
            "resource_type": "mcp:offdomain-server:create_record",
            "resource_id": "RESOURCE-CONTRACT-1",
            "server_name": "offdomain-server",
            "tool_name": "create_record",
            "effect": "create",
            "business_status": "success",
        },
    )
    assert saved["status"] == "committed"
    assert RuntimeCoordinator._select_path(
        session_id,
        "确认提交",
        {},
    ) == RuntimePath.CONTROLLED_ACTION
    replay = asyncio.run(
        RuntimeCoordinator()._advance_dynamic_mcp_action(
            message="确认提交",
            session_id=session_id,
            trace_id="trace-contract-dynamic-replay",
            snapshot=SimpleNamespace(config={}, release_id="release-contract"),
        )
    )
    assert replay["action"] == "idempotent_replay"
    assert replay["receipt"]["receipt_id"] == saved["receipt_id"]
    assert replay["receipt"]["resource_id"] == "RESOURCE-CONTRACT-1"


def test_database_receipt_row_projects_to_public_contract():
    from app.runtime.coordinator import _action_receipt_from_payload

    receipt = _action_receipt_from_payload(
        {
            "receipt_id": "receipt-contract-projection",
            "proposal_id": "proposal-contract-projection",
            "idempotency_key": "idem-contract-projection",
            "status": "committed",
            "resource_type": "mcp:test:create",
            "resource_id": "RESOURCE-PROJECTION-1",
            "result": {"resource_id": "RESOURCE-PROJECTION-1"},
            "committed_at": "2026-07-19 16:00:00",
            "created_at": "internal-database-field",
        }
    )
    assert receipt.receipt_id == "receipt-contract-projection"
    assert receipt.resource_id == "RESOURCE-PROJECTION-1"
    assert not hasattr(receipt, "created_at")


def test_idempotent_replay_is_not_a_new_mcp_invocation():
    from app.runtime.coordinator import _records_new_mcp_invocation

    assert _records_new_mcp_invocation(
        "mcp.offdomain.create_record",
        "committed",
    )
    assert not _records_new_mcp_invocation(
        "mcp.offdomain.create_record",
        "idempotent_replay",
    )
    assert not _records_new_mcp_invocation(
        "work_order.create",
        "committed",
    )


def test_composite_work_order_query_plans_exact_read_tools():
    from app.runtime.tool_planner import plan_tools

    def policy(tool_name):
        return {
            "server_id": 7,
            "server_name": "workorder-server",
            "tool_name": tool_name,
            "effect": "read",
            "risk_level": "L0",
            "allowed_paths": ["consultation"],
            "requires_confirmation": False,
            "enabled": True,
            "policy_reason": "contract fixture",
        }

    tools = [
        {
            "name": "get_my_recent_work_orders",
            "input_schema": {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            },
            "policy": policy("get_my_recent_work_orders"),
        },
        {
            "name": "count_my_open_work_orders",
            "input_schema": {"type": "object", "properties": {}},
            "policy": policy("count_my_open_work_orders"),
        },
        {
            "name": "count_work_orders",
            "input_schema": {
                "type": "object",
                "properties": {"status": {"type": "string"}},
            },
            "policy": policy("count_work_orders"),
        },
    ]
    config = {
        "agents": [
            {
                "agent_id": "maintenance",
                "enabled": True,
                "mcp_server_names": ["workorder-server"],
            }
        ],
        "mcp_servers": [
            {
                "id": 7,
                "name": "workorder-server",
                "enabled": True,
                "tools": tools,
            }
        ],
    }
    message = "查询我房号最近的维修工单，以及系统当前待处理工单数量"
    plans = plan_tools(
        config,
        "maintenance",
        message,
        RuntimePath.CONSULTATION,
        effects=[ToolEffect.READ],
        execution_modes=["auto_preinvoke"],
    )
    by_name = {item.tool_name: item for item in plans}
    assert set(by_name) == {
        "get_my_recent_work_orders",
        "count_work_orders",
    }
    assert by_name["get_my_recent_work_orders"].arguments == {"limit": 5}
    assert by_name["count_work_orders"].arguments == {"status": "待派单"}


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


def _off_domain_runtime_config():
    def policy(tool_name, effect, paths, requires_confirmation=False):
        return {
            "server_id": 91,
            "server_name": "astronomy-server",
            "tool_name": tool_name,
            "effect": effect,
            "risk_level": "L2" if requires_confirmation else "L1",
            "allowed_paths": paths,
            "requires_confirmation": requires_confirmation,
            "enabled": True,
            "policy_reason": "off-domain contract fixture",
        }

    return {
        "agents": [
            {
                "agent_id": "exoplanet_agent",
                "name": "系外行星观测 Agent",
                "description": "处理系外行星凌日、观测窗口和观测申请",
                "category": "vertical",
                "enabled": True,
                "skill_ids": [991],
                "mcp_server_names": ["astronomy-server"],
                "knowledge_doc_ids": [771],
            }
        ],
        "skills": [
            {
                "skill_id": 991,
                "name": "凌日观测计划",
                "description": "规划系外行星凌日观测",
                "enabled": True,
                "metadata": {"positive_triggers": ["系外行星", "凌日观测"]},
            }
        ],
        "knowledge": [
            {
                "knowledge_doc_id": 771,
                "title": "GVX-42 观测手册",
                "category": "天文演示",
            }
        ],
        "mcp_servers": [
            {
                "server_id": 91,
                "name": "astronomy-server",
                "description": "查询和登记系外行星观测任务",
                "enabled": True,
                "is_builtin": False,
                "command": "fake-astronomy",
                "args": [],
                "tools": [
                    {
                        "name": "lookup_window",
                        "description": "查询目标观测窗口",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "time_range": {"type": "string"},
                            },
                            "required": ["target", "time_range"],
                        },
                        "tool_metadata": {
                            "effect": "read",
                            "effect_source": "operator_declared",
                            "risk_level": "L1",
                            "result_contract": {
                                "success_statuses": ["success"],
                                "non_success_statuses": ["not_found", "invalid_input"],
                            },
                            "natural_language_intents": ["查询系外行星观测窗口"],
                            "trigger_keywords": ["观测窗口"],
                            "trigger_mode": "any",
                            "execution_mode": "auto_preinvoke",
                            "argument_bindings": {
                                "target": {
                                    "source": "regex",
                                    "pattern": r"\b(GVX-\d+)\b",
                                    "group": 1,
                                },
                                "time_range": {
                                    "source": "keyword_map",
                                    "mapping": {"明晚": "tomorrow_evening"},
                                },
                            },
                        },
                        "policy": policy(
                            "lookup_window",
                            "read",
                            ["consultation", "extension_acceptance"],
                        ),
                    },
                    {
                        "name": "register_request",
                        "description": "登记观测申请",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "target": {"type": "string"},
                                "time_range": {"type": "string"},
                            },
                            "required": ["target", "time_range"],
                        },
                        "tool_metadata": {
                            "effect": "create",
                            "effect_source": "operator_declared",
                            "risk_level": "L2",
                            "result_contract": {
                                "success_statuses": ["success"],
                                "non_success_statuses": ["invalid_input", "upstream_error"],
                            },
                            "natural_language_intents": ["登记新的观测申请"],
                            "trigger_keywords": ["登记观测申请"],
                            "trigger_mode": "any",
                            "execution_mode": "proposal",
                            "argument_bindings": {
                                "target": {
                                    "source": "regex",
                                    "pattern": r"\b(GVX-\d+)\b",
                                    "group": 1,
                                },
                                "time_range": {
                                    "source": "keyword_map",
                                    "mapping": {"明晚": "tomorrow_evening"},
                                },
                            },
                        },
                        "policy": policy(
                            "register_request",
                            "create",
                            ["controlled_action", "extension_acceptance"],
                            requires_confirmation=True,
                        ),
                    },
                ],
            }
        ],
    }


def test_configuration_driven_off_domain_tool_plan():
    from app.runtime.tool_planner import plan_tools, unique_write_plan

    config = _off_domain_runtime_config()
    read_query = "请查询 GVX-42 明晚的观测窗口。"
    read_plans = plan_tools(
        config,
        "exoplanet_agent",
        read_query,
        RuntimePath.CONSULTATION,
        effects=[ToolEffect.READ],
        execution_modes=["auto_preinvoke"],
    )
    assert len(read_plans) == 1
    assert read_plans[0].tool_name == "lookup_window"
    assert read_plans[0].arguments == {
        "target": "GVX-42",
        "time_range": "tomorrow_evening",
    }
    assert not read_plans[0].missing_required
    # The owner never has to know or type the internal tool name.
    assert "lookup_window" not in read_query

    write_query = "请为 GVX-42 登记观测申请，时间是明晚。"
    write_plan = unique_write_plan(config, write_query)
    assert write_plan is not None
    assert write_plan.tool_name == "register_request"
    assert write_plan.execution_mode == "proposal"
    assert write_plan.arguments["target"] == "GVX-42"
    assert "register_request" not in write_query

    # Trigger keywords improve determinism but are not a hidden hard
    # dependency: an operator-declared intent/description can still select one
    # unambiguous write tool.
    write_tool = config["mcp_servers"][0]["tools"][1]
    write_tool["tool_metadata"]["trigger_keywords"] = []
    description_plan = unique_write_plan(
        config,
        "请为 GVX-42 登记观测申请，时间是明晚。",
    )
    assert description_plan is not None
    assert description_plan.tool_name == "register_request"


def test_off_domain_router_uses_published_capabilities():
    from agents.router import _capability_fallback
    from app.runtime.agent_factory import vertical_agent_cards

    cards = vertical_agent_cards(_off_domain_runtime_config())
    selected, reason, scores = _capability_fallback(
        "请按《GVX-42 观测手册》查询系外行星观测窗口。",
        cards,
    )
    assert selected == "exoplanet_agent"
    assert any(
        item["agent_id"] == "exoplanet_agent" and item["score"] > 0
        for item in scores
    )
    assert any(term in reason for term in ("GVX", "系外行星", "观测"))


def test_scoped_rag_is_applied_before_top_k():
    import rag_retrieval

    original_list_docs = rag_retrieval.db.list_knowledge_docs
    original_get_doc = rag_retrieval.db.get_knowledge_doc
    original_embed = rag_retrieval.rag_embeddings.embed_text
    original_threshold = rag_retrieval.rag_indexer._effective_threshold
    original_search_chunks = rag_retrieval.rag_store.search_chunks
    seen = {}

    docs = [
        {"id": 771, "title": "GVX-42", "content": "观测窗口", "is_indexed": 1},
        {"id": 999, "title": "物业维修", "content": "漏水天气", "is_indexed": 1},
    ]

    def fake_search_chunks(
        query_embedding,
        top_k,
        threshold,
        allowed_document_ids=None,
    ):
        seen["allowed_document_ids"] = set(allowed_document_ids or [])
        return [
            {
                "id": 1,
                "doc_id": 771,
                "chunk_index": 0,
                "content": "GVX-42 明晚观测窗口",
                "score": 0.99,
            }
        ]

    try:
        rag_retrieval.db.list_knowledge_docs = lambda: docs
        rag_retrieval.db.get_knowledge_doc = lambda doc_id: next(
            item for item in docs if item["id"] == doc_id
        )
        rag_retrieval.rag_embeddings.embed_text = lambda query: [0.0] * 512
        rag_retrieval.rag_indexer._effective_threshold = lambda value: 0.0
        rag_retrieval.rag_store.search_chunks = fake_search_chunks
        keyword_index = rag_retrieval._build_keyword_index({771})
        semantic = rag_retrieval._semantic_search(
            "GVX-42 观测窗口",
            top_k=1,
            threshold=0.0,
            allowed_document_ids={771},
        )
    finally:
        rag_retrieval.db.list_knowledge_docs = original_list_docs
        rag_retrieval.db.get_knowledge_doc = original_get_doc
        rag_retrieval.rag_embeddings.embed_text = original_embed
        rag_retrieval.rag_indexer._effective_threshold = original_threshold
        rag_retrieval.rag_store.search_chunks = original_search_chunks

    assert set(keyword_index) == {771}
    assert seen["allowed_document_ids"] == {771}
    assert {item["doc_id"] for item in semantic} == {771}


def test_dynamic_mcp_runtime_has_no_domain_branching():
    source = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "runtime"
        / "mcp_executor.py"
    ).read_text(encoding="utf-8")
    assert "def _relevant(" not in source
    assert "def _plan(" not in source
    assert 'if server_name == "weather-server"' not in source
    assert "plan_tools(" in source


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
    assert len(thin_chat.splitlines()) < 60
    assert "KnowledgeTools" not in agent_factory
    assert "WorkOrderTools" not in agent_factory
    assert "create_badcase" not in coordinator
    assert '"used_in_answer": False' not in coordinator


def test_runtime_summary_uses_actual_evidence():
    from app.runtime.coordinator import _append_runtime_evidence_summary

    rendered = _append_runtime_evidence_summary(
        "业务回答。",
        "最后用一行汇总实际调用的工具和知识库。",
        [{"tool_name": "get_skill_instructions", "arguments": {"skill_id": 8}}],
        [
            ToolInvocation(
                invocation_id="inv-test",
                server_name="weather-server",
                tool_name="get_current_weather",
                effect=ToolEffect.READ,
                invocation_status="success",
                transport_status="success",
                business_status="success",
            )
        ],
        [
            type("Citation", (), {"title": "常见维修问题 FAQ"})(),
            type("Citation", (), {"title": "常见维修问题 FAQ"})(),
            type("Citation", (), {"title": "物业维修服务承诺"})(),
        ],
    )
    footer = rendered.splitlines()[-1]
    assert "get_skill_instructions" in footer
    assert "weather-server/get_current_weather" in footer
    assert footer.count("常见维修问题 FAQ") == 1
    assert "物业维修服务承诺" in footer


def test_only_published_model_native_tools_enter_agno_loop():
    source = (
        REPO_ROOT / "app" / "runtime" / "mcp_executor.py"
    ).read_text(encoding="utf-8")
    assert '.get("execution_mode")' in source
    assert '== "model_native"' in source


def test_cost_strategy_claims_are_evidence_bound():
    repo = Path(__file__).resolve().parents[1]
    frontend = (repo / "frontend" / "index.html").read_text(encoding="utf-8")
    observability = (repo / "app" / "observability.py").read_text(
        encoding="utf-8"
    )
    assert "/api/retrieval/search" not in frontend
    assert "await apiPut('/api/knowledge/retrieval-settings'" not in frontend
    assert "降低 80% 以上单轮成本" not in frontend
    assert "节省约 800 Token" not in frontend
    assert "约 2500 Token" not in frontend
    assert frontend.count(
        "apiPost('/api/runtime/cost-preview/retrieval'"
    ) == 2
    assert "检索预估不等于模型账单" in observability
    assert "质量下降或 Token 未降时拒绝候选" in observability
    runtime_api = (repo / "app" / "runtime" / "api.py").read_text(
        encoding="utf-8"
    )
    assert '"configuration_mutated": False' in runtime_api
    assert '"model_called": False' in runtime_api
    assert '"provider_usage": None' in runtime_api


def test_platform_navigation_serializes_async_page_renders():
    frontend = (REPO_ROOT / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    assert "let navigationQueue = Promise.resolve();" in frontend
    assert "function queueNavigation(" in frontend
    assert "navigationQueue = navigationQueue" in frontend
    assert "await renderContent();" in frontend
    expected_async_dispatches = [
        "await renderMyOrdersPage(main)",
        "await renderStaffWorkOrdersPage(main)",
        "await renderHandoffPage(main)",
        "await renderRuntimePage(main)",
        "await renderAgentsPage(main)",
        "await renderModelsPage(main)",
        "await renderSkillsPage(main)",
        "await renderMcpPage(main)",
        "await renderKnowledgePage(main)",
        "await renderBadcasesPage(main)",
        "await renderEvaluationsPage(main)",
        "await renderCostGovernancePage(main)",
        "await renderCostStrategyPage(main)",
    ]
    for dispatch in expected_async_dispatches:
        assert dispatch in frontend
    assert "{ id: 'work-orders', label: '工单管理'" in frontend
    assert "data-order-view=" in frontend
    assert "const STAFF_ORDER_VIEWS =" in frontend
    assert "{ id: 'pending-orders'" not in frontend
    assert "{ id: 'processing-orders'" not in frontend
    assert "{ id: 'history-orders'" not in frontend
    for capability_menu in ("agents", "skills", "mcp", "knowledge"):
        assert f"{{ id: '{capability_menu}'" in frontend
    assert "{ id: 'capabilities'" not in frontend


def test_multimodal_capability_is_retired():
    """Vision must not remain as a hidden API, chat field or UI path."""
    assert not (REPO_ROOT / "app" / "multimodal.py").exists()
    sources = {
        "main": (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8"),
        "chat": (REPO_ROOT / "app" / "chat.py").read_text(encoding="utf-8"),
        "legacy": (
            REPO_ROOT / "app" / "runtime" / "legacy_chat.py"
        ).read_text(encoding="utf-8"),
        "coordinator": (
            REPO_ROOT / "app" / "runtime" / "coordinator.py"
        ).read_text(encoding="utf-8"),
        "frontend": (
            REPO_ROOT / "frontend" / "index.html"
        ).read_text(encoding="utf-8"),
        "compose": (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8"),
    }
    forbidden = {
        "main": ["app.multimodal", "multimodal_router"],
        "chat": ["image_analysis_ids"],
        "legacy": ["image_analysis_ids", "get_analysis_context"],
        "coordinator": ["image_analysis_ids"],
        "frontend": [
            "/api/multimodal",
            "chat-image-picker",
            "chat-image-input",
            "image_analysis_ids",
            "Kimi",
        ],
        "compose": ["KIMI_API_KEY", "KIMI_BASE_URL", "KIMI_MODEL_ID"],
    }
    for source_name, markers in forbidden.items():
        for marker in markers:
            assert marker not in sources[source_name], (
                source_name,
                marker,
            )


def test_fixed_acceptance_matrix():
    keys = [case["case_key"] for case in ACCEPTANCE_CASES]
    assert keys == [
        "HIT-01",
        "HANDOFF-01",
        "ACTION-01",
        "EXT-OFFDOMAIN-01",
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



def test_owner_chat_runtime_is_single_track():
    """Only RuntimeCoordinator may execute owner-chat business capabilities."""
    legacy = (REPO_ROOT / "app" / "runtime" / "legacy_chat.py").read_text(
        encoding="utf-8"
    )
    chat = (REPO_ROOT / "app" / "chat.py").read_text(encoding="utf-8")
    settings = (REPO_ROOT / "app" / "settings.py").read_text(encoding="utf-8")
    compose = (REPO_ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "RuntimeCoordinator().stream" in legacy
    for marker in (
        "RUNTIME_ENGINE",
        "_stream_v17_response",
        "create_maintenance_agent",
        "create_billing_agent",
        "create_complaint_agent",
        "create_customer_service_agent",
        "ObservableMCPTools",
    ):
        assert marker not in legacy
    assert "RUNTIME_ENGINE" not in settings
    assert "RUNTIME_ENGINE" not in compose
    assert "_policy_mcp_args" not in chat
    assert "_unique_rag_results" not in chat

    for retired in (
        "maintenance.py",
        "billing.py",
        "complaint.py",
        "customer_service.py",
    ):
        assert not (REPO_ROOT / "agents" / retired).exists()


def main():
    tests = [
        test_release_and_snapshot_immutability,
        test_new_vertical_agent_has_explicit_rag_scope,
        test_badcase_ai_context_keeps_contract_failures,
        test_badcase_retest_uses_public_chat_adapter,
        test_tool_policy_default_deny,
        test_citation_single_source_contract,
        test_cost_availability_contract,
        test_agno_completed_metrics_are_extractable,
        test_action_gateway_receipt_and_idempotency,
        test_bare_work_order_command_collects_real_issue_description,
        test_natural_work_order_cancellation_rejects_without_write,
        test_repeated_work_order_confirmation_replays_same_receipt,
        test_pending_dynamic_mcp_action_keeps_controlled_path,
        test_committed_dynamic_mcp_confirmation_replays_receipt,
        test_database_receipt_row_projects_to_public_contract,
        test_idempotent_replay_is_not_a_new_mcp_invocation,
        test_composite_work_order_query_plans_exact_read_tools,
        test_action_success_claim_contract,
        test_mcp_preinvoke_initializes_session,
        test_mcp_business_result_unwrap_contract,
        test_configuration_driven_off_domain_tool_plan,
        test_off_domain_router_uses_published_capabilities,
        test_scoped_rag_is_applied_before_top_k,
        test_dynamic_mcp_runtime_has_no_domain_branching,
        test_citation_violation_recording_contract,
        test_v18_runtime_auto_badcase_contract,
        test_static_conflict_removal,
        test_runtime_summary_uses_actual_evidence,
        test_only_published_model_native_tools_enter_agno_loop,
        test_cost_strategy_claims_are_evidence_bound,
        test_platform_navigation_serializes_async_page_renders,
        test_multimodal_capability_is_retired,
        test_owner_chat_runtime_is_single_track,
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
