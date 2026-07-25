from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "index.html"


def _source() -> str:
    source = FRONTEND.read_text(encoding="utf-8")
    start = source.index("function toolCallName")
    end = source.index("window.openTraceDrawer = openTraceDrawer", start)
    return source[start:end]


def test_action_gateway_work_order_is_classified_as_controlled_write() -> None:
    source = _source()
    assert "toolCallName(call) === 'action_gateway'" in source
    assert "toolCallActionType(call) === 'work_order.create'" in source


def test_controlled_write_is_not_rendered_as_generic_tool() -> None:
    source = _source()
    assert "const ordinaryTools = allTools.filter(call => !isControlledWorkOrderCall(call))" in source
    assert "已调用工具：" in source
    assert "ordinaryTools.map" in source


def test_message_card_names_the_business_action() -> None:
    source = _source()
    assert "受控业务写入：创建维修工单" in source
    assert "renderControlledWriteCard" in source


def test_message_card_explains_internal_non_mcp_channel() -> None:
    source = _source()
    assert "执行通道：应用规则工作流 → ActionGateway → 内部工单服务" in source
    assert "本次非MCP写入 · 本次未调用模型" in source


def test_all_controlled_write_statuses_have_human_labels() -> None:
    source = _source()
    for label in (
        "工单草稿已更新",
        "等待业主确认",
        "信息不完整，暂不能提交",
        "正式工单已提交",
        "重复确认，未再次写入",
        "业主已取消",
        "提交失败",
        "受控操作处理中",
    ):
        assert label in source


def test_success_requires_committed_receipt_and_resource() -> None:
    source = _source()
    assert "status === 'committed' && Boolean(receiptId && resourceId)" in source
    assert "提交凭证不完整，不能确认创建成功" in source


def test_verified_commit_displays_receipt_resource_and_proposal() -> None:
    source = _source()
    assert "真实工单号：" in source
    assert "写入回执：" in source
    assert "查看提议编号" in source
    assert "已启用幂等保护" in source


def test_owner_confirmation_is_hitl_not_human_takeover() -> None:
    source = _source()
    assert "业主本人确认（HITL）· 无需人工接管" in source
    assert "这是业主确认（HITL），不等同于转人工接管。" in source


def test_rule_workflow_trace_button_explains_zero_model_usage() -> None:
    source = _source()
    assert "label: '规则工作流'" in source
    assert "本轮没有模型调用，因此没有模型Token与成本；业务Trace、确认与写入审计仍然有效。" in source


def test_generic_no_model_trace_label_remains_distinct() -> None:
    source = _source()
    assert "label: '无模型用量'" in source
    assert "usageSource === 'not_applicable'" in source


def test_trace_recognizes_new_and_historical_workflow_names() -> None:
    source = _source()
    assert "name === 'work_order_workflow'" in source
    assert "action_gateway: '受控写入网关'" in source
    assert "work_order_workflow: '维修工单确认工作流'" in source


def test_trace_evidence_follows_controlled_write_order() -> None:
    source = _source()
    positions = [
        source.index("1. Draft"),
        source.index("2. Proposal"),
        source.index("3. Owner Approval"),
        source.index("4. ActionGateway Execution"),
        source.index("5. Committed Receipt"),
    ]
    assert positions == sorted(positions)
    assert "受控业务写入证据" in source


def test_trace_displays_receipt_and_idempotency_evidence() -> None:
    source = _source()
    assert "resource_id:" in source
    assert "receipt_id:" in source
    assert "幂等保护：" in source
    assert "重复写入：" in source
    assert "error_summary" in source


def test_trace_explicitly_distinguishes_model_and_mcp() -> None:
    source = _source()
    assert "本轮为确定性规则工作流，没有调用模型，因此模型Token与成本不适用。" in source
    assert "本轮未调用MCP；正式写入由内部工单服务完成。" in source


def test_trace_does_not_render_full_proposal_payload() -> None:
    source = _source()
    assert "默认不展示完整 Payload" in source
    assert "safeTraceJson(controlledProposal.payload)" not in source
    assert "JSON.stringify(controlledProposal.payload" not in source


def test_trace_masks_mobile_phone_numbers() -> None:
    source = _source()
    assert "1[3-9]\\\\d{9}" in source
    assert "phone.slice(0, 3)}****${phone.slice(-4)" in source
    assert "safeTraceJson(call.arguments" in source
    assert "safeTraceJson(event.metadata)" in source


def test_live_sse_status_uses_business_semantics() -> None:
    source = FRONTEND.read_text(encoding="utf-8")
    assert "liveToolStatusText(data.tool_calls || [])" in source
    assert "受控业务写入：创建维修工单（${controlledWriteStatusLabel" in source


def test_existing_mcp_and_provider_usage_rendering_remain_available() -> None:
    source = _source()
    assert "MCP 调用：" in source
    assert "provider_reported_complete: 'Provider完整'" in source
    assert "provider_reported_total_only: 'Provider仅总量'" in source
    assert "estimated_tokenization: '本地估算'" in source


def test_trace_uses_existing_read_only_evidence_endpoints() -> None:
    source = _source()
    assert "apiGet(`/api/observability/traces/${traceId}`)" in source
    assert "apiGet(`/api/runtime/traces/${traceId}/evidence`)" in source
    assert "apiPost(" not in source
    assert "apiDelete(" not in source


if __name__ == "__main__":
    tests = [
        test_action_gateway_work_order_is_classified_as_controlled_write,
        test_controlled_write_is_not_rendered_as_generic_tool,
        test_message_card_names_the_business_action,
        test_message_card_explains_internal_non_mcp_channel,
        test_all_controlled_write_statuses_have_human_labels,
        test_success_requires_committed_receipt_and_resource,
        test_verified_commit_displays_receipt_resource_and_proposal,
        test_owner_confirmation_is_hitl_not_human_takeover,
        test_rule_workflow_trace_button_explains_zero_model_usage,
        test_generic_no_model_trace_label_remains_distinct,
        test_trace_recognizes_new_and_historical_workflow_names,
        test_trace_evidence_follows_controlled_write_order,
        test_trace_displays_receipt_and_idempotency_evidence,
        test_trace_explicitly_distinguishes_model_and_mcp,
        test_trace_does_not_render_full_proposal_payload,
        test_trace_masks_mobile_phone_numbers,
        test_live_sse_status_uses_business_semantics,
        test_existing_mcp_and_provider_usage_rendering_remain_available,
        test_trace_uses_existing_read_only_evidence_endpoints,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS: {len(tests)}/{len(tests)} S4-A.2 controlled-write UI contracts")
