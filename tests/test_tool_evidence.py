"""Deterministic contracts for immutable read-Tool result evidence."""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

from app.runtime.citation_renderer import (
    build_run_evidence_bundle,
    render_bundle_citations,
    render_citations,
)
from app.runtime.contracts import (
    ActionReceipt,
    EvidenceItem,
    EvidenceSet,
    ToolEffect,
    ToolInvocation,
    content_hash,
)
from app.runtime.mcp_executor import (
    GovernedMCPTools,
    _evaluate_read_tool_result,
    _tool_evidence_context,
)


def _evaluate(result, *, invocation_id="tool_test", result_contract=None):
    return _evaluate_read_tool_result(
        result,
        result_contract or {"success_statuses": ["success"]},
        invocation_id=invocation_id,
        server_name="fixture-server",
        tool_name="fixture-tool",
    )


def _facts_by_path(evidence):
    assert evidence is not None
    return {fact.json_path: fact for fact in evidence.facts}


def test_nested_structured_result_preserves_exact_units():
    status, summary, evidence = _evaluate(
        {
            "status": "success",
            "data": {
                "temperature_c": 29,
                "humidity_pct": 70,
                "wind": "东风 3 级",
            },
        }
    )
    facts = _facts_by_path(evidence)

    assert status == "success"
    assert len(summary) <= 500
    assert facts["$.data.temperature_c"].display_value == "29℃"
    assert facts["$.data.humidity_pct"].normalized_value == "70"
    assert facts["$.data.humidity_pct"].unit == "%"
    assert facts["$.data.humidity_pct"].display_value == "70%"
    assert facts["$.data.wind"].value == "东风 3 级"


def test_fact_after_summary_cutoff_remains_in_evidence():
    status, summary, evidence = _evaluate(
        {
            "status": "success",
            "a_padding": "x" * 700,
            "data": {"humidity_pct": 70},
        }
    )
    facts = _facts_by_path(evidence)

    assert status == "success"
    assert len(summary) == 500
    assert "humidity_pct" not in summary
    assert facts["$.data.humidity_pct"].display_value == "70%"


def test_request_arguments_cannot_enter_tool_evidence():
    signature = inspect.signature(_evaluate_read_tool_result)
    assert "arguments" not in signature.parameters

    _, _, evidence = _evaluate(
        {"status": "success", "data": {"result": 65}}
    )
    invocation = ToolInvocation(
        invocation_id="tool_test",
        server_name="fixture-server",
        tool_name="fixture-tool",
        effect=ToolEffect.READ,
        arguments={"requested_value": 70},
        discovery_status="success",
        transport_status="success",
        invocation_status="success",
        business_status="success",
        result_summary="{'requested_value': 70, 'result': 65}",
        tool_evidence=evidence,
    )

    values = {fact.normalized_value for fact in invocation.tool_evidence.facts}
    assert "65" in values
    assert "70" not in values


def test_non_success_business_results_never_create_evidence():
    for business_status in ("failed", "not_found", "timeout", "invalid_input"):
        status, _, evidence = _evaluate(
            {
                "status": business_status,
                "data": {"humidity_pct": 70},
            }
        )
        assert status == business_status
        assert evidence is None


def test_sensitive_scalars_are_excluded_but_payload_hash_covers_payload():
    _, _, first = _evaluate(
        {
            "status": "success",
            "data": {
                "work_order_id": "WO-100",
                "api_key": "must-not-become-a-fact",
                "nested": {"refresh_token": "also-secret"},
            },
        },
        invocation_id="tool_same",
    )
    _, _, second = _evaluate(
        {
            "status": "success",
            "data": {
                "work_order_id": "WO-100",
                "api_key": "different-secret",
                "nested": {"refresh_token": "also-secret"},
            },
        },
        invocation_id="tool_same",
    )
    paths = set(_facts_by_path(first))

    assert "$.data.work_order_id" in paths
    assert "$.data.api_key" not in paths
    assert "$.data.nested.refresh_token" not in paths
    assert first.payload_hash != second.payload_hash


def test_calculator_structured_result_keeps_result_84_compatibility():
    _, _, evidence = _evaluate(
        {
            "content": "84.0",
            "metadata": {"structured_content": {"result": 84.0}},
        }
    )
    fact = _facts_by_path(evidence)["$.result"]

    assert fact.normalized_value == "84"
    assert fact.semantic_type == "calculated_result"
    assert fact.unit is None


def test_preinvoke_context_is_rendered_from_facts_not_summary():
    _, _, evidence = _evaluate(
        {"status": "success", "data": {"humidity_pct": 70}}
    )
    invocation = ToolInvocation(
        invocation_id="tool_test",
        server_name="fixture-server",
        tool_name="fixture-tool",
        effect=ToolEffect.READ,
        discovery_status="success",
        transport_status="success",
        invocation_status="success",
        business_status="success",
        result_summary="summary-lie-71-percent",
        tool_evidence=evidence,
    )
    context = _tool_evidence_context(invocation)

    assert "$.data.humidity_pct = 70%" in context
    assert "summary-lie-71-percent" not in context


def _successful_invocation(result, *, arguments=None, result_contract=None):
    status, summary, evidence = _evaluate(
        result,
        result_contract=result_contract,
    )
    return ToolInvocation(
        invocation_id="tool_test",
        server_name="fixture-server",
        tool_name="fixture-tool",
        effect=ToolEffect.READ,
        arguments=arguments or {},
        discovery_status="success",
        transport_status="success",
        invocation_status="success",
        business_status=status,
        result_summary=summary,
        tool_evidence=evidence,
    )


def test_renderer_requires_exact_tool_value_and_unit():
    invocation = _successful_invocation(
        {"status": "success", "data": {"humidity_pct": 70}}
    )
    _, _, accepted = render_citations(
        "当前湿度70%。",
        EvidenceSet(query="查询当前天气"),
        tool_invocations=[invocation],
    )
    assert not accepted

    for unsupported in ("70分钟", "70元", "71%"):
        _, _, violations = render_citations(
            f"结果是{unsupported}。",
            EvidenceSet(query="查询当前天气"),
            tool_invocations=[invocation],
        )
        assert any(
            item.get("code") == "ungrounded_critical_value"
            for item in violations
        ), (unsupported, violations)


def test_arguments_and_summary_cannot_override_structured_result():
    invocation = _successful_invocation(
        {"status": "success", "data": {"humidity_pct": 65}},
        arguments={"humidity_pct": 70},
    ).model_copy(update={"result_summary": "伪摘要湿度71%"})

    _, _, accepted = render_citations(
        "当前湿度65%。",
        EvidenceSet(query="查询当前天气"),
        tool_invocations=[invocation],
    )
    assert not accepted
    for unsupported in ("70%", "71%"):
        _, _, violations = render_citations(
            f"当前湿度{unsupported}。",
            EvidenceSet(query="查询当前天气"),
            tool_invocations=[invocation],
        )
        assert any(
            item.get("code") == "ungrounded_critical_value"
            for item in violations
        ), (unsupported, violations)


def test_user_query_value_cannot_override_structured_result():
    invocation = _successful_invocation(
        {"status": "success", "data": {"humidity_pct": 65}},
        arguments={"humidity_pct": 70},
    )
    _, _, violations = render_citations(
        "当前湿度70%。",
        EvidenceSet(query="阈值是70%，请查询真实湿度"),
        tool_invocations=[invocation],
    )
    assert any(
        item.get("code") == "ungrounded_critical_value"
        for item in violations
    ), violations


def test_categorical_tool_facts_are_exact_not_broadly_linked():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {
                "city": "上海",
                "condition": "多云",
                "temperature_c": 29,
                "wind": "东风3级",
            },
        }
    )
    _, _, correct = render_citations(
        "上海天气29℃、多云、东风3级。",
        EvidenceSet(query="查询天气"),
        tool_invocations=[weather],
    )
    assert not correct, correct
    for wrong in ("上海天气晴朗。", "上海天气29℃、西风3级。"):
        _, _, violations = render_citations(
            wrong,
            EvidenceSet(query="查询天气"),
            tool_invocations=[weather],
        )
        assert any(
            item.get("code") == "unsupported_tool_fact"
            and item.get("detail")
            for item in violations
        ), (wrong, violations)

    work_order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "待派单"},
        }
    )
    _, _, violations = render_citations(
        "工单 WO-100 状态已完成。",
        EvidenceSet(query="查询最近工单"),
        tool_invocations=[work_order],
    )
    assert any(
        item.get("code") == "unsupported_tool_fact"
        for item in violations
    ), violations


def test_structured_rows_cannot_cross_pair_ids_and_statuses():
    orders = _successful_invocation(
        {
            "status": "success",
            "data": {
                "items": [
                    {"id": "WO-100", "status": "待派单"},
                    {"id": "WO-200", "status": "已完成"},
                ]
            },
        }
    )
    _, _, valid = render_citations(
        "WO-100状态待派单；WO-200状态已完成。",
        EvidenceSet(query="查询工单"),
        tool_invocations=[orders],
    )
    assert not valid, valid
    _, _, violations = render_citations(
        "WO-100状态已完成；WO-200状态待派单。",
        EvidenceSet(query="查询工单"),
        tool_invocations=[orders],
    )
    assert sum(
        item.get("code") == "unsupported_tool_fact"
        for item in violations
    ) >= 2, violations


def test_status_before_identifier_cannot_cross_pair_rows():
    orders = _successful_invocation(
        {
            "status": "success",
            "data": {
                "items": [
                    {"id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
                    {"id": "WO-200", "status": "\u5df2\u5b8c\u6210"},
                ]
            },
        }
    )
    answer = (
        "\u72b6\u6001\u5df2\u5b8c\u6210\u7684\u5de5\u5355\u662fWO-100\uff0c"
        "\u72b6\u6001\u5f85\u6d3e\u5355\u7684\u5de5\u5355\u662fWO-200\u3002"
    )
    bundle = build_run_evidence_bundle(
        EvidenceSet(query="\u67e5\u8be2\u5de5\u5355"),
        tool_invocations=[orders],
    )
    _, _, violations, final_bundle = render_bundle_citations(answer, bundle)
    assert sum(
        item.get("code") == "unsupported_tool_fact"
        for item in violations
    ) >= 2, violations
    assert final_bundle.tool_evidence_links == []


def test_split_wind_fields_reject_wrong_direction_without_selector():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {"wind_direction": "\u4e1c\u98ce", "wind_level": 3},
        }
    )
    _, _, violations = render_citations(
        "\u5f53\u524d\u98ce\u5411\u4e3a\u897f\u98ce\u3002",
        EvidenceSet(query="\u67e5\u8be2\u5f53\u524d\u5929\u6c14"),
        tool_invocations=[weather],
    )
    assert any(
        item.get("code") == "unsupported_tool_fact"
        and item.get("semantic_type") == "wind_direction"
        for item in violations
    ), violations


def test_split_wind_fields_accept_correct_natural_word_order():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {"wind_direction": "\u4e1c\u98ce", "wind_level": 3},
        }
    )
    _, _, violations = render_citations(
        "\u5f53\u524d\u98ce\u529b\u4e3a3\u7ea7\uff0c\u98ce\u5411\u4e3a\u4e1c\u98ce\u3002",
        EvidenceSet(query="\u67e5\u8be2\u5f53\u524d\u5929\u6c14"),
        tool_invocations=[weather],
    )
    assert not violations, violations


def test_wrong_identifier_cannot_borrow_a_real_status_link():
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "待派单"},
        }
    )
    for answer in (
        "最近工单为WO-999，状态待派单。",
        "WO-100已关闭。",
    ):
        bundle = build_run_evidence_bundle(
            EvidenceSet(query="查询最近工单"),
            tool_invocations=[order],
        )
        _, _, violations, final_bundle = render_bundle_citations(answer, bundle)
        assert any(
            item.get("code") == "unsupported_tool_fact"
            for item in violations
        ), (answer, violations)
        assert final_bundle.tool_evidence_links == []
        assert final_bundle.delivered_evidence_ids == []


def test_different_identifier_family_cannot_borrow_real_status():
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
        }
    )
    answer = "\u6700\u8fd1\u5de5\u5355\u4e3aAB-999\uff0c\u72b6\u6001\u5f85\u6d3e\u5355\u3002"
    bundle = build_run_evidence_bundle(
        EvidenceSet(query="\u67e5\u8be2\u6700\u8fd1\u5de5\u5355"),
        tool_invocations=[order],
    )
    _, _, violations, final_bundle = render_bundle_citations(answer, bundle)
    assert any(
        item.get("code") == "unsupported_tool_fact"
        and item.get("semantic_type") == "business_identifier"
        for item in violations
    ), violations
    assert final_bundle.tool_evidence_links == []


def test_rag_status_claim_is_not_misclassified_as_tool_claim():
    content = "设备状态由值班人员登记。"
    item = EvidenceItem(
        evidence_id="ev_status_policy",
        knowledge_id="doc-status",
        knowledge_version="v1",
        document_id="doc-status",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="status-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="设备登记制度",
    )
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "待派单"},
        }
    )
    _, citations, violations = render_citations(
        "设备状态由值班人员登记 [[evidence:ev_status_policy]]。",
        EvidenceSet(items=[item], query="说明设备登记制度"),
        tool_invocations=[order],
    )
    assert len(citations) == 1, citations
    assert not any(
        violation.get("code") == "unsupported_tool_fact"
        for violation in violations
    ), violations


def test_rag_identifier_and_shared_status_do_not_adopt_tool_evidence():
    content = "\u8bbe\u5907DEV-100\u7684\u72b6\u6001\u5f85\u6d3e\u5355\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0\u3002"
    item = EvidenceItem(
        evidence_id="ev_status_policy",
        knowledge_id="doc-status",
        knowledge_version="v1",
        document_id="doc-status",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="status-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="\u8bbe\u5907\u767b\u8bb0\u5236\u5ea6",
    )
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
        }
    )
    bundle = build_run_evidence_bundle(
        EvidenceSet(items=[item], query="\u8bf4\u660e\u8bbe\u5907\u767b\u8bb0\u5236\u5ea6"),
        tool_invocations=[order],
    )
    answer = (
        "\u8bbe\u5907DEV-100\u7684\u72b6\u6001\u5f85\u6d3e\u5355\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0 "
        "[[evidence:ev_status_policy]]\u3002"
    )
    _, citations, violations, final_bundle = render_bundle_citations(answer, bundle)
    assert len(citations) == 1, citations
    assert not violations, violations
    assert final_bundle.tool_evidence_links == []


def test_sentence_final_rag_marker_covers_each_supported_subclause():
    content = (
        "\u8bbe\u5907DEV-100\u72b6\u6001\u5f85\u6d3e\u5355\uff0c"
        "\u503c\u73ed\u4eba\u5458\u5e94\u767b\u8bb0\u3002"
    )
    item = EvidenceItem(
        evidence_id="ev_status_policy",
        knowledge_id="doc-status",
        knowledge_version="v1",
        document_id="doc-status",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="status-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="\u8bbe\u5907\u767b\u8bb0\u5236\u5ea6",
    )
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
        }
    )
    answer = (
        "\u8bbe\u5907DEV-100\u72b6\u6001\u5f85\u6d3e\u5355\uff0c"
        "\u503c\u73ed\u4eba\u5458\u5e94\u767b\u8bb0 [[evidence:ev_status_policy]]\u3002"
    )
    bundle = build_run_evidence_bundle(
        EvidenceSet(items=[item], query="\u8bf4\u660e\u8bbe\u5907\u767b\u8bb0"),
        tool_invocations=[order],
    )
    _, citations, violations, final_bundle = render_bundle_citations(answer, bundle)
    assert len(citations) == 1, citations
    assert not violations, violations
    assert final_bundle.tool_evidence_links == []


def test_sentence_final_rag_marker_cannot_hide_unsupported_tool_sibling():
    content = "\u8bbe\u5907\u4fe1\u606f\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0\u3002"
    item = EvidenceItem(
        evidence_id="ev_status_policy",
        knowledge_id="doc-status",
        knowledge_version="v1",
        document_id="doc-status",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="status-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="\u8bbe\u5907\u767b\u8bb0\u5236\u5ea6",
    )
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
        }
    )
    answer = (
        "\u8bbe\u5907\u4fe1\u606f\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0\uff0c"
        "\u5de5\u5355\u72b6\u6001\u5df2\u5173\u95ed [[evidence:ev_status_policy]]\u3002"
    )
    _, citations, violations = render_citations(
        answer,
        EvidenceSet(items=[item], query="\u8bf4\u660e\u767b\u8bb0\u5e76\u67e5\u8be2\u5de5\u5355"),
        tool_invocations=[order],
    )
    assert len(citations) == 1, citations
    assert any(
        item.get("code") == "unsupported_tool_fact"
        and item.get("semantic_type") == "business_status"
        for item in violations
    ), violations


def test_rag_marker_does_not_hide_wrong_tool_status_in_later_subclause():
    content = "\u8bbe\u5907\u72b6\u6001\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0\u3002"
    item = EvidenceItem(
        evidence_id="ev_status_policy",
        knowledge_id="doc-status",
        knowledge_version="v1",
        document_id="doc-status",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="status-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="\u8bbe\u5907\u767b\u8bb0\u5236\u5ea6",
    )
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
        }
    )
    answer = (
        "\u8bbe\u5907\u72b6\u6001\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0 "
        "[[evidence:ev_status_policy]]\uff0c\u5de5\u5355\u72b6\u6001\u5df2\u5173\u95ed\u3002"
    )
    _, citations, violations = render_citations(
        answer,
        EvidenceSet(items=[item], query="\u8bf4\u660e\u8bbe\u5907\u767b\u8bb0\u5e76\u67e5\u8be2\u5de5\u5355"),
        tool_invocations=[order],
    )
    assert len(citations) == 1, citations
    assert any(
        item.get("code") == "unsupported_tool_fact"
        and item.get("semantic_type") == "business_status"
        for item in violations
    ), violations


def test_rag_marker_does_not_hide_wrong_tool_status_after_colon_or_dash():
    content = "\u8bbe\u5907\u72b6\u6001\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0\u3002"
    item = EvidenceItem(
        evidence_id="ev_status_policy",
        knowledge_id="doc-status",
        knowledge_version="v1",
        document_id="doc-status",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="status-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="\u8bbe\u5907\u767b\u8bb0\u5236\u5ea6",
    )
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
        }
    )
    for separator in ("\uff1a", "\u2014"):
        answer = (
            "\u8bbe\u5907\u72b6\u6001\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0 "
            f"[[evidence:ev_status_policy]]{separator}\u5de5\u5355\u72b6\u6001\u5df2\u5173\u95ed\u3002"
        )
        _, _, violations = render_citations(
            answer,
            EvidenceSet(items=[item], query="\u8bf4\u660e\u767b\u8bb0\u5e76\u67e5\u8be2\u5de5\u5355"),
            tool_invocations=[order],
        )
        assert any(
            item.get("code") == "unsupported_tool_fact"
            and item.get("semantic_type") == "business_status"
            for item in violations
        ), (separator, violations)


def test_arbitrary_business_status_values_stay_bound_to_their_rows():
    orders = _successful_invocation(
        {
            "status": "success",
            "data": {
                "items": [
                    {"id": "WO-100", "status": "\u65b0\u5efa"},
                    {"id": "WO-200", "status": "\u5173\u95ed"},
                ]
            },
        }
    )
    answer = (
        "\u5173\u95ed\u7684\u5de5\u5355\u662fWO-100\uff0c"
        "\u65b0\u5efa\u7684\u5de5\u5355\u662fWO-200\u3002"
    )
    _, _, violations = render_citations(
        answer,
        EvidenceSet(query="\u67e5\u8be2\u5de5\u5355"),
        tool_invocations=[orders],
    )
    assert sum(
        item.get("code") == "unsupported_tool_fact"
        for item in violations
    ) >= 2, violations


def test_numeric_fact_cannot_borrow_same_unit_from_sibling_field():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {
                "city": "\u4e0a\u6d77",
                "humidity_pct": 70,
                "rain_probability_pct": 71,
            },
        }
    )
    bundle = build_run_evidence_bundle(
        EvidenceSet(query="\u67e5\u8be2\u4e0a\u6d77\u5929\u6c14"),
        tool_invocations=[weather],
    )
    _, _, violations, final_bundle = render_bundle_citations(
        "\u4e0a\u6d77\u6e7f\u5ea671%\u3002", bundle
    )
    assert any(
        item.get("code") == "unsupported_tool_fact"
        and item.get("semantic_type") == "structured_numeric_result"
        for item in violations
    ), violations
    assert final_bundle.tool_evidence_links == []

    _, _, correct, correct_bundle = render_bundle_citations(
        "\u4e0a\u6d77\u6e7f\u5ea670%\u3002", bundle
    )
    assert not correct, correct
    linked_paths = {
        fact.fact_id: fact.json_path
        for fact in weather.tool_evidence.facts
    }
    adopted_paths = {
        linked_paths[fact_id]
        for link in correct_bundle.tool_evidence_links
        for fact_id in link.fact_ids
    }
    assert "$.data.humidity_pct" in adopted_paths
    assert "$.data.rain_probability_pct" not in adopted_paths


def test_numeric_fact_cannot_borrow_same_field_from_other_location():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {
                "items": [
                    {"city": "\u4e0a\u6d77", "humidity_pct": 70},
                    {"city": "\u5317\u4eac", "humidity_pct": 45},
                ]
            },
        }
    )
    _, _, violations = render_citations(
        "\u4e0a\u6d77\u6e7f\u5ea645%\u3002",
        EvidenceSet(query="\u67e5\u8be2\u4e0a\u6d77\u5929\u6c14"),
        tool_invocations=[weather],
    )
    assert any(
        item.get("code") == "unsupported_tool_fact"
        and item.get("semantic_type") == "structured_numeric_result"
        for item in violations
    ), violations


def test_generic_field_tokens_bind_relative_humidity_and_precipitation():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {
                "city": "\u4e0a\u6d77",
                "relative_humidity_percent": 70,
                "precipitation_probability_pct": 71,
            },
        }
    )
    bundle = build_run_evidence_bundle(
        EvidenceSet(query="\u67e5\u8be2\u4e0a\u6d77\u5929\u6c14"),
        tool_invocations=[weather],
    )
    path_by_id = {
        fact.fact_id: fact.json_path for fact in weather.tool_evidence.facts
    }
    cases = (
        ("\u4e0a\u6d77\u6e7f\u5ea670%\u3002", "$.data.relative_humidity_percent"),
        (
            "\u4e0a\u6d77\u964d\u6c34\u6982\u738771%\u3002",
            "$.data.precipitation_probability_pct",
        ),
    )
    for answer, expected_path in cases:
        _, _, violations, final_bundle = render_bundle_citations(answer, bundle)
        assert not violations, (answer, violations)
        linked_paths = {
            path_by_id[fact_id]
            for link in final_bundle.tool_evidence_links
            for fact_id in link.fact_ids
        }
        assert expected_path in linked_paths, (answer, linked_paths)
        other_path = (
            "$.data.precipitation_probability_pct"
            if expected_path.endswith("relative_humidity_percent")
            else "$.data.relative_humidity_percent"
        )
        assert other_path not in linked_paths, (answer, linked_paths)


def test_recognized_numeric_domain_never_falls_back_to_other_field():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {"city": "\u4e0a\u6d77", "rain_probability_pct": 71},
        }
    )
    _, _, violations = render_citations(
        "\u4e0a\u6d77\u6e7f\u5ea671%\u3002",
        EvidenceSet(query="\u67e5\u8be2\u4e0a\u6d77\u5929\u6c14"),
        tool_invocations=[weather],
    )
    assert any(
        item.get("code") == "unsupported_tool_fact"
        and item.get("semantic_type") == "structured_numeric_result"
        for item in violations
    ), violations


def test_one_supported_marker_cannot_adopt_an_unrelated_marker():
    first_content = "\u7269\u4e1a\u5ba2\u670d5\u5206\u949f\u5185\u767b\u8bb0\u3002"
    second_content = "\u7eff\u5316\u6d47\u6c34\u5b89\u6392\u5728\u5468\u4e00\u3002"
    items = [
        EvidenceItem(
            evidence_id="ev_service",
            knowledge_id="doc-service",
            knowledge_version="v1",
            document_id="doc-service",
            document_version="v1",
            document_hash=content_hash(first_content),
            chunk_id="service-1",
            chunk_index=1,
            chunk_hash=content_hash(first_content),
            content_snapshot=first_content,
            retrieval_score=0.9,
            retrieval_mode="fixture",
            title="\u670d\u52a1\u627f\u8bfa",
        ),
        EvidenceItem(
            evidence_id="ev_green",
            knowledge_id="doc-green",
            knowledge_version="v1",
            document_id="doc-green",
            document_version="v1",
            document_hash=content_hash(second_content),
            chunk_id="green-1",
            chunk_index=1,
            chunk_hash=content_hash(second_content),
            content_snapshot=second_content,
            retrieval_score=0.8,
            retrieval_mode="fixture",
            title="\u7eff\u5316\u5b89\u6392",
        ),
    ]
    bundle = build_run_evidence_bundle(
        EvidenceSet(items=items, query="\u767b\u8bb0\u65f6\u9650"),
    )
    _, citations, violations, final_bundle = render_bundle_citations(
        "\u7269\u4e1a\u5ba2\u670d5\u5206\u949f\u5185\u767b\u8bb0 "
        "[[evidence:ev_service]] [[evidence:ev_green]]\u3002",
        bundle,
    )
    assert [item.evidence_id for item in citations] == ["ev_service"], citations
    assert any(
        item.get("code") == "unsupported_evidence_citation"
        and item.get("evidence_id") == "ev_green"
        for item in violations
    ), violations
    assert final_bundle.delivered_evidence_ids == []


def test_same_rag_value_cannot_ground_a_different_uncited_fact():
    content = "\u7d27\u6025\u7ef4\u4fee5\u5206\u949f\u5185\u767b\u8bb0\u3002"
    item = EvidenceItem(
        evidence_id="ev_emergency",
        knowledge_id="doc-service",
        knowledge_version="v1",
        document_id="doc-service",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="service-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="\u670d\u52a1\u627f\u8bfa",
    )
    _, _, violations = render_citations(
        "\u7d27\u6025\u7ef4\u4fee5\u5206\u949f\u5185\u767b\u8bb0 "
        "[[evidence:ev_emergency]]\u3002"
        "\u666e\u901a\u4fdd\u6d01\u4e5f\u5fc5\u987b5\u5206\u949f\u5185\u5b8c\u6210\u3002",
        EvidenceSet(items=[item], query="\u8bf4\u660e\u670d\u52a1\u65f6\u9650"),
    )
    assert any(
        item.get("code") == "ungrounded_critical_value"
        and "\u666e\u901a\u4fdd\u6d01" in item.get("claim_context", "")
        for item in violations
    ), violations


def test_same_tool_value_cannot_ground_an_unbound_business_fact():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {"city": "\u4e0a\u6d77", "humidity_pct": 70},
        }
    )
    _, _, violations = render_citations(
        "\u4e0a\u6d77\u6e7f\u5ea670%\u3002\u8bbe\u5907\u5b8c\u5de5\u738770%\u3002",
        EvidenceSet(query="\u67e5\u8be2\u4e0a\u6d77\u5929\u6c14\u548c\u8bbe\u5907"),
        tool_invocations=[weather],
    )
    assert any(
        item.get("code") == "ungrounded_critical_value"
        and "\u8bbe\u5907\u5b8c\u5de5\u7387" in item.get("claim_context", "")
        for item in violations
    ), violations


def test_location_selector_cannot_replace_numeric_field_semantics():
    weather = _successful_invocation(
        {
            "status": "success",
            "data": {"city": "\u4e0a\u6d77", "humidity_pct": 70},
        }
    )
    for answer in (
        "\u4e0a\u6d77\u5de5\u5355\u5b8c\u6210\u738770%\u3002",
        "\u4e0a\u6d77\u8bbe\u5907\u5b8c\u5de5\u738770%\u3002",
    ):
        _, _, violations = render_citations(
            answer,
            EvidenceSet(query="\u67e5\u8be2\u4e0a\u6d77\u8bbe\u5907"),
            tool_invocations=[weather],
        )
        assert any(
            item.get("code") == "ungrounded_critical_value"
            for item in violations
        ), (answer, violations)


def test_conjunction_cannot_hide_wrong_tool_fact_behind_sentence_marker():
    content = "\u8bbe\u5907\u4fe1\u606f\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0\u3002"
    item = EvidenceItem(
        evidence_id="ev_status_policy",
        knowledge_id="doc-status",
        knowledge_version="v1",
        document_id="doc-status",
        document_version="v1",
        document_hash=content_hash(content),
        chunk_id="status-1",
        chunk_index=1,
        chunk_hash=content_hash(content),
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="fixture",
        title="\u8bbe\u5907\u767b\u8bb0\u5236\u5ea6",
    )
    order = _successful_invocation(
        {
            "status": "success",
            "data": {"work_order_id": "WO-100", "status": "\u5f85\u6d3e\u5355"},
        }
    )
    for connector in ("\u4e14", "\u5e76\u4e14", "\u540c\u65f6"):
        answer = (
            "\u8bbe\u5907\u4fe1\u606f\u7531\u503c\u73ed\u4eba\u5458\u767b\u8bb0"
            f"{connector}\u5de5\u5355\u72b6\u6001\u5df2\u5173\u95ed "
            "[[evidence:ev_status_policy]]\u3002"
        )
        _, _, violations = render_citations(
            answer,
            EvidenceSet(items=[item], query="\u8bf4\u660e\u767b\u8bb0\u5e76\u67e5\u8be2\u5de5\u5355"),
            tool_invocations=[order],
        )
        assert any(
            violation.get("code") == "unsupported_tool_fact"
            and violation.get("semantic_type") == "business_status"
            for violation in violations
        ), (connector, violations)


def test_model_native_wrapper_returns_only_frozen_safe_context():
    assert GovernedMCPTools is not None
    toolkit = SimpleNamespace(
        server_name="fixture-server",
        result_contracts={"lookup": {"success_statuses": ["success"]}},
        recorded_invocations=[],
    )

    async def success(**_kwargs):
        return {
            "status": "success",
            "data": {
                "result": 65,
                "api_key": "must-never-reach-agent",
            },
        }

    wrapped = GovernedMCPTools._wrap_entrypoint(toolkit, success, "lookup")
    context = asyncio.run(wrapped(requested_value=70))
    assert "$.data.result = 65" in context
    assert "must-never-reach-agent" not in context
    assert "requested_value" not in context
    assert "70" not in context

    async def failed(**_kwargs):
        return {
            "status": "not_found",
            "data": {"internal_reason": "raw-failure-body"},
        }

    failed_context = asyncio.run(
        GovernedMCPTools._wrap_entrypoint(toolkit, failed, "lookup")()
    )
    assert "business_status=not_found" in failed_context
    assert "raw-failure-body" not in failed_context
    assert toolkit.recorded_invocations[-1].tool_evidence is None


def test_committed_receipt_requires_real_resource_id():
    missing_resource = ActionReceipt(
        receipt_id="receipt-1",
        proposal_id="proposal-1",
        idempotency_key="idem-1",
        status="committed",
        resource_id=None,
    )
    valid = missing_resource.model_copy(
        update={"receipt_id": "receipt-2", "resource_id": "WO-100"}
    )
    bundle = build_run_evidence_bundle(
        EvidenceSet(query="q"),
        action_receipts=[missing_resource, valid],
    )
    assert [item.receipt_id for item in bundle.committed_receipts] == ["receipt-2"]


def test_calculator_result_84_still_supports_user_requested_unit():
    invocation = _successful_invocation(
        {"metadata": {"structured_content": {"result": 84}}}
    )
    _, _, violations = render_citations(
        "计算结果是84份。",
        EvidenceSet(query="请计算总共多少份"),
        tool_invocations=[invocation],
    )
    assert not violations, violations
