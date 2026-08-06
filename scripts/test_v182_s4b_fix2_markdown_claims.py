"""No-model contracts for S4-B-Fix2 Markdown claim normalization."""

import json
from pathlib import Path

from app.runtime.citation_renderer import render_citations
from app.runtime.contracts import EvidenceItem, EvidenceSet
from app.runtime.mcp_executor import _evaluate_read_tool_result


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "s4b_fix2_trace_164ecfcec7ab46df.json"
)


def _item(evidence_id: str, content: str, chunk_index: int = 0) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        knowledge_id="124",
        knowledge_version="cfa36b3d0adadfa5",
        document_id="124",
        document_version="cfa36b3d0adadfa5",
        document_hash="doc-hash",
        chunk_id=f"doc-124-chunk-{chunk_index}",
        chunk_index=chunk_index,
        chunk_hash=f"chunk-hash-{chunk_index}",
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="keyword+semantic",
        title="火星温室补给规则（验收专用）",
    )


def _fixture():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    for index, invocation in enumerate(payload["tool_invocations"], start=1):
        invocation_id = f"fixture_tool_{index}"
        _, summary, tool_evidence = _evaluate_read_tool_result(
            invocation.pop("structured_result"),
            {},
            invocation_id=invocation_id,
            server_name=invocation["server_name"],
            tool_name=invocation["tool_name"],
        )
        invocation["invocation_id"] = invocation_id
        invocation["result_summary"] = summary
        invocation["tool_evidence"] = tool_evidence.model_dump(mode="json")
    evidence = EvidenceSet(
        query=payload["question"],
        items=[
            _item(item["evidence_id"], item["content"], item["chunk_index"])
            for item in payload["evidence"]
        ],
        retrieval_status="completed",
    )
    return payload, evidence


def test_real_trace_fixture_renders_84_with_direct_citation():
    payload, evidence = _fixture()
    rendered, citations, violations = render_citations(
        payload["raw_answer"], evidence, payload["tool_invocations"]
    )
    assert "84 份标准营养液" in rendered
    assert not violations, violations
    by_id = {item.evidence_id: item for item in citations}
    assert "ev_65882f4cc021cd80a2c5" in by_id
    assert "每天需要3份" in by_id["ev_65882f4cc021cd80a2c5"].content_snapshot
    assert "【引用" in rendered


def test_table_row_is_one_normalized_fact():
    _, evidence = _fixture()
    rendered, citations, violations = render_citations(
        "| 项目 | 值 | 证据 |\n"
        "|---|---|---|\n"
        "| 每舱每日标准份数 | **3 份** | "
        "[[evidence:ev_65882f4cc021cd80a2c5]] |",
        evidence,
    )
    assert len(citations) == 1
    assert "【引用1】" in rendered
    assert not violations, violations


def test_citation_only_line_is_not_a_new_fact():
    _, evidence = _fixture()
    rendered, citations, violations = render_citations(
        "种植舱数量 × 连续天数 × 每舱每日标准份数 = 标准营养液总份数\n"
        "> — [[evidence:ev_2ab20a020a2ba87234eb]]",
        evidence,
    )
    assert len(citations) == 1
    assert "> — 【引用1】" in rendered
    assert not violations, violations


def test_orphan_citation_display_is_ignored_without_fake_claim():
    _, evidence = _fixture()
    rendered, citations, violations = render_citations(
        "> — [[evidence:ev_2ab20a020a2ba87234eb]]",
        evidence,
    )
    assert not citations
    assert "[[evidence:" not in rendered
    assert not violations, violations


def test_unrelated_standalone_citation_is_rejected():
    _, evidence = _fixture()
    _, citations, violations = render_citations(
        "物业费每月100元。\n"
        "> — [[evidence:ev_2ab20a020a2ba87234eb]]",
        evidence,
    )
    assert not citations
    assert any(item["code"] == "unsupported_evidence_citation" for item in violations)


def test_table_hallucinated_number_remains_blocked():
    _, evidence = _fixture()
    _, citations, violations = render_citations(
        "| 项目 | 值 | 证据 |\n"
        "|---|---|---|\n"
        "| 每舱每日标准份数 | 5份 | "
        "[[evidence:ev_65882f4cc021cd80a2c5]] |",
        evidence,
    )
    assert not citations
    assert any(
        item["code"] == "unsupported_critical_value" and "5份" in item["values"]
        for item in violations
    ), violations


def test_multiple_citations_share_one_claim_when_one_directly_supports_it():
    _, evidence = _fixture()
    rendered, citations, violations = render_citations(
        "每舱每日标准份数为3份，并按多因子公式计算。"
        "[[evidence:ev_65882f4cc021cd80a2c5]]"
        "[[evidence:ev_2ab20a020a2ba87234eb]]",
        evidence,
    )
    assert len(citations) == 2
    assert rendered.count("【引用") == 2
    assert not violations, violations


def test_workday_mismatch_remains_blocked():
    evidence = EvidenceSet(
        query="维修投诉多久反馈？",
        items=[_item("ev_workday", "维修投诉应在3个工作日内反馈。")],
    )
    _, citations, violations = render_citations(
        "维修投诉应在5个工作日内反馈。[[evidence:ev_workday]]", evidence
    )
    assert not citations
    assert any(
        item["code"] == "unsupported_critical_value"
        and "5工作日" in item["values"]
        for item in violations
    ), violations


def test_tool_arguments_without_success_result_do_not_ground_output():
    _, evidence = _fixture()
    _, _, violations = render_citations(
        "按公式计算后总计28份。[[evidence:ev_2ab20a020a2ba87234eb]]",
        evidence,
        [
            {
                "arguments": {"a": 4, "b": 7},
                "invocation_status": "failed",
                "business_status": "unknown",
                "result_summary": None,
            }
        ],
    )
    assert any(
        item["code"] in {"unsupported_critical_value", "ungrounded_critical_value"}
        and "28份" in item.get("values", [])
        for item in violations
    ), violations


def test_institutional_claim_without_direct_evidence_is_rejected():
    _, evidence = _fixture()
    _, citations, violations = render_citations(
        "制度明确规定所有物业服务永久免费。"
        "[[evidence:ev_65882f4cc021cd80a2c5]]",
        evidence,
    )
    assert not citations
    assert any(item["code"] == "unsupported_evidence_citation" for item in violations)


def test_fix1_user_rag_tool_sources_and_list_prose_still_pass():
    payload, evidence = _fixture()
    answer = (
        "- 4个种植舱连续7天，每舱每天3份。"
        "[[evidence:ev_65882f4cc021cd80a2c5]]\n"
        "- 总量按种植舱数量、天数和每日份数相乘。"
        "[[evidence:ev_2ab20a020a2ba87234eb]]\n"
        "计算：4 × 7 = 28，28 × 3 = 84份。"
    )
    rendered, citations, violations = render_citations(
        answer, evidence, payload["tool_invocations"]
    )
    assert "84份" in rendered
    assert len(citations) == 2
    assert not violations, violations


def main():
    tests = [
        test_real_trace_fixture_renders_84_with_direct_citation,
        test_table_row_is_one_normalized_fact,
        test_citation_only_line_is_not_a_new_fact,
        test_orphan_citation_display_is_ignored_without_fake_claim,
        test_unrelated_standalone_citation_is_rejected,
        test_table_hallucinated_number_remains_blocked,
        test_multiple_citations_share_one_claim_when_one_directly_supports_it,
        test_workday_mismatch_remains_blocked,
        test_tool_arguments_without_success_result_do_not_ground_output,
        test_institutional_claim_without_direct_evidence_is_rejected,
        test_fix1_user_rag_tool_sources_and_list_prose_still_pass,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"V1.8.2-S4-B-Fix2 citation contracts passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
