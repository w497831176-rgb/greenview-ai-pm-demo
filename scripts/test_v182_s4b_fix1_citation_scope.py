"""No-model contracts for S4-B-Fix1 atomic citation value validation."""

from app.runtime.citation_renderer import render_citations
from app.runtime.contracts import EvidenceItem, EvidenceSet, ToolEffect, ToolInvocation


def _evidence(evidence_id: str, chunk_index: int, content: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        knowledge_id="124",
        knowledge_version="v1",
        document_id="124",
        document_version="v1",
        document_hash="doc-hash",
        chunk_id=f"chunk-{chunk_index}",
        chunk_index=chunk_index,
        chunk_hash=f"chunk-hash-{chunk_index}",
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="keyword+semantic",
        title="火星温室补给规则（验收专用）",
    )


def _mars_evidence() -> EvidenceSet:
    return EvidenceSet(
        query=(
            "根据《火星温室补给规则（验收专用）》，"
            "4个种植舱连续7天需要多少份标准营养液？"
        ),
        items=[
            _evidence(
                "ev_rule",
                0,
                "每个种植舱每天需要3份标准营养液。",
            ),
            _evidence(
                "ev_formula",
                1,
                "总需求量计算公式为：种植舱数量 × 连续天数 × "
                "每舱每日标准份数 = 标准营养液总份数。",
            ),
            _evidence(
                "ev_example",
                2,
                "例如：2个种植舱连续5天，需要2 × 5 × 3 = 30份标准营养液。",
            ),
        ],
        retrieval_status="completed",
    )


def _successful_multiply(a: float, b: float, result: float) -> ToolInvocation:
    return ToolInvocation(
        server_name="mars-calculator-mcp",
        tool_name="multiply",
        effect=ToolEffect.READ,
        arguments={"a": a, "b": b},
        discovery_status="success",
        transport_status="success",
        invocation_status="success",
        business_status="success",
        result_summary=(
            f"content='{result}' metadata={{'structured_content': "
            f"{{'result': {result}}}}}"
        ),
    )


def test_mars_answer_accepts_user_rag_and_tool_values():
    evidence = _mars_evidence()
    answer = (
        "4个种植舱连续7天，每舱每天3份。"
        "[[evidence:ev_rule]]\n"
        "依照公式计算总需求量。[[evidence:ev_formula]]\n"
        "计算：4 × 7 = 28，28 × 3 = 84份。"
    )
    rendered, citations, violations = render_citations(
        answer,
        evidence,
        tool_invocations=[
            _successful_multiply(4, 7, 28),
            _successful_multiply(28, 3, 84),
        ],
    )
    assert "84份" in rendered
    assert {item.evidence_id for item in citations} == {"ev_rule", "ev_formula"}
    assert not violations, violations


def test_adjacent_citations_do_not_share_example_values():
    evidence = _mars_evidence()
    answer = (
        "- 每舱每日标准 **3份** → [[evidence:ev_rule]]\n"
        "- 计算公式 **种植舱数量 × 连续天数 × 每舱每日标准份数** "
        "→ [[evidence:ev_formula]]\n"
        "- 同类示例佐证（2个种植舱连续5天=30份）→ "
        "[[evidence:ev_example]]"
    )
    _, citations, violations = render_citations(answer, evidence)
    assert len(citations) == 3
    assert not violations, violations


def test_unused_retrieved_example_values_do_not_pollute_answer():
    evidence = _mars_evidence()
    answer = (
        "每个种植舱每天需要3份标准营养液。[[evidence:ev_rule]]\n"
        "总量按种植舱数量、天数和每舱每日份数相乘。"
        "[[evidence:ev_formula]]"
    )
    _, citations, violations = render_citations(answer, evidence)
    assert len(citations) == 2
    assert not violations, violations


def test_workday_mismatch_remains_blocked():
    evidence = EvidenceSet(
        query="维修投诉多久反馈？",
        items=[_evidence("ev_workday", 0, "维修投诉应在3个工作日内反馈。")],
    )
    _, citations, violations = render_citations(
        "维修投诉应在5个工作日内反馈。[[evidence:ev_workday]]",
        evidence,
    )
    assert not citations
    assert any(
        item["code"] == "unsupported_critical_value"
        and "5工作日" in item["values"]
        for item in violations
    ), violations


def test_hallucinated_duration_without_any_source_is_blocked():
    evidence = EvidenceSet(
        query="这项服务多久能完成？",
        items=[_evidence("ev_service", 0, "该服务需先完成申请审核。")],
    )
    _, citations, violations = render_citations(
        "该服务需先完成申请审核，并在5天内完成。"
        "[[evidence:ev_service]]",
        evidence,
    )
    assert not citations
    assert any(
        item["code"] == "unsupported_critical_value"
        and "5天" in item["values"]
        for item in violations
    ), violations


def test_each_atomic_claim_uses_only_its_own_citation():
    evidence = EvidenceSet(
        query="登记和到场时效是什么？",
        items=[
            _evidence("ev_register", 0, "客服中心5分钟内登记。"),
            _evidence("ev_arrive", 1, "工程人员30分钟内到场。"),
        ],
    )
    answer = (
        "- 客服中心5分钟内登记。[[evidence:ev_register]]\n"
        "- 工程人员30分钟内到场。[[evidence:ev_arrive]]"
    )
    _, citations, violations = render_citations(answer, evidence)
    assert len(citations) == 2
    assert not violations, violations

    crossed = (
        "- 客服中心30分钟内登记。[[evidence:ev_register]]\n"
        "- 工程人员5分钟内到场。[[evidence:ev_arrive]]"
    )
    _, citations, violations = render_citations(crossed, evidence)
    assert not citations
    assert len(
        [item for item in violations if item["code"] == "unsupported_critical_value"]
    ) == 2


def main():
    tests = [
        test_mars_answer_accepts_user_rag_and_tool_values,
        test_adjacent_citations_do_not_share_example_values,
        test_unused_retrieved_example_values_do_not_pollute_answer,
        test_workday_mismatch_remains_blocked,
        test_hallucinated_duration_without_any_source_is_blocked,
        test_each_atomic_claim_uses_only_its_own_citation,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"V1.8.2-S4-B-Fix1 citation contracts passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
