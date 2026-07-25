"""No-model contracts for V1.8.2-S3 RAG evidence quality."""

import asyncio
from unittest.mock import patch

import rag_chunking
import rag_retrieval
from app.runtime.citation_renderer import (
    _critical_values as citation_critical_values,
    build_evidence_set,
    prompt_evidence_allowlist,
    render_citations,
)
from app.runtime.contracts import EvidenceItem, EvidenceSet
from app.runtime.coordinator import (
    KNOWLEDGE_INSUFFICIENT_RESPONSE,
    _knowledge_evidence_decision,
    _requires_direct_knowledge_evidence,
    _results_from_snapshot,
    _static_response_stream,
)


def _result(doc_id, chunk_index, content, score=0.8, title="物业维修服务承诺"):
    return {
        "doc_id": doc_id,
        "doc_title": title,
        "title": title,
        "chunk_index": chunk_index,
        "content": content,
        "score": score,
        "source": "keyword",
        "is_indexed": True,
    }


def test_heading_is_merged_with_fact():
    text = (
        "# 物业维修服务承诺\n\n"
        "第四章 紧急维修\n\n"
        "客服中心应在5分钟内登记，工程人员应在30分钟内到场。"
    )
    chunks = rag_chunking.split_text(text, chunk_size=512)
    assert len(chunks) == 1, chunks
    assert "第四章 紧急维修" in chunks[0]
    assert "5分钟内登记" in chunks[0]
    assert not rag_retrieval._is_structural_chunk(chunks[0], "物业维修服务承诺")
    assert rag_retrieval._is_structural_chunk("业主可以通过以下方式报修：")


def test_faq_question_and_answer_stay_together():
    text = "Q1：报修后多久上门？\n\nA：一般维修会先联系业主并预约上门。"
    chunks = rag_chunking.split_text(text, chunk_size=512)
    assert len(chunks) == 1, chunks
    assert "Q1" in chunks[0] and "A：" in chunks[0]


def test_markdown_table_repeats_header():
    text = (
        "## 收费表\n"
        "| 项目 | 价格 |\n"
        "| --- | --- |\n"
        "| 灯具更换 | 20元 |\n"
        "| 水龙头更换 | 30元 |\n"
        "| 门锁维修 | 40元 |"
    )
    chunks = rag_chunking.split_text(text, chunk_size=70, chunk_overlap=0)
    assert len(chunks) >= 2, chunks
    assert all("| 项目 | 价格 |" in chunk for chunk in chunks), chunks
    assert all("| --- | --- |" in chunk for chunk in chunks), chunks


def test_semantic_overfetch_filters_heading_before_top_k():
    rows = [
        {"doc_id": 1, "chunk_index": 0, "content": "第一章 总则", "score": 0.99},
        {"doc_id": 1, "chunk_index": 1, "content": "第四章 紧急维修", "score": 0.98},
        {
            "doc_id": 1,
            "chunk_index": 2,
            "content": "紧急维修应在30分钟内到场。",
            "score": 0.90,
        },
    ]
    doc = {"id": 1, "title": "物业维修服务承诺", "category": "维修", "is_indexed": True}
    with patch("rag_retrieval.rag_embeddings.embed_text", return_value=[0.1]), patch(
        "rag_retrieval.rag_store.search_chunks", return_value=rows
    ) as search, patch("rag_retrieval.db.get_knowledge_doc", return_value=doc):
        results = rag_retrieval._semantic_search("紧急维修多久到场", top_k=1, threshold=0)
    assert search.call_args.kwargs["top_k"] > 1
    assert len(results) == 1
    assert results[0]["chunk_index"] == 2


def test_content_only_gate_sorts_before_top_k():
    irrelevant = _result(1, 1, "本文件由物业服务中心负责解释。")
    irrelevant["title"] = irrelevant["doc_title"] = "紧急维修多久到场制度"
    relevant = _result(1, 2, "紧急维修应在30分钟内到场。")
    with patch("rag_retrieval._keyword_search", return_value=[irrelevant, relevant]), patch(
        "rag_retrieval._semantic_search", return_value=[]
    ):
        results = rag_retrieval._single_query_search(
            "紧急维修多久到场",
            {"top_k": 1, "context_threshold": 0.2},
        )
    assert len(results) == 1, results
    assert results[0]["chunk_index"] == 2


def test_semantic_gate_preserves_paraphrase_but_rejects_missing_value():
    paraphrase = _result(
        1,
        2,
        "维修完成后24小时内，物业客服人员进行回访。",
    )
    paraphrase["semantic_score"] = 0.72
    with patch("rag_retrieval._keyword_search", return_value=[paraphrase]), patch(
        "rag_retrieval._semantic_search", return_value=[]
    ):
        results = rag_retrieval._single_query_search(
            "维修结束以后，物业多长时间会回访我？",
            {"top_k": 1, "context_threshold": 0.2},
        )
    assert len(results) == 1, results
    assert results[0]["evidence_reason"] == "accepted_semantic"

    unsupported = _result(1, 3, "维修人员上门后应当清理现场。")
    unsupported["semantic_score"] = 0.80
    with patch("rag_retrieval._keyword_search", return_value=[unsupported]), patch(
        "rag_retrieval._semantic_search", return_value=[]
    ):
        results = rag_retrieval._single_query_search(
            "维修人员是否必须赠送三次免费保洁？",
            {"top_k": 1, "context_threshold": 0.2},
        )
    assert results == [], results


def test_named_document_is_scoped_but_content_must_match():
    docs = [
        {"id": 1, "title": "物业维修服务承诺", "category": "维修", "is_indexed": True},
        {"id": 2, "title": "其他文档", "category": "其他", "is_indexed": True},
    ]
    chunks = [
        {"chunk_index": 0, "content": "第一章 总则"},
        {"chunk_index": 1, "content": "紧急维修5分钟内登记，30分钟内到场。"},
    ]
    with patch("rag_retrieval.db.list_knowledge_docs", return_value=docs), patch(
        "rag_retrieval.db.get_knowledge_doc", return_value=docs[0]
    ), patch("rag_retrieval.rag_store.list_chunks_for_doc", return_value=chunks):
        results = rag_retrieval._title_boosted_results(
            "请依据《物业维修服务承诺》说明紧急维修登记和到场时限",
            "物业维修服务承诺",
            {"top_k": 5, "context_threshold": 0.2},
            allowed_document_ids={1, 2},
        )
    assert len(results) == 1, results
    assert results[0]["chunk_index"] == 1


def test_structural_chunk_never_enters_evidence_set():
    evidence = build_evidence_set(
        "紧急维修",
        [
            _result(1, 0, "第四章 紧急维修"),
            _result(1, 1, "紧急维修应在30分钟内到场。"),
        ],
        allowed_document_ids={1},
    )
    assert len(evidence.items) == 1
    assert evidence.items[0].chunk_index == 1


def test_empty_evidence_prompt_forbids_industry_fallback():
    prompt = prompt_evidence_allowlist(EvidenceSet(items=[], query="收费多少"))
    assert "不得给出确定性结论" in prompt
    assert "不得补充行业经验" in prompt


def _evidence_item(content):
    return EvidenceItem(
        evidence_id="ev_test",
        knowledge_id="1",
        knowledge_version="v1",
        document_id="1",
        document_version="v1",
        document_hash="doc-hash",
        chunk_id="chunk-1",
        chunk_index=1,
        chunk_hash="chunk-hash",
        content_snapshot=content,
        retrieval_score=0.9,
        retrieval_mode="keyword+semantic",
        title="物业维修服务承诺",
    )


def test_numeric_claim_must_be_in_cited_chunk():
    evidence = EvidenceSet(
        items=[_evidence_item("紧急维修应在30分钟内到场。")],
        query="多久到场",
    )
    _, citations, violations = render_citations(
        "紧急维修应在1小时内到场。[[evidence:ev_test]]",
        evidence,
    )
    assert not citations
    assert any(item["code"] == "unsupported_critical_value" for item in violations)

    rendered, citations, violations = render_citations(
        "紧急维修应在30分钟内到场。[[evidence:ev_test]]",
        evidence,
    )
    assert "【引用1】" in rendered
    assert len(citations) == 1
    assert not violations


def test_workday_critical_values_are_distinct_and_normalized():
    assert citation_critical_values("3个工作日、5个工作日") == {
        "3工作日",
        "5工作日",
    }
    assert citation_critical_values("3工作日") == {"3工作日"}
    assert rag_retrieval._critical_values("3个工作日、5个工作日") == {
        "3工作日",
        "5工作日",
    }
    assert citation_critical_values("5分钟、30分钟、24小时") == {
        "5分钟",
        "30分钟",
        "24小时",
    }


def test_workday_claim_mismatch_is_blocked():
    evidence = EvidenceSet(
        items=[_evidence_item("维修投诉应当在24小时内响应，并在3个工作日内反馈。")],
        query="维修投诉多久响应并反馈",
    )
    _, citations, violations = render_citations(
        "维修投诉会在24小时内响应、5个工作日内反馈。"
        "[[evidence:ev_test]]",
        evidence,
    )
    assert not citations
    assert any(
        item["code"] == "unsupported_critical_value"
        and "5工作日" in item["values"]
        for item in violations
    ), violations


def test_supported_workday_and_emergency_values_still_pass():
    workday_evidence = EvidenceSet(
        items=[_evidence_item("维修投诉应当在24小时内响应，并在3个工作日内反馈。")],
        query="维修投诉多久响应并反馈",
    )
    rendered, citations, violations = render_citations(
        "维修投诉会在24小时内响应、3个工作日内反馈。"
        "[[evidence:ev_test]]",
        workday_evidence,
    )
    assert "【引用1】" in rendered
    assert len(citations) == 1
    assert not violations

    emergency_evidence = EvidenceSet(
        items=[_evidence_item("紧急维修5分钟内登记，工程人员30分钟内到场。")],
        query="物业紧急维修响应时效",
    )
    rendered, citations, violations = render_citations(
        "紧急维修5分钟内登记、30分钟内到场。[[evidence:ev_test]]",
        emergency_evidence,
    )
    assert "【引用1】" in rendered
    assert len(citations) == 1
    assert not violations


def test_unknown_service_is_blocked_without_bound_knowledge():
    for question in (
        "小区有没有免费搬家服务？",
        "小区提供免费搬家服务吗？",
        "是否支持无人机上门维修？",
    ):
        decision = _knowledge_evidence_decision(
            question,
            evidence_count=0,
            structured_realtime_query=False,
            allowed_document_ids=set(),
        )
        assert decision["required"], (question, decision)
        assert decision["blocked"], (question, decision)
        assert decision["evidence_decision"] == "rejected_insufficient"
        assert decision["reason"] == "no_accepted_evidence"
        assert decision["model_invoked"] is False
        assert decision["allowed_document_ids"] == []


def test_supported_service_knowledge_allows_model_answer():
    decision = _knowledge_evidence_decision(
        "小区是否提供预约上门维修服务？",
        evidence_count=1,
        structured_realtime_query=False,
        allowed_document_ids={1},
    )
    assert decision["required"]
    assert not decision["blocked"]
    assert decision["evidence_decision"] == "accepted"
    assert decision["model_invoked"] is True


def test_snapshot_fallback_uses_content_gate_and_filters_heading():
    versions = {
        1: {
            "title": "物业维修服务承诺",
            "document_hash": "doc-hash",
            "document_version": "v1",
            "chunk_snapshots": [
                {"chunk_index": 0, "content": "第四章 紧急维修", "chunk_hash": "a"},
                {
                    "chunk_index": 1,
                    "content": "一般维修提供预约上门服务。",
                    "chunk_hash": "b",
                },
            ],
        }
    }
    results, used_fallback = _results_from_snapshot(
        "小区是否提供无人机上门维修？",
        [],
        versions,
        {1},
        5,
        0.2,
    )
    assert used_fallback
    assert results == [], results


def test_knowledge_gate_contract_and_static_response():
    assert _requires_direct_knowledge_evidence("维修期间是否免费安排酒店？")
    assert KNOWLEDGE_INSUFFICIENT_RESPONSE.startswith("当前知识依据不足")

    async def consume():
        return [item async for item in _static_response_stream(KNOWLEDGE_INSUFFICIENT_RESPONSE)]

    events = asyncio.run(consume())
    assert len(events) == 1
    assert events[0].content == KNOWLEDGE_INSUFFICIENT_RESPONSE


def main():
    tests = [
        test_heading_is_merged_with_fact,
        test_faq_question_and_answer_stay_together,
        test_markdown_table_repeats_header,
        test_semantic_overfetch_filters_heading_before_top_k,
        test_content_only_gate_sorts_before_top_k,
        test_semantic_gate_preserves_paraphrase_but_rejects_missing_value,
        test_named_document_is_scoped_but_content_must_match,
        test_structural_chunk_never_enters_evidence_set,
        test_empty_evidence_prompt_forbids_industry_fallback,
        test_numeric_claim_must_be_in_cited_chunk,
        test_workday_critical_values_are_distinct_and_normalized,
        test_workday_claim_mismatch_is_blocked,
        test_supported_workday_and_emergency_values_still_pass,
        test_unknown_service_is_blocked_without_bound_knowledge,
        test_supported_service_knowledge_allows_model_answer,
        test_snapshot_fallback_uses_content_gate_and_filters_heading,
        test_knowledge_gate_contract_and_static_response,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"V1.8.2-S3 RAG quality contracts passed: {len(tests)}/{len(tests)}")


if __name__ == "__main__":
    main()
