"""Offline deterministic checks for composite-query RAG evidence convergence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch


_TEMP_DATA = tempfile.TemporaryDirectory(prefix="yiai-s10i-rag-")
os.environ["PROPERTY_DATA_DIR"] = str(Path(_TEMP_DATA.name) / "data")

import rag_retrieval  # noqa: E402  (safe temp data dir must be set first)


CATALOG = [
    {
        "knowledge_doc_id": 1,
        "title": "物业维修服务承诺",
        "document_version": "v27-doc1",
        "document_hash": "doc1-hash",
    },
    {
        "knowledge_doc_id": 2,
        "title": "公共区域管理说明",
        "document_version": "v27-doc2",
        "document_hash": "doc2-hash",
    },
]


QUERIES = [
    "请依据《物业维修服务承诺》说明紧急维修登记和到场时限。",
    "请依据《物业维修服务承诺》说明紧急维修登记和到场时限，同时查询上海天气。",
    (
        "请依据《物业维修服务承诺》说明紧急维修登记和到场时限，"
        "同时查询上海天气及最近维修工单；只读、不写入、不转人工。"
    ),
    (
        "先查询最近维修工单；再查询上海天气；"
        "请依据《物业维修服务承诺》说明紧急维修登记和到场时限；"
        "只读、不写入、不转人工。"
    ),
]


def _candidate(
    doc_id: int,
    chunk_index: int,
    content: str,
    score: float,
    title: str,
):
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


def test_clause_plan_is_order_invariant_and_snapshot_scoped():
    expected_subquery = "紧急维修登记和到场时限"
    for query in QUERIES:
        plan = rag_retrieval.build_retrieval_plan(
            query,
            document_catalog=CATALOG,
            allowed_document_ids={1, 2},
        )
        named = [item for item in plan if item["named_document_scope"]]
        assert len(named) == 1, (query, plan)
        item = named[0]
        assert item["subquery"] == expected_subquery, (query, item)
        assert item["allowed_document_ids"] == [1], item
        assert item["retrieval_path"] == "named_document_clause_hybrid"
        assert item["named_document_scope"]["requested_title"] == "物业维修服务承诺"
        assert item["named_document_scope"]["matched"] is True
        assert "天气" not in item["subquery"]
        assert "工单" not in item["subquery"]
        assert "不写入" not in item["subquery"]
        assert "不转人工" not in item["subquery"]


def test_unbound_named_document_never_expands_to_global_scope():
    plan = rag_retrieval.build_retrieval_plan(
        "依据《未发布文档》说明服务时限，同时查询当前状态。",
        document_catalog=CATALOG,
        allowed_document_ids={1, 2},
    )
    named = next(item for item in plan if item["named_document_scope"])
    assert named["named_document_scope"]["matched"] is False
    assert named["allowed_document_ids"] == []


def test_real_content_gate_recalls_target_for_all_composite_variants():
    target = _candidate(
        1,
        1,
        "紧急维修5分钟内完成工单登记；工程人员30分钟内到场。",
        8.0,
        "物业维修服务承诺",
    )

    def keyword_search(subquery, *args, **kwargs):
        scope = kwargs.get("allowed_document_ids")
        if "紧急维修" in subquery and scope == {1}:
            return [dict(target)]
        return []

    for query in QUERIES:
        with patch(
            "rag_retrieval._keyword_search", side_effect=keyword_search
        ), patch("rag_retrieval._semantic_search", return_value=[]), patch(
            "rag_retrieval.rag_embeddings.get_runtime_info", return_value={}
        ):
            result = rag_retrieval.advanced_search(
                query,
                {"top_k": 5, "context_threshold": 0.2},
                allowed_document_ids=[1, 2],
                document_catalog=CATALOG,
            )
        assert result["settings"]["top_k"] == 5
        assert result["settings"]["context_threshold"] == 0.2
        assert any(
            item["doc_id"] == 1 and item["chunk_index"] == 1
            for item in result["results"]
        ), (query, result)
        target_result = next(item for item in result["results"] if item["doc_id"] == 1)
        assert target_result["subquery"] == "紧急维修登记和到场时限"
        assert target_result["named_document_scope"]["documents"][0]["document_id"] == 1
        assert target_result["retrieval_path"] == "named_document_clause_hybrid"
        assert target_result["retrieval_matches"], target_result


def test_named_document_coverage_survives_higher_scoring_other_clauses():
    def fake_single_query(subquery, settings, allowed_document_ids=None):
        if allowed_document_ids == {1}:
            return [
                _candidate(
                    1,
                    1,
                    "紧急维修5分钟内登记，30分钟内到场。",
                    0.21,
                    "物业维修服务承诺",
                )
            ]
        return [
            _candidate(
                doc_id,
                doc_id,
                f"其他合格候选{doc_id}",
                1.0 - doc_id / 100,
                f"其他文档{doc_id}",
            )
            for doc_id in range(2, 9)
        ]

    with patch(
        "rag_retrieval._single_query_search", side_effect=fake_single_query
    ), patch("rag_retrieval.rag_embeddings.get_runtime_info", return_value={}):
        result = rag_retrieval.advanced_search(
            QUERIES[2],
            {"top_k": 5, "context_threshold": 0.2},
            allowed_document_ids=list(range(1, 9)),
            document_catalog=CATALOG,
        )
    assert len(result["results"]) == 5
    assert any(item["doc_id"] == 1 for item in result["results"]), result
    assert result["filter_summary"]["named_document_coverage_count"] == 1
    assert result["settings"]["top_k"] == 5
    assert result["settings"]["context_threshold"] == 0.2


def test_title_match_alone_cannot_admit_irrelevant_content():
    irrelevant = _candidate(
        1,
        3,
        "本文件由物业服务中心负责解释。",
        10.0,
        "物业维修服务承诺",
    )

    with patch("rag_retrieval._keyword_search", return_value=[irrelevant]), patch(
        "rag_retrieval._semantic_search", return_value=[]
    ), patch("rag_retrieval.rag_embeddings.get_runtime_info", return_value={}):
        result = rag_retrieval.advanced_search(
            "请依据《物业维修服务承诺》说明文档未包含的外星电梯折扣。",
            {"top_k": 5, "context_threshold": 0.2},
            allowed_document_ids=[1, 2],
            document_catalog=CATALOG,
        )
    assert result["results"] == [], result


def test_generic_named_document_contract_is_not_property_question_specific():
    catalog = [
        {
            "knowledge_doc_id": 42,
            "title": "月球温室值守公约",
            "document_version": "v1",
            "document_hash": "moon-hash",
        }
    ]
    plan = rag_retrieval.build_retrieval_plan(
        "同时查询遥测状态；依据《月球温室值守公约》说明舱门巡检时限。",
        document_catalog=catalog,
        allowed_document_ids={42},
    )
    named = next(item for item in plan if item["named_document_scope"])
    assert named["allowed_document_ids"] == [42]
    assert named["subquery"] == "舱门巡检时限"


def main():
    tests = [
        test_clause_plan_is_order_invariant_and_snapshot_scoped,
        test_unbound_named_document_never_expands_to_global_scope,
        test_real_content_gate_recalls_target_for_all_composite_variants,
        test_named_document_coverage_survives_higher_scoring_other_clauses,
        test_title_match_alone_cannot_admit_irrelevant_content,
        test_generic_named_document_contract_is_not_property_question_specific,
    ]
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
        print(f"S10-I RAG evidence convergence passed: {len(tests)}/{len(tests)}")
    finally:
        _TEMP_DATA.cleanup()


if __name__ == "__main__":
    main()
