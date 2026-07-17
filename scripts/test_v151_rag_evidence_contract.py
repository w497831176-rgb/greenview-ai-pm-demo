"""No-model regression checks for the V1.5.1 RAG evidence contract."""

from unittest.mock import patch

import rag_chunking
import rag_retrieval


def item(doc_id, chunk_index, content, score, source):
    return {
        "doc_id": doc_id,
        "doc_title": "电动车充电临时规定",
        "title": "电动车充电临时规定",
        "chunk_index": chunk_index,
        "content": content,
        "score": score,
        "source": source,
        "is_indexed": True,
    }


def test_structured_chunking():
    content = "一、只能在 17 号充电区充电。\n二、楼道飞线充电严禁。\n三、充电桩故障请报修。"
    chunks = rag_chunking.split_text(content, strategy="auto", chunk_size=512)
    assert len(chunks) == 3, chunks
    assert chunks[1].startswith("二、")


def test_settings_are_normalized():
    settings = rag_retrieval.normalize_retrieval_settings(
        {"keyword_weight": 3, "semantic_weight": 7, "top_k": 99}
    )
    assert settings["top_k"] == 10
    assert settings["keyword_weight"] == 0.3
    assert settings["semantic_weight"] == 0.7
    assert settings["context_threshold"] == 0.2


def test_hybrid_provenance_and_final_evidence():
    keyword = [item(1, 0, "楼道飞线充电严禁", 8.0, "keyword")]
    semantic = [item(1, 0, "楼道飞线充电严禁", 0.9, "semantic")]
    with patch("rag_retrieval._keyword_search", return_value=keyword), patch(
        "rag_retrieval._semantic_search", return_value=semantic
    ), patch("rag_retrieval._extract_quoted_titles", return_value=[]), patch(
        "rag_retrieval._extract_sub_queries", return_value=[]
    ):
        result = rag_retrieval.advanced_search("楼道飞线充电允许吗？", {"top_k": 3})
    evidence = result["results"]
    assert len(evidence) == 1, evidence
    assert evidence[0]["chunk_index"] == 0
    assert set(evidence[0]["retrieval_sources"]) == {"keyword", "semantic"}
    assert evidence[0]["evidence_status"] == "accepted"


def main():
    test_structured_chunking()
    test_settings_are_normalized()
    test_hybrid_provenance_and_final_evidence()
    print("V1.5.1 RAG evidence contract checks passed.")


if __name__ == "__main__":
    main()
