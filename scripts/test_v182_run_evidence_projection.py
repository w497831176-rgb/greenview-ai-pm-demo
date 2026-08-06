"""Deterministic Trace projection checks; temp database, no network/Provider."""

from __future__ import annotations

import asyncio
import atexit
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Establish the disposable SQLite boundary before importing any app/db module.
_TEMP_ROOT = tempfile.TemporaryDirectory(prefix="yiai-s10i-projection-")
_TEST_DATA_DIR = (Path(_TEMP_ROOT.name) / "property-data").resolve()
_FORBIDDEN_DATA_ROOTS = (
    Path("/app/data"),
    Path("/volume3/docker/agno-demo-os"),
    ROOT,
)
if any(
    _TEST_DATA_DIR == item.resolve() or item.resolve() in _TEST_DATA_DIR.parents
    for item in _FORBIDDEN_DATA_ROOTS
):
    raise RuntimeError(f"unsafe projection test data directory: {_TEST_DATA_DIR}")
os.environ["PROPERTY_DATA_DIR"] = str(_TEST_DATA_DIR)
for _provider_key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
    os.environ.pop(_provider_key, None)
atexit.register(_TEMP_ROOT.cleanup)

from app.runtime.evidence_ledger import project_evidence_for_trace  # noqa: E402


def _candidate(index: int) -> dict:
    return {
        "evidence_id": f"ev_{index}",
        "document_id": "1",
        "chunk_id": f"doc-1-chunk-{index}",
        "chunk_index": index,
        "title": "测试文档",
        "content_snapshot": f"分片 {index}",
        "retrieval_mode": "keyword+semantic",
    }


def test_flat_ledger_retrieved_adopted_unused() -> None:
    ledger = {
        "retrieval_evidence": [_candidate(1), _candidate(2), _candidate(3)],
        "citation_links": [{"evidence_id": "ev_2"}],
    }
    projection = project_evidence_for_trace(ledger)
    assert projection["projection_source"] == "legacy_flat_ledger"
    assert projection["counts"]["retrieved_rag_candidates"] == 3
    assert [item["evidence_id"] for item in projection["adopted_rag"]] == ["ev_2"]
    assert [item["evidence_id"] for item in projection["unused_rag"]] == [
        "ev_1",
        "ev_3",
    ]


def test_safety_intercept_keeps_all_candidates_visible_as_unused() -> None:
    ledger = {
        "retrieval_evidence": [_candidate(1), _candidate(2), _candidate(3)],
        "citation_links": [],
        "contract_violations": [
            {
                "code": "unsupported_critical_value",
                "detail": "blocked",
                "metadata": {"values": ["71%"]},
            }
        ],
        "evaluation_results": [
            {
                "case": "knowledge_evidence_gate",
                "passed": True,
                "decision": "rejected_insufficient",
            }
        ],
    }
    projection = project_evidence_for_trace(ledger)
    assert projection["adopted_rag"] == []
    assert len(projection["unused_rag"]) == 3
    assert projection["withheld"][0]["delivery_status"] == (
        "withheld_before_user_delivery"
    )
    assert projection["withheld"][0]["content_available"] is False


def test_historical_structured_tool_result_explains_validator_miss() -> None:
    ledger = {
        "retrieval_evidence": [_candidate(index) for index in range(1, 6)],
        "citation_links": [],
        "tool_invocations": [
            {
                "invocation_id": "tool-weather",
                "server_name": "weather-server",
                "tool_name": "get_current_weather",
                "effect": "read",
                "transport_status": "success",
                "invocation_status": "success",
                "business_status": "success",
                "result_summary": (
                    '{"status":"success","data":{"temperature_c":29,'
                    '"humidity_pct":70}}'
                ),
            }
        ],
        "contract_violations": [
            {
                "code": "ungrounded_critical_value",
                "detail": "Model citation was not present in the immutable EvidenceSet.",
                "metadata": {"values": ["70%"]},
            }
        ],
        "evaluation_results": [
            {
                "case": "knowledge_evidence_gate",
                "decision": "rejected_insufficient",
            }
        ],
    }
    projection = project_evidence_for_trace(ledger)
    assert projection["counts"]["retrieved_rag_candidates"] == 5
    assert projection["counts"]["adopted_rag"] == 0
    assert projection["counts"]["unused_rag"] == 5
    violation = projection["violation"][0]
    assert violation["code"] == "tool_result_value_not_recognized_by_validation"
    assert violation["source_code"] == "ungrounded_critical_value"
    assert violation["tool_result_evidence"] == [
        {
            "invocation_id": "tool-weather",
            "server_name": "weather-server",
            "tool_name": "get_current_weather",
            "json_path": "$.data.humidity_pct",
            "normalized_value": "70",
            "unit": "%",
        }
    ]
    assert projection["withheld"][0]["code"] == (
        "tool_result_value_not_recognized_by_validation"
    )


def test_arguments_and_failed_results_never_rewrite_violation_reason() -> None:
    base_violation = {
        "code": "ungrounded_critical_value",
        "detail": "legacy",
        "metadata": {"values": ["70%"]},
    }
    for invocation in (
        {
            "invocation_id": "tool-argument-only",
            "effect": "read",
            "transport_status": "success",
            "invocation_status": "success",
            "business_status": "success",
            "arguments": {"humidity_pct": 70},
            "result_summary": '{"status":"success","data":{"humidity_pct":65}}',
        },
        {
            "invocation_id": "tool-failed",
            "effect": "read",
            "transport_status": "success",
            "invocation_status": "success",
            "business_status": "failed",
            "result_summary": '{"status":"failed","data":{"humidity_pct":70}}',
        },
    ):
        projection = project_evidence_for_trace(
            {
                "retrieval_evidence": [],
                "citation_links": [],
                "tool_invocations": [invocation],
                "contract_violations": [base_violation],
            }
        )
        assert projection["violation"][0]["code"] == "ungrounded_critical_value"


def test_new_bundle_is_authoritative_over_flat_legacy_lists() -> None:
    candidates = [_candidate(1), _candidate(2)]
    tool_evidence = {
        "evidence_id": "tool_ev_1",
        "invocation_id": "tool_1",
        "server_name": "server",
        "tool_name": "lookup",
        "business_status": "success",
        "payload_hash": "hash",
        "facts": [],
    }
    ledger = {
        "retrieval_evidence": [_candidate(99)],
        "citation_links": [{"evidence_id": "ev_99"}],
        "contract_violations": [
            {"code": "skill_selected_not_loaded", "detail": "skill evidence"}
        ],
        "run_evidence_bundle": {
            "retrieved_rag_candidates": {
                "items": candidates,
                "query": "q",
                "retrieval_status": "completed",
            },
            "delivered_evidence_ids": ["ev_2", "tool_ev_1"],
            "tool_evidence": [tool_evidence],
            "tool_evidence_links": [
                {
                    "evidence_id": "tool_ev_1",
                    "invocation_id": "tool_1",
                    "fact_ids": [],
                    "claim_values": ["65%"],
                }
            ],
            "withheld": [],
            "violations": [],
        },
    }
    projection = project_evidence_for_trace(ledger)
    assert projection["projection_source"] == "run_evidence_bundle"
    assert [item["evidence_id"] for item in projection["adopted_rag"]] == ["ev_2"]
    assert [item["evidence_id"] for item in projection["unused_rag"]] == ["ev_1"]
    assert projection["tool_evidence_links"][0]["tool_evidence"] == tool_evidence
    assert projection["violation"] == [
        {"code": "skill_selected_not_loaded", "detail": "skill evidence"}
    ]


def test_violation_merge_keeps_distinct_facts_and_dedupes_legacy_shape() -> None:
    first = {
        "code": "unsupported_tool_fact",
        "evidence_id": "tool_ev_1",
        "values": ["晴朗"],
        "claim_context": "天气晴朗",
    }
    second = {
        "code": "unsupported_tool_fact",
        "evidence_id": "tool_ev_2",
        "values": ["已完成"],
        "claim_context": "工单状态已完成",
    }
    ledger = {
        "run_evidence_bundle": {
            "retrieved_rag_candidates": {"items": [], "query": "q"},
            "violations": [first, second],
        },
        "contract_violations": [
            {
                "code": "unsupported_tool_fact",
                "detail": "legacy wrapper detail",
                "metadata": {
                    "evidence_id": "tool_ev_1",
                    "values": ["晴朗"],
                    "claim_context": "天气晴朗",
                },
            }
        ],
    }
    projection = project_evidence_for_trace(ledger)
    assert projection["violation"] == [first, second]

    no_identity = project_evidence_for_trace(
        {
            "run_evidence_bundle": {
                "retrieved_rag_candidates": {"items": [], "query": "q"},
                "violations": [
                    {
                        "code": "required_citation_missing",
                        "detail": "bundle detail",
                    }
                ],
            },
            "contract_violations": [
                {
                    "code": "required_citation_missing",
                    "detail": "legacy wrapper detail",
                }
            ],
        }
    )
    assert no_identity["violation"] == [
        {"code": "required_citation_missing", "detail": "bundle detail"}
    ]


def test_runtime_evidence_api_returns_projection() -> None:
    from db import property_db

    property_db.init_db()
    from app.runtime import api

    row = {
        "trace_id": "trace-1",
        "ledger": {
            "retrieval_evidence": [_candidate(1)],
            "citation_links": [],
        },
    }
    with patch.object(api, "get_evidence_ledger", return_value=row):
        response = asyncio.run(api.trace_evidence("trace-1"))
    assert response["evidence"] == row
    assert response["projection"]["counts"]["retrieved_rag_candidates"] == 1
    assert response["projection"]["counts"]["unused_rag"] == 1


def test_trace_frontend_consumes_backend_projection() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    ).read_text(encoding="utf-8")
    assert "assistantMsg.citations" not in source
    assert "runtimeEvidenceData.projection" in source
    assert "evidenceUnavailableHtml" in source
    assert "const runtimeContractHtml = evidenceProjectionAvailable ?" in source
    assert (
        "evidenceProjectionAvailable || Object.keys(runtimeEvidence).length"
        not in source
    )
    assert (
        "apiGet(`/api/runtime/traces/${traceId}/evidence`).catch(() => ({}))"
        not in source
    )


def main() -> None:
    tests = [
        test_flat_ledger_retrieved_adopted_unused,
        test_safety_intercept_keeps_all_candidates_visible_as_unused,
        test_historical_structured_tool_result_explains_validator_miss,
        test_arguments_and_failed_results_never_rewrite_violation_reason,
        test_new_bundle_is_authoritative_over_flat_legacy_lists,
        test_violation_merge_keeps_distinct_facts_and_dedupes_legacy_shape,
        test_runtime_evidence_api_returns_projection,
        test_trace_frontend_consumes_backend_projection,
    ]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} run-evidence projection checks")


if __name__ == "__main__":
    main()
