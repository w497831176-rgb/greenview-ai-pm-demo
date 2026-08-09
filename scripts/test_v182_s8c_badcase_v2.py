"""S8-C deterministic contract checks. No Provider/model call is permitted."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


TEMP_DIR = tempfile.TemporaryDirectory(prefix="yiai-s8c-")
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name

from app.badcase_schema import _enrich_badcase, user_status_label
from app.runtime.badcase_capture import runtime_badcase_decision
from app.runtime.citation_renderer import build_skill_evidence, render_citations
from app.runtime.contracts import EvidenceItem, EvidenceSet, RunEvidenceLedger
from db import property_db


checks: list[str] = []


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    checks.append(name)


def ledger(**fields):
    value = RunEvidenceLedger(
        trace_id=fields.pop("trace_id", "trace-test"),
        session_id="session-test",
        config_snapshot={"snapshot_id": "snapshot-test"},
    )
    for key, item in fields.items():
        setattr(value, key, item)
    return value


def violation(code: str):
    return [{"code": code, "detail": code}]


def decide(value, **context):
    return runtime_badcase_decision(value, delivery_context=context or {"normal_completed": True})


def evidence(content: str, query: str = "") -> EvidenceSet:
    return EvidenceSet(
        query=query,
        retrieval_status="success",
        items=[
            EvidenceItem(
                evidence_id="ev_1",
                knowledge_id="knowledge-1",
                knowledge_version="1",
                document_id="doc-1",
                document_version="1",
                document_hash="hash-doc",
                chunk_id="chunk-1",
                chunk_index=0,
                chunk_hash="hash-chunk",
                content_snapshot=content,
                retrieval_mode="vector",
                title="测试依据",
            )
        ],
    )


# Five formal capture classes.
check("01 provider failure is formal", runtime_badcase_decision(ledger(), runtime_error="down", runtime_error_type="provider_failure")["disposition"] == "formal_badcase")
tool_fail = {"server_name": "workorder", "tool_name": "get", "invocation_status": "failed", "transport_status": "failed", "business_status": "failed", "required": True}
check("02 required tool failure is formal", decide(ledger(tool_invocations=[tool_fail]))["disposition"] == "formal_badcase")
check("03 unsupported final value is formal", decide(ledger(contract_violations=violation("ungrounded_critical_value")))["disposition"] == "formal_badcase")
check("04 registered capability failure is formal", decide(ledger(contract_violations=violation("skill_selected_not_loaded")))["disposition"] == "formal_badcase")
check("05 risky action is formal", decide(ledger(contract_violations=violation("false_success_without_receipt")))["disposition"] == "formal_badcase")
fixed_eval = {"evaluation_run_id": 9, "assertion_id": "agent", "passed": False, "status": "failed"}
check("06 fixed evaluation failure is formal", decide(ledger(evaluation_results=[fixed_eval]))["disposition"] == "formal_badcase")

# Observation/ignore boundary.
check("07 answer wording flag cannot suppress structured citation failure", runtime_badcase_decision(ledger(contract_violations=violation("ungrounded_critical_value")), delivery_context={"safe_rejection": True})["disposition"] == "formal_badcase")
check("08 renderer interception is structured citation failure", runtime_badcase_decision(ledger(contract_violations=violation("citation_claim_mismatch")), delivery_context={"renderer_intercepted": True})["disposition"] == "formal_badcase")
check("09 healthy skip creates nothing", decide(ledger())["disposition"] == "none")
empty_tool = {"invocation_status": "success", "transport_status": "success", "business_status": "not_found"}
check("10 normal not-found creates nothing", decide(ledger(tool_invocations=[empty_tool]))["disposition"] == "none")
check("11 cost-only issue is observation", decide(ledger(contract_violations=violation("cost_usage_unavailable")))["disposition"] == "system_observation")
check("12 latency-only issue is observation", decide(ledger(contract_violations=violation("latency_observability")))["disposition"] == "system_observation")

# Legal evidence: user, Skill, RAG, Tool and Receipt.
rendered, citations, issues = render_citations("需要等待3天", EvidenceSet(query="需要等待3天"))
check("13 user-provided value is legal evidence", not any(i.get("code") == "ungrounded_critical_value" for i in issues))
skill_source = {"skill_id": 17, "name": "宠物托管", "version": "1.0.1", "snapshot_id": "snap-26", "content_hash": "skill-hash", "content_snapshot": "提供24小时托管，收费100元/天，电话077512345678。"}
answer = "可以24小时托管，收费100元/天，电话077512345678。"
rendered, citations, issues = render_citations(answer, EvidenceSet(query="宠物托管多少钱？"), skill_sources=[skill_source])
check("14 activated Skill grounds critical values", not any(i.get("code") == "ungrounded_critical_value" for i in issues))
skill_items = build_skill_evidence(answer, [skill_source])
check("15 Skill evidence identity is complete", bool(skill_items) and all(skill_items[0].get(k) for k in ("skill_id", "name", "version", "snapshot_id", "content_hash")))
check("16 Skill evidence keeps supporting excerpt", "100元" in skill_items[0]["supporting_excerpt"] and "24小时" in skill_items[0]["supporting_excerpt"] and "077512345678" in skill_items[0]["supporting_excerpt"])
rendered, citations, issues = render_citations("收费100元【引用1】", evidence("收费100元。"))
check("17 adopted RAG evidence grounds value", bool(citations) and not any(i.get("code") == "ungrounded_critical_value" for i in issues))
tool_ok = {"invocation_status": "success", "business_status": "success", "result_summary": "{'result': 100}"}
rendered, citations, issues = render_citations("查询结果是100元", EvidenceSet(), tool_invocations=[tool_ok])
check("18 successful Tool result grounds value", not any(i.get("code") == "ungrounded_critical_value" for i in issues))
receipt = {"status": "committed", "result": {"fee": "100元"}}
rendered, citations, issues = render_citations("正式回执费用100元", EvidenceSet(), action_receipts=[receipt])
check("19 committed Receipt grounds value", not any(i.get("code") == "ungrounded_critical_value" for i in issues))

# Human/AI semantics and four user statuses.
flash_case = _enrich_badcase({"id": 1, "status": "classified", "source": "auto", "category": "other", "root_cause": "Flash suggestion", "actions": []})
check("20 Flash root cause is AI suggestion", flash_case["presentation"]["sections"][2]["text"].startswith("AI分析建议"))
darwin_case = _enrich_badcase({"id": 2, "status": "fixing", "source": "auto", "category": "other", "root_cause": "expert", "darwin_analysis": "{}", "actions": []})
check("21 Darwin root cause is expert suggestion", darwin_case["presentation"]["sections"][2]["text"].startswith("AI专家建议"))
human_case = _enrich_badcase({"id": 3, "status": "rejected", "source": "auto", "category": "other", "root_cause": "old AI", "actions": [{"action_type": "mark-auto-false-positive", "action_detail": '{"reason":"人工复核为误抓"}', "created_by": "operator", "created_at": "2026-07-28"}]})
check("22 explicit operator review is human confirmation", human_case["presentation"]["sections"][2]["text"].startswith("人工确认"))
check("23 false positive is terminal outcome", human_case["terminal_outcome_label"] == "自动误抓")
check("23a false positive is a history record", human_case["record_layer_label"] == "历史记录")
check("23b Flash suggestion remains visibly separated", "AI分析建议（历史保留）" in human_case["presentation"]["sections"][2]["text"])
check("24 four user statuses", [user_status_label(s) for s in ("pending", "fixing", "verifying", "closed")] == ["待审核", "处理中", "待验证", "已结束"])
check("25 six-section detail contract", [s["title"] for s in human_case["presentation"]["sections"]] == ["怎么发现", "问题分类", "确认原因", "处理建议", "实际行动及理由", "最终结果"])
check("26 six-node business timeline", len(human_case["presentation"]["business_timeline"]) == 6)

# Isolated list scope: historical evidence-insufficient and explicit observations
# are removed from the default current view without deleting records.
property_db.init_db()
history_case = property_db.create_badcase("old", "old", source="auto", trace_id="missing-ledger")
current_case = property_db.create_badcase("feedback", "feedback", source="user_feedback", trace_id="feedback-trace")
observation_case = property_db.create_badcase("observation", "observation", source="evaluation", trace_id="observation-trace")
property_db.add_badcase_action(observation_case["id"], "system-observation", '{"reason":"intercepted"}', "pending", "pending", "operator")
current_ids = {item["id"] for item in property_db.list_badcases_page(view_scope="current", page_size=50)["items"]}
history_ids = {item["id"] for item in property_db.list_badcases_page(view_scope="history", page_size=50)["items"]}
check("27 current view excludes old evidence-insufficient record", history_case["id"] not in current_ids and current_case["id"] in current_ids)
check("28 history view retains old record", history_case["id"] in history_ids)
check("29 system observation moves to history", observation_case["id"] in history_ids and observation_case["id"] not in current_ids)
check("30 default pagination remains bounded", property_db.list_badcases_page(view_scope="all")["page_size"] == 20)

# Static UI contract checks do not launch a browser.
html = Path("frontend/index.html").read_text(encoding="utf-8")
check("31 Skill loader is filtered from ordinary tools", "!isSkillLoader(call)" in html)
check("32 Skill evidence has business wording", "业务规则依据：" in html and "已加载业务规则：" in html)
check("33 technical trace wording is honest", "技术追踪 ·" in html and "Trace · 未知" not in html)
check("34 current/history tabs exist", "badcase-current-tab" in html and "badcase-history-tab" in html)
check(
    "35 technical evidence is collapsed",
    "高级证据：草稿、Trace、Release、历次复测与审计" in html
    and "原始技术证据" in html,
)

print(f"PASS: {len(checks)} deterministic S8-C checks")
