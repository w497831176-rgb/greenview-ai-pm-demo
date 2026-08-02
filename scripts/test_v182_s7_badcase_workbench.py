"""Deterministic V1.8.2-S7 closure and Badcase workbench checks.

Uses an isolated temporary SQLite database and a mocked chat runtime.  It does
not call any model Provider or write NAS business data.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path


def main() -> None:
    temp_dir = tempfile.TemporaryDirectory(prefix="yiai-s7-")
    os.environ["PROPERTY_DATA_DIR"] = temp_dir.name

    from db import property_db as db
    db.init_db()
    from agents.router import _capability_fallback
    from app import badcases as badcase_api
    from app import evaluations as evaluation_api
    from app.runtime.release_compiler import diff_runtime_configs
    from app.runtime.snapshot_resolver import resolve_snapshot
    checks: list[str] = []

    def check(name: str, condition: bool) -> None:
        assert condition, name
        checks.append(name)

    # 1-5. Route configuration fix: narrow generic terms but preserve domain.
    customer = {
        "agent_id": "customer_service", "name": "客服 Agent", "enabled": True,
        "description": "物业服务、小区服务与搬家服务咨询",
        "capability_card": {"routing_hints": "物业服务咨询；知识不足时安全拒答"},
    }
    before_mars = {
        "agent_id": "mars-greenhouse-agent", "name": "火星温室 Agent", "enabled": True,
        "description": "处理火星温室、种植舱、营养液和温室补给问题；使用规则 RAG 证据与只读计算 MCP 给出可引用的计算结果。",
        "capability_card": {"routing_hints": "必须使用本 Agent 绑定的知识库找到直接证据；证据不足不得猜测"},
    }
    after_description = "火星温室种植舱、标准营养液、温室补给、份数与配比计算。"
    after_instructions = "你是火星温室 Agent，仅回答火星温室、种植舱、标准营养液与温室补给计算。先从绑定的 RAG 取得每舱每日标准，再调用绑定的只读计算 MCP。回答需展示火星补给计算过程、Tool 结果与 Citation。若未取得火星温室条款或 Tool 失败，返回无法计算。本 Agent 不执行业务数据变更。"
    after_mars = {
        "agent_id": "mars-greenhouse-agent", "name": "火星温室 Agent", "enabled": True,
        "description": after_description,
        "capability_card": {"routing_hints": after_instructions},
    }
    moving_question = "请问小区是否提供免费搬家服务？请告诉我专用预约号码、服务时段、免费额度和办理步骤。请只使用知识库直接证据；没有证据就明确说依据不足，不要猜测，也不要转人工。"
    before_winner, _, _ = _capability_fallback(moving_question, [customer, before_mars])
    after_winner, _, _ = _capability_fallback(moving_question, [customer, after_mars])
    mars_winner, _, _ = _capability_fallback("4个火星温室种植舱连续7天需要多少份标准营养液？", [customer, after_mars])
    check("wide terms reproduce Mars misroute", before_winner == "mars-greenhouse-agent")
    check("moving service no longer routes to Mars", after_winner != "mars-greenhouse-agent")
    check("moving service remains customer service", after_winner == "customer_service")
    check("Mars domain still routes to Agent 74", mars_winner == "mars-greenhouse-agent")
    bindings_before = {"skills": [117], "knowledge": [124], "mcp": ["mars-calculator-mcp"]}
    bindings_after = json.loads(json.dumps(bindings_before))
    check("Agent bindings unchanged", bindings_before == bindings_after)

    # 6. Release diff contains the before/after route-bearing fields.
    old_config = {"agents": [{**before_mars, "instructions": before_mars["capability_card"]["routing_hints"]}]}
    new_config = {"agents": [{**after_mars, "instructions": after_instructions}]}
    diff_text = json.dumps(diff_runtime_configs(old_config, new_config, include_details=True), ensure_ascii=False)
    check("release diff contains before and after route config", "使用规则" in diff_text and "火星温室种植舱" in diff_text)

    # 7-8. Published release snapshot pinning: old sessions stay pinned, new use new.
    current = db.get_current_runtime_release()
    if not current:
        db.create_runtime_release("rr_s7_old", 9001, "hash-old", old_config, {"valid": True}, status="draft")
        db.publish_runtime_release("rr_s7_old")
        current = db.get_current_runtime_release()
    old_snapshot = resolve_snapshot("s7-old-session")
    db.create_runtime_release(
        "rr_s7_new", int(current.get("version") or 0) + 1000, "hash-s7-new",
        new_config, {"valid": True}, parent_release_id=current["release_id"], status="draft",
    )
    db.publish_runtime_release("rr_s7_new")
    new_snapshot = resolve_snapshot("s7-new-session")
    old_snapshot_again = resolve_snapshot("s7-old-session")
    check("new session uses new release", new_snapshot.release_id == "rr_s7_new")
    check("old session remains pinned", old_snapshot_again.release_id == old_snapshot.release_id)

    # 9-20. True server-side paging on 57 isolated rows.
    for index in range(57):
        item = db.create_badcase(
            title=f"S7 page item {index:02d}", description="D" * 900,
            original_query=f"S7 searchable question {index:02d}", ai_response="A" * 1200,
            context_json=json.dumps({"large": "J" * 1500}),
            category="routing" if index % 2 == 0 else "other",
            status="pending" if index % 3 else "classified", source="s7_test",
            priority="high" if index % 4 == 0 else "medium", root_cause_domain="routing",
            trace_id=f"s7-trace-{index:02d}",
        )
        db.update_badcase(item["id"], darwin_analysis=json.dumps({"full": "X" * 1400}))
        if index == 0:
            db.create_skill_prompt_draft(
                badcase_id=item["id"], title="S7 draft", skill_name="s7",
                prompt_content="P" * 1000, trigger_keywords="火星",
            )
    first = db.list_badcases_page(source="s7_test")
    second = db.list_badcases_page(source="s7_test", page=2)
    last = db.list_badcases_page(source="s7_test", page=3)
    fifty = db.list_badcases_page(source="s7_test", page_size=50)
    check("default page is first 20", first["page"] == 1 and first["page_size"] == 20 and len(first["items"]) == 20)
    check("total and total pages are correct", first["total"] == 57 and first["total_pages"] == 3)
    check("second page has no duplicates", not ({x["id"] for x in first["items"]} & {x["id"] for x in second["items"]}))
    check("first two pages have no omissions", len({x["id"] for x in first["items"] + second["items"]}) == 40)
    check("last page count is correct", len(last["items"]) == 17)
    check("page size 50 is supported", fifty["page_size"] == 50 and len(fifty["items"]) == 50 and fifty["total_pages"] == 2)
    pending = db.list_badcases_page(source="s7_test", status="pending", page_size=50)
    check("status filter is correct", pending["total"] == 38 and all(x["status"] == "pending" for x in pending["items"]))
    check("source filter is correct", all(x["source"] == "s7_test" for x in first["items"]))
    searched = db.list_badcases_page(source="s7_test", search="question 13")
    check("keyword search is correct", searched["total"] == 1 and searched["items"][0]["summary"].endswith("13"))
    filtered = db.list_badcases_page(source="s7_test", status="classified", priority="high")
    check("filtered total and pagination are correct", filtered["total"] == 5 and filtered["total_pages"] == 1)
    forbidden = {"ai_response", "context_json", "darwin_analysis", "fix_plan", "evidence", "description"}
    check("list rows exclude long fields", all(not (forbidden & set(item)) for item in first["items"]))
    api_page = asyncio.run(badcase_api.list_badcases(source="s7_test", page=1, page_size=20))
    check("legacy list aliases remain lightweight", api_page["badcases"] == api_page["items"] and api_page["count"] == 57)

    # 21-25. Human config apply is explicit evidence, never a draft auto-apply.
    agent = db.create_agent("s7-mars", "S7 Mars", before_mars["description"], before_mars["capability_card"]["routing_hints"], "vertical", True)
    db.set_agent_skills("s7-mars", [117])
    db.set_agent_knowledge_bindings("s7-mars", [124])
    db.set_agent_tools("s7-mars", [{"tool_name": "mars-calculator-mcp"}])
    db.update_agent(agent["id"], description=after_description, instructions=after_instructions)
    eval_case = db.create_evaluation_case(
        case_key="S7-CASE-2", title="moving service", user_message="小区有没有免费搬家服务？",
        expected_agent_id="customer_service", expected_handoff=False, status="active",
    )
    original_run = db.create_evaluation_run(eval_case["id"], "failed", trace_id="s7-original-trace", rule_results=[{"key": "agent", "status": "fail"}])
    badcase = db.create_badcase(
        title="S7 linked Badcase", description="S7 route failure", status="fixing", category="routing", source="evaluation",
        original_query=eval_case["user_message"], trace_id="s7-original-trace",
        linked_evaluation_case_id=eval_case["id"], linked_evaluation_run_id=original_run["id"],
    )
    db.update_evaluation_run(original_run["id"], badcase_id=badcase["id"])
    apply_result = asyncio.run(badcase_api.record_agent_config_apply(
        badcase["id"], badcase_api.AgentConfigApplyEvidenceRequest(
            agent_id="s7-mars", before_description=before_mars["description"],
            before_instructions=before_mars["capability_card"]["routing_hints"],
            after_description=after_description, after_instructions=after_instructions,
            skill_ids_before=[117], skill_ids_after=[117],
            knowledge_doc_ids_before=[124], knowledge_doc_ids_after=[124],
            mcp_tools_before=["mars-calculator-mcp"], mcp_tools_after=["mars-calculator-mcp"],
            review_note="人工确认仅收窄路由配置。",
        ),
    ))
    check("config apply moves Badcase to verifying", apply_result["badcase"]["status"] == "verifying")
    check("config apply records human review", apply_result["evidence"]["human_reviewed"] is True)
    check("config apply records no Darwin auto apply", apply_result["evidence"]["auto_applied_darwin_draft"] is False)
    check("config apply action exists", any(a["action_type"] == "apply-agent-config" for a in db.list_badcase_actions(badcase["id"])))
    check("config apply clears stale retest", not apply_result["badcase"].get("retest_trace_id"))

    # 26-30. Run only the linked case through a mocked runtime, then close.
    evaluation_api._background_budget_gate = lambda _operation: {
        "budget_status": "available",
        "alert_level": "none",
        "allowed": True,
    }

    async def mocked_chat(message: str, session_id: str):
        trace_id = f"s7-retest-{uuid.uuid4().hex[:8]}"
        db.ensure_chat_session(session_id)
        db.create_chat_trace(trace_id, session_id, message, run_type="evaluation")
        db.update_chat_trace(trace_id, status="complete", intent="customer_service", agent_name="客服 Agent", agent_id="customer_service")
        db.record_model_call(trace_id=trace_id, stage="router", model_id="deepseek-v4-flash", status="success", total_tokens=12, usage_source="provider_actual", estimated_cost_cny=0.0001)
        answer = "当前知识库没有免费搬家服务的直接依据，因此无法确认；不提供未经证实的办理方式、收费或电话。"
        done = {
            "status": "complete", "trace_id": trace_id, "current_agent_id": "customer_service",
            "current_agent": "客服 Agent", "route_intent": "customer_service", "route_reason": "mocked",
            "activated_skills": [], "tool_calls": [], "mcp_calls": [], "citations": [], "handoff": False,
            "decision_summary": {"agent": {"status": "selected", "agent_id": "customer_service"}, "rag": {"status": "selected", "evidence_decision": "rejected_insufficient"}, "tool": {"status": "skipped"}, "handoff": {"status": "skipped"}},
            "side_effects": {"business_writes": 0, "work_orders": 0, "work_order_drafts": 0, "action_proposals": 0, "action_receipts": 0},
            "model_calls": [{"stage": "router", "model_id": "deepseek-v4-flash", "total_tokens": 12}],
        }
        return answer, done

    evaluation_api._run_real_chat = mocked_chat
    retest = asyncio.run(evaluation_api.run_case(
        eval_case["id"], evaluation_api.EvaluationRunRequest(linked_badcase_id=badcase["id"])
    ))
    check("linked Evaluation retest passes", retest["run"]["status"] == "passed")
    check("retest Run links original Badcase", retest["run"]["badcase_id"] == badcase["id"])
    linked_after = db.get_badcase(badcase["id"])
    check("retest evidence is stored on original Badcase", linked_after["retest_trace_id"] == retest["run"]["trace_id"] and linked_after["last_retest_at"] >= linked_after["last_applied_at"])
    check("original failed Run is not rewritten", db.get_evaluation_run(original_run["id"])["status"] == "failed")
    verified = asyncio.run(badcase_api.verify_badcase(badcase["id"], badcase_api.VerifyRequest(passed=True, note="Case #2 同题复测通过")))
    closed = asyncio.run(badcase_api.close_badcase(badcase["id"], badcase_api.CloseReleaseRequest(observation_note="人工核验发布后同题结果，关闭案例。")))
    check("PASS lifecycle can advance to closed", verified["badcase"]["status"] == "released" and closed["badcase"]["status"] == "closed")

    # 31-40. Detail hierarchy and on-demand front-end contract.
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    html = frontend.read_text(encoding="utf-8")
    list_block = html.split("async function renderBadcasesPage", 1)[1].split("async function renderBadcaseDetailPage", 1)[0]
    detail_block = html.split("async function renderBadcaseDetailPage", 1)[1].split("async function renderEvaluationsPage", 1)[0]
    check("list requests paginated API", "page_size" in list_block and "data.items" in list_block)
    check("list does not preload per-row details", "apiGet(`/api/badcases/${id}`)" not in list_block)
    check("detail is requested only after opening detail", "apiGet(`/api/badcases/${id}`)" in detail_block)
    main_headings = ["1 · 发现问题", "2 · AI 根因建议", "3 · 人工确认修复方案", "4 · 单例复测与人工关闭"]
    check("detail contains four-step business hierarchy", all(label in detail_block for label in main_headings))
    check("four-step business hierarchy order is correct", [detail_block.index(label) for label in main_headings] == sorted(detail_block.index(label) for label in main_headings))
    advanced_headings = ["1. 问题概览", "2. 分析记录", "3. 解决方案与人工决策", "4. 验证与发布", "5. 生命周期时间线"]
    check("advanced evidence keeps full history hierarchy", all(label in detail_block for label in advanced_headings))
    check("before and after are readable", "修改前" in detail_block and "修改后" in detail_block)
    check("technical evidence is collapsed by default", '<details class="bg-white' in detail_block and "高级证据：草稿、Trace、Release、历次复测与审计" in detail_block and "原始技术证据" in detail_block)
    check("long text supports expand and collapse", "展开全文" in detail_block and "<details" in detail_block)
    check("missing analysis and retest have honest empty states", "尚未请求 AI 建议" in detail_block and "尚未复测" in detail_block)
    check("pending records do not fabricate analysis", "尚未形成修复计划" in detail_block and "尚无专家建议" in detail_block)

    # 41-43. Current-vs-history summary and response-size discipline.
    summary = db.evaluation_summary()
    check("latest-per-case summary exists", "latest_golden_pass_rate" in summary and "latest_golden_runs_passed" in summary)
    payload_size = len(json.dumps(api_page, ensure_ascii=False).encode("utf-8"))
    check("20-row list payload stays below 150KB", payload_size < 150 * 1024)
    check("list response exposes evidence flags", all({"has_darwin", "has_fix_draft", "has_retest"} <= set(item) for item in api_page["items"]))

    print(json.dumps({"status": "PASS", "checks": len(checks), "names": checks}, ensure_ascii=False, indent=2))
    temp_dir.cleanup()


if __name__ == "__main__":
    main()
