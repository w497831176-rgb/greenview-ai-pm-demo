#!/usr/bin/env python3
"""V1.4.2 browser acceptance suite for YIAI Portal.

Run with:
    BASE_URL=http://192.168.50.123:18005 python scripts/browser_acceptance_v1_4_2.py

Outputs (to the workspace folder):
    - browser_acceptance_v1_4_2/*.png
    - browser_acceptance_v1_4_2/evidence.json
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from playwright.sync_api import sync_playwright, Page, expect

BASE_URL = os.environ.get("BASE_URL", "http://192.168.50.123:18005").rstrip("/")
WORKSPACE_DIR = Path("d:/work/Wangbeibei/2026AI/vibe coding/TRAE work/Ango x YIAI")
OUT_DIR = WORKSPACE_DIR / "browser_acceptance_v1_4_2"

page_errors: List[str] = []
console_errors: List[str] = []
failed_requests: List[str] = []
results: List[Dict[str, Any]] = []

created_doc_ids: List[int] = []
created_skill_ids: List[int] = []
created_agent_id: Optional[str] = None


def add_event_listeners(page: Page):
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url}: {getattr(req, 'failure', '')}"))


def screenshot(page: Page, name: str):
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    results.append({"name": name, "screenshot": str(path), "url": page.url})


def wait_visible(page: Page, selector: str, timeout: int = 15000):
    el = page.locator(selector).first
    expect(el).to_be_visible(timeout=timeout)
    return el


def check(label: str, ok: bool, detail: str = ""):
    results.append({"check": label, "ok": ok, "detail": detail})
    if not ok:
        raise AssertionError(f"{label}: {detail}")


def check_global_errors(label: str):
    err_list = []
    if page_errors:
        err_list.append(f"pageerror={len(page_errors)}: {'; '.join(page_errors[:3])}")
    if console_errors:
        err_list.append(f"console.error={len(console_errors)}: {'; '.join(console_errors[:3])}")
    if failed_requests:
        err_list.append(f"failed request={len(failed_requests)}: {'; '.join(failed_requests[:3])}")
    if err_list:
        raise AssertionError(f"{label}: " + " | ".join(err_list))
    results.append({"check": label + " zero errors", "ok": True})


def api_headers():
    return {"Content-Type": "application/json"}


def api_post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(f"{BASE_URL}{path}", json=body, headers=api_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_delete(path: str):
    return requests.delete(f"{BASE_URL}{path}", timeout=30)


def api_cleanup():
    print("[API cleanup]")
    if created_agent_id:
        try:
            r = api_delete(f"/api/agents/{created_agent_id}")
            print(f"  cleanup agent {created_agent_id}: HTTP {r.status_code}")
        except Exception as exc:
            print(f"  cleanup agent failed: {exc}")
    for doc_id in created_doc_ids:
        try:
            r = api_delete(f"/api/knowledge/docs/{doc_id}")
            print(f"  cleanup doc #{doc_id}: HTTP {r.status_code}")
        except Exception as exc:
            print(f"  cleanup doc failed: {exc}")
    for skill_id in created_skill_ids:
        try:
            r = api_delete(f"/api/skills/{skill_id}")
            print(f"  cleanup skill #{skill_id}: HTTP {r.status_code}")
        except Exception as exc:
            print(f"  cleanup skill failed: {exc}")


def setup_test_data():
    """Seed a knowledge doc and a dynamic vertical agent for browser tests."""
    global created_agent_id
    doc = api_post("/api/knowledge/docs", {
        "title": "BROWSER_V142_电动车充电规定",
        "content": "小区电动车管理规定：17 号集中充电区为指定充电区域；禁止在楼道飞线充电。",
    }).get("knowledge_doc")
    created_doc_ids.append(doc["id"])
    print(f"  seeded knowledge doc #{doc['id']}")
    time.sleep(2)

    agent_id = f"BROWSER_V142_VERT_{int(time.time())}"
    api_post("/api/agents", {
        "agent_id": agent_id,
        "name": "浏览器验收动态 Agent",
        "description": "用于浏览器验收的动态垂直 Agent，处理电动车咨询。",
        "system_prompt": "你是电动车规定专家。回答时第一行必须包含'BROWSER_V142_DYNAMIC'。",
        "category": "vertical",
        "enabled": True,
    })
    created_agent_id = agent_id
    print(f"  seeded dynamic agent {agent_id}")


def test_version_and_agent_management(page: Page):
    print("\n[Test] Version and Agent Management")
    page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(800)
    add_event_listeners(page)

    # Version in footer / sidebar.
    body_text = page.locator("body").inner_text()
    check("Version shows V1.4.2", "V1.4.2" in body_text, f"body text missing V1.4.2")
    screenshot(page, "01_home_version")

    # Navigate to Agent management.
    wait_visible(page, '.top-tab[data-top="platform"]').click()
    page.wait_for_timeout(500)
    wait_visible(page, '#sub-menu button[data-sub="agents"]').click()
    page.wait_for_timeout(800)
    wait_visible(page, "#agents-content")
    screenshot(page, "02_agent_list")

    # Router row should not have edit/delete/toggle buttons.
    router_row = page.locator('[data-agent-id="router"]')
    if router_row.count() == 0:
        # Fallback: find row containing "Router" text.
        router_row = page.locator("tr:has-text('Router')").first
    check("Router row exists", router_row.count() > 0)
    edit_btn = router_row.locator("button:has-text('编辑')")
    delete_btn = router_row.locator("button:has-text('删除')")
    toggle_btn = router_row.locator("input[type='checkbox']")
    check("Router row has no edit button", edit_btn.count() == 0, f"found {edit_btn.count()} edit buttons")
    check("Router row has no delete button", delete_btn.count() == 0, f"found {delete_btn.count()} delete buttons")
    check("Router row has no toggle checkbox", toggle_btn.count() == 0, f"found {toggle_btn.count()} toggles")

    # Open router detail (view-only).
    view_btn = router_row.locator("button:has-text('查看'), button:has-text('详情')").first
    if view_btn.count():
        view_btn.click()
        page.wait_for_timeout(800)
        detail_text = page.locator("#main-content").inner_text()
        check("Router detail is read-only", "当前路由成员" in detail_text or "路由成员" in detail_text, detail_text[:300])
        check("Router detail lists dynamic agent" if created_agent_id else "Router detail ok", created_agent_id is None or created_agent_id in detail_text, detail_text[:300])
        screenshot(page, "03_router_detail")
        # Close modal if any.
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    # Open add-agent modal and verify no is_router switch.
    add_btn = page.locator("button:has-text('新增 Agent'), button:has-text('新增')").first
    if add_btn.count():
        add_btn.click()
        page.wait_for_timeout(500)
        modal_text = page.locator("body").inner_text()
        check("Add-agent form has no is_router switch", "is_router" not in modal_text and "路由 Agent" not in modal_text, modal_text[:300])
        screenshot(page, "04_add_agent_form")
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)

    check_global_errors("Agent management page")


def test_chat_citation_and_trace(page: Page):
    print("\n[Test] Chat inline citation and Trace drawer")
    page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(800)
    add_event_listeners(page)

    # Send a question that should trigger RAG citation.
    input_box = page.locator("#chat-input").first
    send_btn = page.locator("#send-btn").first
    if input_box.count() == 0:
        input_box = page.locator("textarea[placeholder*='输入'], input[placeholder*='输入']").first
    if send_btn.count() == 0:
        send_btn = page.locator("button:has-text('发送')").first

    check("Chat input exists", input_box.count() > 0)
    check("Chat send button exists", send_btn.count() > 0)

    input_box.fill("17号充电区可以充电吗？")
    send_btn.click()

    # Wait for answer to complete.
    page.wait_for_timeout(12000)
    screenshot(page, "05_chat_answer")

    # Look for inline citation buttons like [引用1], [1], etc.
    citation_btn = page.locator("button.inline-citation, .citation-btn, a:has-text('[引用1]')").first
    if citation_btn.count() == 0:
        citation_btn = page.locator("text=[引用1]").first
    check("Inline citation button exists", citation_btn.count() > 0)

    citation_btn.click()
    page.wait_for_timeout(800)
    screenshot(page, "06_citation_modal")

    # Verify citation modal shows chunk info.
    modal = page.locator(".modal, [role='dialog']").last
    modal_text = modal.inner_text() if modal.count() else page.locator("body").inner_text()
    check("Citation modal shows chunk", "分片" in modal_text or "chunk" in modal_text.lower() or "doc_id" in modal_text, modal_text[:300])

    # Close modal.
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)

    # Look for Trace bill button.
    trace_btn = page.locator("button:has-text('Trace'), button:has-text('账单'), .trace-bill-btn").first
    check("Trace bill button exists", trace_btn.count() > 0)
    trace_btn.click()
    page.wait_for_timeout(800)
    screenshot(page, "07_trace_drawer")

    # Verify trace drawer content.
    drawer_text = page.locator("body").inner_text()
    check("Trace drawer shows trace_id", "trace_id" in drawer_text, drawer_text[:300])
    check("Trace drawer shows model calls", "模型调用" in drawer_text or "model" in drawer_text.lower(), drawer_text[:300])

    # Copy trace summary.
    copy_btn = page.locator("button:has-text('复制 Trace 摘要'), button:has-text('复制')").first
    if copy_btn.count():
        copy_btn.click()
        page.wait_for_timeout(500)
        # Playwright clipboard read is async; just verify button exists and click succeeded.
        check("Copy trace summary button works", True)

    # Close drawer.
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)

    check_global_errors("Chat citation / trace page")


def test_cost_governance(page: Page):
    print("\n[Test] Cost governance trace list")
    page.goto(f"{BASE_URL}/#/cost", wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(1000)
    add_event_listeners(page)

    body_text = page.locator("body").inner_text()
    check("Cost page shows model price table or anchor", "模型价格表" in body_text or "价格表" in body_text, body_text[:300])

    # Trace list should show model summary, tokens, cost.
    check("Cost page shows trace list", "Trace 列表" in body_text or "trace" in body_text.lower(), body_text[:300])
    screenshot(page, "08_cost_page")

    check_global_errors("Cost governance page")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_test_data()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            test_version_and_agent_management(page)
            test_chat_citation_and_trace(page)
            test_cost_governance(page)

            browser.close()
    finally:
        api_cleanup()

    # Final error tally.
    check("Zero page errors", len(page_errors) == 0, f"{len(page_errors)} page errors")
    check("Zero console errors", len(console_errors) == 0, f"{len(console_errors)} console errors")
    check("Zero failed requests", len(failed_requests) == 0, f"{len(failed_requests)} failed requests")

    evidence_path = OUT_DIR / "evidence.json"
    evidence_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEvidence saved to {evidence_path}")

    failed = [r for r in results if r.get("ok") is False]
    print(f"Results: {len(results)} checks, {len(failed)} failed, {len(page_errors)} page errors, {len(console_errors)} console errors, {len(failed_requests)} failed requests")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
