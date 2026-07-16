#!/usr/bin/env python3
"""V1.4 browser acceptance suite for YIAI Portal.

Run with:
    BASE_URL=http://192.168.50.123:18005 python scripts/browser_acceptance_v1_4.py

Outputs (to the workspace folder):
    - browser_acceptance_v1_4/*.png
    - browser_acceptance_v1_4/evidence.json
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from playwright.sync_api import sync_playwright, Page, expect

BASE_URL = os.environ.get("BASE_URL", "http://192.168.50.123:18005").rstrip("/")
WORKSPACE_DIR = Path("d:/work/Wangbeibei/2026AI/vibe coding/TRAE work/Ango x YIAI")
OUT_DIR = WORKSPACE_DIR / "browser_acceptance_v1_4"

page_errors: List[str] = []
console_errors: List[str] = []
failed_requests: List[str] = []
results: List[Dict[str, Any]] = []
created_badcase_ids: List[int] = []


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


def assert_text(page: Page, substring: str, label: str):
    content = page.locator("#main-content").first.inner_text()
    ok = substring in content
    assert ok, f"{label}: expected text '{substring}' not found in #main-content"
    results.append({"check": label, "ok": True, "text": substring})


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


def api_create_badcase(title: str, category: str = "pending") -> int:
    resp = requests.post(
        f"{BASE_URL}/api/badcases",
        json={
            "title": title,
            "description": "Browser acceptance test badcase",
            "category": category,
            "source": "manual",
            "original_query": "浏览器验收测试查询",
            "ai_response": "浏览器验收测试回答",
        },
        timeout=20,
    )
    resp.raise_for_status()
    case_id = resp.json()["badcase"]["id"]
    created_badcase_ids.append(case_id)
    print(f"  created badcase #{case_id} via API")
    return case_id


def api_cleanup():
    for case_id in created_badcase_ids:
        try:
            r = requests.delete(f"{BASE_URL}/api/badcases/{case_id}", timeout=10)
            print(f"  cleanup badcase #{case_id}: HTTP {r.status_code}")
        except Exception as exc:
            print(f"  cleanup failed for badcase #{case_id}: {exc}")


def run_platform_badcase(page: Page):
    wait_visible(page, '.top-tab[data-top="platform"]').click()
    page.wait_for_timeout(800)
    wait_visible(page, "#main-content")
    assert_text(page, "Agent 管理", "platform main content loaded")
    screenshot(page, "01_platform_overview")

    # Badcase library
    wait_visible(page, '#sub-menu button[data-sub="badcases"]').click()
    page.wait_for_timeout(1000)
    wait_visible(page, "#badcases-content")
    screenshot(page, "02_badcases_list")

    # Create a pending badcase via API, then filter by manual source and view it
    ts = str(int(time.time()))
    case_id = api_create_badcase(f"DEMO_TEST_V140_BROWSER_{ts}", category="pending")

    page.locator("#badcase-filter-source").select_option("manual")
    page.locator("#badcase-filter-btn").click()
    page.wait_for_timeout(1000)
    wait_visible(page, f"button[data-id='{case_id}']")
    screenshot(page, "03_badcases_filtered")

    page.locator(f"button[data-id='{case_id}']").click()
    page.wait_for_timeout(1000)
    wait_visible(page, "#badcase-detail-content")
    screenshot(page, "04_badcase_detail_pending")

    # Verify status-specific buttons for pending
    for selector in ["#badcase-classify", "#badcase-auto-classify", "#badcase-reject"]:
        wait_visible(page, selector)
    results.append({"check": "pending badcase action buttons visible", "ok": True})

    # Classify it as knowledge_gap via UI to reach classified state
    page.locator("#badcase-category").select_option("knowledge_gap")
    with page.expect_response(
        lambda resp: "/api/badcases/" in resp.url and "/classify" in resp.url and resp.status == 200,
        timeout=30000,
    ):
        page.locator("#badcase-classify").click()
    page.wait_for_timeout(500)
    page.reload(wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(800)

    # Navigate back to the badcase detail
    wait_visible(page, '.top-tab[data-top="platform"]').click()
    page.wait_for_timeout(500)
    wait_visible(page, '#sub-menu button[data-sub="badcases"]').click()
    page.wait_for_timeout(800)
    page.locator("#badcase-filter-source").select_option("manual")
    page.locator("#badcase-filter-status").select_option("classified")
    page.locator("#badcase-filter-btn").click()
    page.wait_for_timeout(800)
    wait_visible(page, f"button[data-id='{case_id}']").click()
    page.wait_for_timeout(800)
    wait_visible(page, "#badcase-detail-content")
    screenshot(page, "05_badcase_detail_classified")

    # Verify classified state buttons
    for selector in ["#badcase-darwin", "#badcase-reject"]:
        wait_visible(page, selector)
    results.append({"check": "classified badcase action buttons visible", "ok": True})


def run_cost_governance(page: Page):
    wait_visible(page, '#sub-menu button[data-sub="cost-governance"]').click()
    page.wait_for_timeout(1200)
    wait_visible(page, "#main-content")
    screenshot(page, "06_cost_governance")

    # Overview period cards
    for text in ("今日调用次数", "今日总 Token", "今日估算成本", "平均单轮 Token"):
        assert_text(page, text, f"overview card '{text}'")

    # Period cards
    for text in ("今日", "近7天", "本月"):
        assert_text(page, text, f"period card '{text}'")

    # Distribution / charts
    assert_text(page, "按模型（Flash / Pro）", "model distribution chart")
    assert_text(page, "按阶段", "stage distribution chart")

    # Trace table
    assert_text(page, "调用 Trace 列表（含成本可解释性）", "trace table header")

    # Open first trace detail to verify cost formula rendering
    detail_btns = page.locator("button.cg-trace-detail").all()
    if detail_btns:
        detail_btns[0].click()
        page.wait_for_timeout(800)
        screenshot(page, "07_cost_trace_detail")
        modal_body = page.locator("#modal-body").first.inner_text()
        assert "成本计算公式" in modal_body, "trace detail cost formula not found in modal"
        results.append({"check": "trace detail cost formula visible", "ok": True})


def run_acceptance():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-proxy-server"])
            context = browser.new_context(viewport={"width": 1280, "height": 900})
            page = context.new_page()
            add_event_listeners(page)

            # First load
            page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1500)
            wait_visible(page, "#main-content")
            assert_text(page, "AI 物业维修助手", "owner tab initial content")
            check_global_errors("first load")

            # Owner tab quick smoke
            wait_visible(page, '.top-tab[data-top="owner"]').click()
            page.wait_for_timeout(800)
            wait_visible(page, "#main-content")
            screenshot(page, "00_owner_chat")
            check_global_errors("owner tab")

            # Platform tab: badcase operations
            run_platform_badcase(page)
            check_global_errors("platform badcase")

            # Cost governance page
            run_cost_governance(page)
            check_global_errors("cost governance")

            # Hard refresh and repeat platform checks
            page.reload(wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1500)
            wait_visible(page, "#main-content")
            assert_text(page, "AI 物业维修助手", "after hard refresh main content")
            check_global_errors("after hard refresh")

            wait_visible(page, '.top-tab[data-top="platform"]').click()
            page.wait_for_timeout(500)
            wait_visible(page, '#sub-menu button[data-sub="badcases"]').click()
            page.wait_for_timeout(1000)
            wait_visible(page, "#badcases-content")
            check_global_errors("after refresh badcases")
            screenshot(page, "08_after_refresh")

            browser.close()

        summary = {
            "base_url": BASE_URL,
            "out_dir": str(OUT_DIR),
            "total_checks": len(results),
            "page_errors": page_errors,
            "console_errors": console_errors,
            "failed_requests": failed_requests,
            "results": results,
            "created_badcase_ids": created_badcase_ids,
            "passed": not (page_errors or console_errors or failed_requests),
        }
        (OUT_DIR / "evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["passed"] else 1
    finally:
        api_cleanup()


if __name__ == "__main__":
    sys.exit(run_acceptance())
