#!/usr/bin/env python3
"""V1.3.3 browser acceptance suite for YIAI Portal.

Run with:
    BASE_URL=http://192.168.50.123:18005 python scripts/browser_acceptance_v1_3_3.py

Outputs:
    - screenshots/*.png
    - evidence.json
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from playwright.sync_api import sync_playwright, Page, expect

BASE_URL = os.environ.get("BASE_URL", "http://192.168.50.123:18005").rstrip("/")
OUT_DIR = Path(os.environ.get("OUT_DIR", "browser_acceptance_v1_3_3"))

page_errors: List[str] = []
console_errors: List[str] = []
failed_requests: List[str] = []
results: List[Dict[str, Any]] = []


def add_event_listeners(page: Page):
    page.on("pageerror", lambda exc: page_errors.append(str(exc)))
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url}: {req.failure_error_string}"))


def screenshot(page: Page, name: str):
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    results.append({"name": name, "screenshot": str(path), "url": page.url})


def assert_non_empty(page: Page, selector: str, label: str):
    el = page.locator(selector).first
    expect(el).to_be_visible(timeout=15000)
    text = el.inner_text().strip()
    ok = len(text) > 0 and len(el.locator("*").all_inner_texts()) > 0
    assert ok, f"{label} is empty"
    results.append({"check": label, "ok": True, "selector": selector})


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


def run_owner_tab(page: Page):
    page.locator('.top-tab[data-top="owner"]').click()
    page.wait_for_timeout(800)
    expect(page.locator("#main-content")).to_be_visible()
    assert_non_empty(page, "#main-content", "owner main content")

    # Wait for session list to populate and create a new session.
    page.wait_for_selector("#chat-session-list", timeout=15000)
    existing_count = page.locator("#chat-session-list .session-item").count()
    first_id_before = page.locator("#chat-session-list .session-item").first.get_attribute("data-session-id") if existing_count else None
    page.locator("#chat-new-session").click()
    page.wait_for_timeout(1200)
    after_count = page.locator("#chat-session-list .session-item").count()
    first_id_after = page.locator("#chat-session-list .session-item").first.get_attribute("data-session-id")
    assert first_id_after != first_id_before, "new session did not appear at top"
    assert after_count >= min(existing_count, 1), f"session list empty after new session: {after_count}"
    results.append({"check": "owner new session", "ok": True, "before_id": first_id_before, "after_id": first_id_after})

    # Refresh and verify top session remains.
    page.locator("#chat-refresh-btn").click()
    page.wait_for_timeout(1200)
    first_id_refresh = page.locator("#chat-session-list .session-item").first.get_attribute("data-session-id")
    assert first_id_refresh == first_id_after, "refresh changed top session unexpectedly"
    results.append({"check": "owner refresh sessions", "ok": True, "top_id": first_id_refresh})

    screenshot(page, "01_owner_chat")


def run_staff_tab(page: Page):
    page.locator('.top-tab[data-top="staff"]').click()
    page.wait_for_timeout(800)
    assert_non_empty(page, "#main-content", "staff main content")
    screenshot(page, "02_staff")


def run_platform_tab(page: Page):
    page.locator('.top-tab[data-top="platform"]').click()
    page.wait_for_timeout(800)
    assert_non_empty(page, "#main-content", "platform main content")
    screenshot(page, "03_platform_overview")

    # Badcase library
    page.locator('#sub-menu button[data-sub="badcases"]').click()
    page.wait_for_timeout(1000)
    assert_non_empty(page, "#badcases-content", "Badcase list content")
    screenshot(page, "04_badcases_list")

    # Open the most recent badcase detail (first row).
    detail_btns = page.locator("button[data-id]").all()
    if detail_btns:
        detail_btns[0].click()
        page.wait_for_timeout(1000)
        assert_non_empty(page, "#main-content", "Badcase detail content")
        # Ensure key sections exist.
        for selector in ["#badcase-darwin", "#badcase-extract"]:
            expect(page.locator(selector).first).to_be_visible(timeout=10000)
        screenshot(page, "05_badcase_detail")
    else:
        results.append({"check": "badcase detail", "ok": False, "reason": "no badcase rows"})

    # Cost governance page
    page.locator('#sub-menu button[data-sub="cost-governance"]').click()
    page.wait_for_timeout(1000)
    assert_non_empty(page, "#main-content", "cost governance content")
    screenshot(page, "06_cost_governance")


def run_acceptance():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        add_event_listeners(page)

        # First load
        page.goto(f"{BASE_URL}/", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1500)
        expect(page.locator("#main-content")).to_be_visible()
        assert_non_empty(page, "#main-content", "initial main content")
        check_global_errors("first load")

        run_owner_tab(page)
        check_global_errors("owner tab")
        run_staff_tab(page)
        check_global_errors("staff tab")
        run_platform_tab(page)
        check_global_errors("platform tab")

        # Hard refresh and repeat basic checks
        page.reload(wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(1500)
        assert_non_empty(page, "#main-content", "after hard refresh main content")
        check_global_errors("after hard refresh")
        page.locator('.top-tab[data-top="platform"]').click()
        page.wait_for_timeout(800)
        page.locator('#sub-menu button[data-sub="badcases"]').click()
        page.wait_for_timeout(1000)
        assert_non_empty(page, "#badcases-content", "badcases after refresh")
        check_global_errors("after refresh badcases")
        screenshot(page, "07_after_refresh")

        browser.close()

    summary = {
        "base_url": BASE_URL,
        "total_checks": len(results),
        "page_errors": page_errors,
        "console_errors": console_errors,
        "failed_requests": failed_requests,
        "results": results,
        "passed": not (page_errors or console_errors or failed_requests),
    }
    (OUT_DIR / "evidence.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(run_acceptance())
