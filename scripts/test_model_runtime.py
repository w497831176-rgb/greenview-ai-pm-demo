"""
Model runtime policy acceptance tests.

Verifies that the SQLite model catalog, A/B test endpoint, and Darwin deep-fix
endpoint all follow the unified model strategy:
- deepseek-v4-flash is the runtime default for owner-facing chat
- deepseek-v4-pro is only used for A/B tests and Darwin deep-fix
- No API key plaintext is exposed by /api/models

Usage:
    python scripts/test_model_runtime.py [--base-url BASE_URL]

Environment:
    TEST_BASE_URL overrides --base-url (default: https://maiyouxiong.duckdns.org:18004).
"""

import argparse
import json
import os
import sys
import uuid

import requests
import urllib3

urllib3.disable_warnings()

for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(_k, None)

_SESSION = requests.Session()
_SESSION.verify = False
_SESSION.proxies = {"http": None, "https": None}


def base_url():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("TEST_BASE_URL", "https://maiyouxiong.duckdns.org:18004"),
    )
    args, _ = parser.parse_known_args()
    return args.base_url.rstrip("/")


def api_get(base: str, path: str):
    url = f"{base}{path}"
    resp = _SESSION.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_post(base: str, path: str, payload=None):
    url = f"{base}{path}"
    resp = _SESSION.post(url, json=payload or {}, timeout=300)
    resp.raise_for_status()
    return resp.json()


def test_model_catalog_policy(base: str):
    """Verify GET /api/models exposes safe fields and Flash is the default."""
    print("\n=== Model catalog policy ===")
    data = api_get(base, "/api/models")
    models = data.get("models", [])
    if not models:
        raise AssertionError("No models returned")

    by_id = {m.get("model_id") or m.get("key"): m for m in models}
    if "deepseek-v4-flash" not in by_id:
        raise AssertionError("deepseek-v4-flash missing from catalog")
    if "deepseek-v4-pro" not in by_id:
        raise AssertionError("deepseek-v4-pro missing from catalog")

    flash = by_id["deepseek-v4-flash"]
    pro = by_id["deepseek-v4-pro"]

    # Safe fields only.
    for m in (flash, pro):
        if "api_key" in m:
            raise AssertionError("api_key plaintext exposed in /api/models")
        if m.get("credential_status") not in ("server_env", "missing"):
            raise AssertionError(f"unexpected credential_status: {m.get('credential_status')}")

    if not flash.get("is_default"):
        raise AssertionError("deepseek-v4-flash must be is_default=true")
    if pro.get("is_default"):
        raise AssertionError("deepseek-v4-pro must be is_default=false")
    if not flash.get("thinking_enabled"):
        raise AssertionError("deepseek-v4-flash must have thinking_enabled=true")
    if not pro.get("thinking_enabled"):
        raise AssertionError("deepseek-v4-pro must have thinking_enabled=true")

    print(f"Flash: is_default={flash['is_default']}, thinking={flash['thinking_enabled']}, credential={flash['credential_status']}")
    print(f"Pro:   is_default={pro['is_default']}, thinking={pro['thinking_enabled']}, credential={pro['credential_status']}")
    print("Model catalog policy PASSED")
    return {"flash": flash, "pro": pro}


def test_ab_test_fixed_models(base: str):
    """A/B test must always compare flash vs pro."""
    print("\n=== A/B test fixed models ===")
    payload = {"prompt": "简要说明物业客服应该如何处理业主投诉夜间施工扰民"}
    data = api_post(base, "/api/models/ab-test", payload)

    model_a = data.get("model_a", {})
    model_b = data.get("model_b", {})

    a_id = model_a.get("model_id")
    b_id = model_b.get("model_id")

    expected = {"deepseek-v4-flash", "deepseek-v4-pro"}
    if {a_id, b_id} != expected:
        raise AssertionError(f"A/B models mismatch: a={a_id}, b={b_id}")

    for key, result in (("model_a", model_a), ("model_b", model_b)):
        if result.get("error"):
            raise AssertionError(f"{key} returned error: {result['error']}")
        if not result.get("response"):
            raise AssertionError(f"{key} returned empty response")

    print(f"A = {a_id}, B = {b_id}")
    print("A/B test fixed models PASSED")
    return data


def test_darwin_uses_pro(base: str):
    """Darwin deep-fix must explicitly use deepseek-v4-pro."""
    print("\n=== Darwin deep-fix uses Pro ===")
    # Create a minimal badcase first.
    session_id = f"darwin-{uuid.uuid4().hex[:12]}"
    create_resp = api_post(
        base,
        "/api/badcases",
        {
            "title": "模型策略验收测试 badcase",
            "description": "用于验证 Darwin 修复是否使用 deepseek-v4-pro",
            "category": "knowledge",
            "session_id": session_id,
            "source_message_id": None,
        },
    )
    badcase = create_resp.get("badcase")
    if not badcase or not badcase.get("id"):
        raise AssertionError("Failed to create badcase")
    case_id = badcase["id"]

    fix_resp = api_post(base, f"/api/badcases/{case_id}/darwin-fix", {})
    model_id = fix_resp.get("model_id")
    fix_plan = fix_resp.get("fix_plan")

    if model_id != "deepseek-v4-pro":
        raise AssertionError(f"Darwin fix used wrong model: {model_id}")
    if not fix_plan:
        raise AssertionError("Darwin fix returned empty fix_plan")

    print(f"Badcase {case_id}: model_id={model_id}")
    print("Darwin deep-fix uses Pro PASSED")
    return fix_resp


def test_build_model_default(base: str):
    """Runtime default for owner chat is flash (observed via a real chat SSE in test_model_runtime_sse.py)."""
    print("\n=== Runtime default check via settings endpoint ===")
    # The public API does not expose build_model directly; we rely on SSE done event verification.
    print("See scripts/test_model_runtime_sse.py for runtime default evidence")
    return {}


def main():
    base = base_url()
    print(f"Testing against {base}")
    results = {}
    try:
        results["catalog"] = test_model_catalog_policy(base)
        results["ab_test"] = test_ab_test_fixed_models(base)
        results["darwin"] = test_darwin_uses_pro(base)
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"\nHTTP ERROR: {e.response.status_code} {e.response.text[:500]}")
        sys.exit(1)

    out_path = "model_runtime_acceptance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base_url": base, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nAll model runtime tests passed. Evidence saved to {out_path}")


if __name__ == "__main__":
    main()
