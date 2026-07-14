"""
Model runtime policy acceptance tests.

Verifies that the SQLite model catalog, A/B test endpoint, Darwin deep-fix
endpoint, legacy key endpoints, and Skill model overrides all follow the
unified model strategy:
- deepseek-v4-flash is the runtime default for owner-facing chat
- deepseek-v4-pro is only used for A/B tests and Darwin deep-fix
- No API key plaintext is exposed or persisted by /api/models
- A/B is fixed to Flash vs Pro and cannot be tampered from the client
- Skill model_id cannot override the owner-facing default model

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


def api_get(base: str, path: str, timeout: int = 30):
    url = f"{base}{path}"
    resp = _SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_post(base: str, path: str, payload=None, timeout: int = 300, expect_status=None):
    url = f"{base}{path}"
    resp = _SESSION.post(url, json=payload or {}, timeout=timeout)
    if expect_status and resp.status_code != expect_status:
        raise AssertionError(
            f"POST {path} expected {expect_status}, got {resp.status_code}: {resp.text[:500]}"
        )
    resp.raise_for_status()
    return resp.json()


def api_delete(base: str, path: str, timeout: int = 30):
    url = f"{base}{path}"
    resp = _SESSION.delete(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def sse_chat_done(base: str, message: str, session_id: str) -> dict:
    """Call /api/chat/stream and return the final done payload."""
    url = (
        f"{base}/api/chat/stream?message={requests.utils.quote(message)}"
        f"&session_id={session_id}&user_id=acceptance-test"
    )
    done_payload = None
    with _SESSION.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        buffer = ""
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                if not block.strip():
                    continue
                event_type = None
                data = None
                for line in block.split("\n"):
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                if event_type == "done" and data:
                    try:
                        done_payload = json.loads(data)
                    except json.JSONDecodeError:
                        done_payload = {"raw": data}
    return done_payload


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


def test_ab_test_tamper_resistance(base: str):
    """Client cannot override A/B models; endpoint still returns Flash vs Pro."""
    print("\n=== A/B test tamper resistance ===")
    payload = {
        "prompt": "测试问题",
        "model_a": "deepseek-v4-pro",
        "model_b": "deepseek-v4-pro",
    }
    data = api_post(base, "/api/models/ab-test", payload)

    a_id = data.get("model_a", {}).get("model_id")
    b_id = data.get("model_b", {}).get("model_id")

    if a_id != "deepseek-v4-flash":
        raise AssertionError(f"tampered model_a was used: {a_id}")
    if b_id != "deepseek-v4-pro":
        raise AssertionError(f"tampered model_b was used: {b_id}")

    print(f"Client override ignored; A={a_id}, B={b_id}")
    print("A/B test tamper resistance PASSED")
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


def test_legacy_key_endpoints_retired(base: str):
    """Legacy browser key/default endpoints must be gone."""
    print("\n=== Legacy key/default endpoints retired ===")
    for path in [
        "/api/models/deepseek-v4-flash/key",
        "/api/models/deepseek-v4-pro/key",
        "/api/models/deepseek-v4-flash/default",
        "/api/models/deepseek-v4-pro/default",
    ]:
        url = f"{base}{path}"
        resp = _SESSION.post(url, json={}, timeout=30)
        if resp.status_code != 410:
            raise AssertionError(
                f"Expected 410 for {path}, got {resp.status_code}: {resp.text[:200]}"
            )
        print(f"{path} -> 410")

    # Also verify PUT /api/model-configs/{model_id} ignores browser-submitted api_key.
    cfg = api_get(base, "/api/model-configs/deepseek-v4-flash").get("model_config", {})
    original_name = cfg.get("name", "DeepSeek V4 Flash")
    resp = _SESSION.put(
        f"{base}/api/model-configs/deepseek-v4-flash",
        json={
            "name": original_name,
            "provider": "deepseek",
            "api_key": "ignored-browser-value",
            "base_url": "https://api.deepseek.com",
            "model_params": {"use_thinking": True},
            "enabled": True,
            "description": "常规文本 Router 与垂直 Agent 主力模型",
        },
        timeout=30,
    )
    resp.raise_for_status()
    updated = resp.json().get("model_config", {})
    if "api_key" in updated:
        raise AssertionError("PUT returned api_key plaintext")
    print("PUT /api/model-configs ignores browser api_key")

    print("Legacy key/default endpoints retired PASSED")
    return {"retired_paths_checked": 4}


def test_skill_model_override_blocked(base: str):
    """Owner chat ignores any Skill model_id override."""
    print("\n=== Skill model override blocked ===")
    created_skill_ids = []
    try:
        # Create two temporary skills with non-Flash model_id and matching triggers.
        for model_id in ["deepseek-v4-pro", "some-arbitrary-model-id"]:
            skill_name = f"model-override-test-{uuid.uuid4().hex[:8]}"
            trigger = f"模型覆盖测试{uuid.uuid4().hex[:8]}"
            resp = api_post(
                base,
                "/api/skills",
                {
                    "name": skill_name,
                    "description": "Temporary skill for model override policy test",
                    "instructions": "如果看到这条 Skill，请回答：已收到 Skill 指令。",
                    "category": "测试",
                    "enabled": True,
                    "trigger_condition": trigger,
                    "model_id": model_id,
                },
            )
            skill_id = resp.get("skill", {}).get("id")
            if not skill_id:
                raise AssertionError("Failed to create test skill")
            created_skill_ids.append(skill_id)
            print(f"Created skill {skill_id} with model_id={model_id}, trigger={trigger}")

            session_id = f"skill-override-{uuid.uuid4().hex[:12]}"
            done = sse_chat_done(base, trigger, session_id)
            if done is None:
                raise AssertionError(f"Skill {skill_id}: SSE did not emit done event")

            actual_model = done.get("model_id")
            thinking = done.get("thinking_enabled")
            reason = done.get("model_selection_reason")
            print(f"  done: model_id={actual_model}, thinking={thinking}, reason={reason}")

            if actual_model != "deepseek-v4-flash":
                raise AssertionError(
                    f"Skill {skill_id} override leaked to {actual_model}"
                )
            if thinking is not True:
                raise AssertionError(f"Skill {skill_id}: thinking_enabled={thinking}")
            if reason != "owner-facing default":
                raise AssertionError(f"Skill {skill_id}: reason={reason}")

        print("Skill model override blocked PASSED")
        return {"created_skill_ids": created_skill_ids}
    finally:
        for skill_id in created_skill_ids:
            try:
                api_delete(base, f"/api/skills/{skill_id}")
                print(f"Cleaned up test skill {skill_id}")
            except Exception as e:
                print(f"Failed to clean up skill {skill_id}: {e}")


def main():
    base = base_url()
    print(f"Testing against {base}")
    results = {}
    try:
        results["catalog"] = test_model_catalog_policy(base)
        results["ab_test"] = test_ab_test_fixed_models(base)
        results["ab_test_tamper"] = test_ab_test_tamper_resistance(base)
        results["darwin"] = test_darwin_uses_pro(base)
        results["legacy_endpoints"] = test_legacy_key_endpoints_retired(base)
        results["skill_override"] = test_skill_model_override_blocked(base)
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
