"""
Model runtime policy SSE acceptance tests.

Verifies that ordinary owner chat queries return done events with:
- model_id = deepseek-v4-flash
- thinking_enabled = true
- model_selection_reason = owner-facing default

Usage:
    python scripts/test_model_runtime_sse.py [--base-url BASE_URL]

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


def sse_chat(base: str, message: str, session_id: str):
    """Call /api/chat/stream and return the final done payload plus all events."""
    url = f"{base}/api/chat/stream?message={requests.utils.quote(message)}&session_id={session_id}&user_id=acceptance-test"
    events = []
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
                if event_type and data:
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        payload = {"raw": data}
                    events.append({"event": event_type, "data": payload})
                    if event_type == "done":
                        done_payload = payload
    return done_payload, events


def assert_chat_model(done: dict, query: str):
    """Assert done event matches the Flash owner-facing default policy."""
    if done is None:
        raise AssertionError(f"[{query}] SSE did not emit a done event")

    model_id = done.get("model_id")
    thinking = done.get("thinking_enabled")
    reason = done.get("model_selection_reason")

    print(f"  model_id={model_id}, thinking_enabled={thinking}, reason={reason}")

    if model_id != "deepseek-v4-flash":
        raise AssertionError(f"[{query}] expected model_id=deepseek-v4-flash, got {model_id}")
    if thinking is not True:
        raise AssertionError(f"[{query}] expected thinking_enabled=true, got {thinking}")
    if reason != "owner-facing default":
        raise AssertionError(f"[{query}] expected model_selection_reason='owner-facing default', got {reason}")


def test_owner_chat_queries(base: str):
    """Run the three required owner chat queries and verify model metadata."""
    queries = [
        "装修施工允许的时间是什么？",
        "厨房漏水，需要报修",
        "夜间施工扰民，我要投诉",
    ]
    results = []
    for query in queries:
        print(f"\n=== {query} ===")
        session_id = f"model-rt-{uuid.uuid4().hex[:12]}"
        done, events = sse_chat(base, query, session_id)
        assert_chat_model(done, query)
        results.append({"query": query, "session_id": session_id, "done": done})
        print("PASSED")
    return results


def main():
    base = base_url()
    print(f"Testing SSE against {base}")
    try:
        results = test_owner_chat_queries(base)
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"\nHTTP ERROR: {e.response.status_code} {e.response.text[:500]}")
        sys.exit(1)

    out_path = "model_runtime_sse_acceptance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base_url": base, "cases": results}, f, ensure_ascii=False, indent=2)
    print(f"\nAll SSE model runtime tests passed. Evidence saved to {out_path}")


if __name__ == "__main__":
    main()
