"""
RAG citation relevance acceptance test.

Validates that RAG citations are grounded in actually relevant documents,
not decorative top-k candidates. Runs against a real SSE chat endpoint and
the retrieval debug endpoint.

Usage:
    python scripts/test_rag_relevance.py [--base-url BASE_URL]

Environment:
    TEST_BASE_URL overrides --base-url (default: https://maiyouxiong.duckdns.org:18004).
"""

import argparse
import json
import os
import re
import sys
import uuid
from urllib.parse import urljoin

import requests


def base_url():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("TEST_BASE_URL", "https://maiyouxiong.duckdns.org:18004"))
    args, _ = parser.parse_known_args()
    return args.base_url.rstrip("/")


def sse_chat(base: str, message: str, session_id: str):
    """Call /api/chat/stream and return the final done payload plus all events."""
    url = f"{base}/api/chat/stream?message={requests.utils.quote(message)}&session_id={session_id}&user_id=acceptance-test"
    events = []
    done_payload = None
    with requests.get(url, stream=True, timeout=120) as resp:
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


def chat_history(base: str, session_id: str):
    url = f"{base}/api/chat/history?session_id={session_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def retrieval_debug(base: str, query: str):
    url = f"{base}/api/knowledge/retrieval-debug"
    resp = requests.post(
        url,
        json={"query": query, "top_k": 5, "expected_doc_title": "装修管理规定与流程"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def extract_citation_titles(citations):
    return [c.get("doc_title") for c in (citations or []) if c.get("doc_title")]


def assert_contains(titles, expected, msg):
    if expected not in titles:
        raise AssertionError(f"{msg}: expected '{expected}' in {titles}")


def assert_not_contains(titles, forbidden, msg):
    for t in forbidden:
        if t in titles:
            raise AssertionError(f"{msg}: forbidden '{t}' found in {titles}")


def test_decoration_query(base: str):
    query = "装修施工允许的时间是什么？"
    session_id = f"rag-rel-{uuid.uuid4().hex[:12]}"
    print(f"\n=== A. {query} ===")

    done, events = sse_chat(base, query, session_id)
    if done is None:
        raise AssertionError("SSE did not emit a done event")

    sse_titles = extract_citation_titles(done.get("citations"))
    print(f"SSE done citations: {sse_titles}")

    assert_contains(sse_titles, "装修管理规定与流程", "SSE done citations missing expected doc")
    assert_not_contains(
        sse_titles,
        ["维修收费标准", "测试 badcase"],
        "SSE done citations contain unrelated docs",
    )

    hist = chat_history(base, session_id)
    assistant_messages = [m for m in hist.get("messages", []) if m.get("role") == "assistant"]
    if not assistant_messages:
        raise AssertionError("No assistant message in history")
    hist_titles = extract_citation_titles(assistant_messages[-1].get("citations"))
    print(f"History citations: {hist_titles}")

    assert_contains(hist_titles, "装修管理规定与流程", "History citations missing expected doc")
    assert_not_contains(
        hist_titles,
        ["维修收费标准", "测试 badcase"],
        "History citations contain unrelated docs",
    )

    debug = retrieval_debug(base, query)
    final_titles = [r.get("title") for r in debug.get("results", [])]
    keyword_titles = [r.get("title") for r in debug.get("keyword_results", [])]
    semantic_titles = [r.get("title") for r in debug.get("semantic_results", [])]
    print(f"Debug final results: {final_titles}")
    print(f"Debug keyword candidates: {keyword_titles}")
    print(f"Debug semantic candidates: {semantic_titles}")

    assert_contains(final_titles, "装修管理规定与流程", "Debug final missing expected doc")
    assert_not_contains(
        final_titles,
        ["维修收费标准", "测试 badcase"],
        "Debug final contains unrelated docs",
    )

    print("A PASSED")
    return {
        "query": query,
        "session_id": session_id,
        "sse_done": done,
        "history_citations": hist_titles,
        "debug": debug,
    }


def test_complaint_query(base: str):
    query = "夜间施工扰民，我要投诉。"
    session_id = f"rag-rel-{uuid.uuid4().hex[:12]}"
    print(f"\n=== B. {query} ===")

    done, events = sse_chat(base, query, session_id)
    if done is None:
        raise AssertionError("SSE did not emit a done event")

    sse_titles = extract_citation_titles(done.get("citations"))
    print(f"SSE done citations: {sse_titles}")

    forbidden = ["维修收费标准", "维修责任划分说明", "物业维修服务承诺", "测试 badcase"]
    assert_not_contains(sse_titles, forbidden, "SSE done citations contain unrelated docs")

    hist = chat_history(base, session_id)
    assistant_messages = [m for m in hist.get("messages", []) if m.get("role") == "assistant"]
    if not assistant_messages:
        raise AssertionError("No assistant message in history")
    hist_titles = extract_citation_titles(assistant_messages[-1].get("citations"))
    print(f"History citations: {hist_titles}")

    assert_not_contains(hist_titles, forbidden, "History citations contain unrelated docs")

    debug = retrieval_debug(base, query)
    final_titles = [r.get("title") for r in debug.get("results", [])]
    keyword_titles = [r.get("title") for r in debug.get("keyword_results", [])]
    semantic_titles = [r.get("title") for r in debug.get("semantic_results", [])]
    print(f"Debug final results: {final_titles}")
    print(f"Debug keyword candidates: {keyword_titles}")
    print(f"Debug semantic candidates: {semantic_titles}")

    assert_not_contains(final_titles, forbidden, "Debug final contains unrelated docs")

    print("B PASSED")
    return {
        "query": query,
        "session_id": session_id,
        "sse_done": done,
        "history_citations": hist_titles,
        "debug": debug,
    }


def main():
    base = base_url()
    print(f"Testing against {base}")
    results = []
    try:
        results.append(test_decoration_query(base))
        results.append(test_complaint_query(base))
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)

    out_path = "rag_relevance_acceptance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"base_url": base, "cases": results}, f, ensure_ascii=False, indent=2)
    print(f"\nAll tests passed. Evidence saved to {out_path}")


if __name__ == "__main__":
    main()
