"""Run exactly one S10-B.2 chat acceptance request and fully consume its SSE.

The client deliberately treats only ``done`` and ``error`` as semantic terminal
events.  It also drains the physical HTTP response after that event so the
server is not cancelled while it writes its transport-flush padding.

Use ``--self-test`` before any real request.  The self-test is deterministic and
does not open a network connection.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Dict, Iterable, List, Optional


TERMINAL_EVENTS = {"done", "error"}


def parse_sse_lines(lines: Iterable[bytes]) -> List[Dict[str, Any]]:
    """Parse an SSE byte stream without treating intermediate events as final."""

    events: List[Dict[str, Any]] = []
    event_name = "message"
    data_lines: List[str] = []

    def dispatch() -> None:
        nonlocal event_name, data_lines
        if not data_lines:
            event_name = "message"
            return
        raw_data = "\n".join(data_lines)
        try:
            payload: Any = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = {"raw": raw_data}
        events.append({"event": event_name, "data": payload})
        event_name = "message"
        data_lines = []

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            dispatch()
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value
        elif field == "data":
            data_lines.append(value)

    # Proxies may close immediately after the final record without a blank line.
    dispatch()
    return events


def terminal_event(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    terminals = [event for event in events if event.get("event") in TERMINAL_EVENTS]
    return terminals[-1] if terminals else None


def run_self_test() -> None:
    complete = (
        b'event: start\ndata: {"trace_id":"trace-test"}\n\n'
        b'event: route\ndata: {"current_agent":"test-agent"}\n\n'
        b'event: progress\ndata: {"stage":"model.invoke"}\n\n'
        b'event: delta\ndata: {"content":"hello"}\n\n'
        b'event: final\ndata: {"content":"hello world"}\n\n'
        b'event: done\ndata: {"status":"complete","content":"hello world"}\n\n'
        b': transport-flush padding\n\n'
    )
    events = parse_sse_lines(io.BytesIO(complete))
    assert [event["event"] for event in events] == [
        "start",
        "route",
        "progress",
        "delta",
        "final",
        "done",
    ]
    assert terminal_event(events)["event"] == "done"

    route_only = parse_sse_lines(
        io.BytesIO(b'event: start\ndata: {}\n\nevent: route\ndata: {}\n\n')
    )
    assert terminal_event(route_only) is None

    error = parse_sse_lines(
        io.BytesIO(b'event: start\ndata: {}\n\nevent: error\ndata: {"error":"failed"}')
    )
    assert terminal_event(error)["event"] == "error"

    multiline = parse_sse_lines(
        io.BytesIO(b'event: final\ndata: {"content":\ndata: "ok"}\n\n')
    )
    assert multiline[0]["data"]["content"] == "ok"

    print("SELF_TEST_PASS: waits for done/error and rejects route-only EOF")


def request_json(
    url: str,
    *,
    method: str,
    body: Optional[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Any]:
    encoded = None
    headers = {"Accept": "application/json"}
    if body is not None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: int) -> Dict[str, Any]:
    return request_json(url, method="GET", body=None, timeout=timeout)


def inspect_persisted_evidence(args: argparse.Namespace) -> int:
    """Read persisted evidence for a completed case using GET endpoints only."""

    case_path = Path(args.inspect_evidence_file)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    session_id = case.get("session_id")
    trace_id = case.get("trace_id")
    if not session_id or not trace_id:
        raise ValueError("acceptance evidence file has no session_id or trace_id")

    base_url = args.base_url.rstrip("/")
    quoted_session = urllib.parse.quote(str(session_id), safe="")
    quoted_trace = urllib.parse.quote(str(trace_id), safe="")
    trace_detail = get_json(
        f"{base_url}/api/observability/traces/{quoted_trace}", args.timeout
    )
    evidence = get_json(
        f"{base_url}/api/runtime/traces/{quoted_trace}/evidence", args.timeout
    )
    snapshot = get_json(
        f"{base_url}/api/runtime/sessions/{quoted_session}/snapshot", args.timeout
    )
    history = get_json(
        f"{base_url}/api/chat/history?{urllib.parse.urlencode({'session_id': session_id})}",
        args.timeout,
    )

    ledger_row = evidence.get("evidence") or {}
    ledger = evidence.get("ledger") or ledger_row.get("ledger") or {}
    model_calls = trace_detail.get("model_calls") or []
    trace = trace_detail.get("trace") or {}
    messages = history.get("messages") or []
    assistant_messages = [item for item in messages if item.get("role") == "assistant"]

    def count(name: str) -> int:
        value = ledger.get(name)
        return len(value) if isinstance(value, list) else (1 if value else 0)

    selected_calls = []
    for call in model_calls:
        usage = call.get("usage_normalized") or {}
        selected_calls.append(
            {
                "id": call.get("id"),
                "stage": call.get("stage"),
                "status": call.get("status"),
                "requested_model": call.get("requested_model"),
                "provider_response_model": call.get("provider_response_model"),
                "provider_request_id": call.get("provider_request_id"),
                "provider_request_sequence": call.get("provider_request_sequence"),
                "thinking_enabled": call.get("thinking_enabled"),
                "usage_source": call.get("usage_source") or usage.get("usage_source"),
                "input_cache_hit_tokens": usage.get("input_cache_hit_tokens"),
                "input_cache_miss_tokens": usage.get("input_cache_miss_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "total_tokens": call.get("total_tokens"),
                "cost_cny": call.get("estimated_cost_cny"),
            }
        )

    summary = {
        "case": case.get("case"),
        "session_id": session_id,
        "trace_id": trace_id,
        "sse_event_names": case.get("event_names") or [],
        "sse_terminal": case.get("terminal"),
        "trace": {
            key: trace.get(key)
            for key in (
                "status",
                "intent",
                "agent_name",
                "agent_id",
                "runtime_path",
                "created_at",
                "updated_at",
            )
        },
        "route_decision": ledger.get("route_decision") or {},
        "lane_decision": ledger.get("lane_decision") or {},
        "capability_decision": ledger.get("capability_decision") or {},
        "runtime_release": {
            key: (snapshot.get("snapshot") or {}).get(key)
            for key in ("release_id", "release_version", "snapshot_hash", "created_at")
        },
        "ledger_status": {
            "status": ledger_row.get("status"),
            "runtime_path": ledger_row.get("runtime_path"),
        },
        "capability_counts": {
            name: count(name)
            for name in (
                "activated_skills",
                "retrieval_evidence",
                "citation_links",
                "tool_invocations",
                "action_proposals",
                "action_receipts",
                "handoff_events",
                "badcase_links",
                "contract_violations",
            )
        },
        "activated_skills": ledger.get("activated_skills") or [],
        "retrieval_evidence": ledger.get("retrieval_evidence") or [],
        "citation_links": ledger.get("citation_links") or [],
        "tool_invocations": ledger.get("tool_invocations") or [],
        "action_proposals": ledger.get("action_proposals") or [],
        "action_receipts": ledger.get("action_receipts") or [],
        "handoff_events": ledger.get("handoff_events") or [],
        "badcase_links": ledger.get("badcase_links") or [],
        "contract_violations": ledger.get("contract_violations") or [],
        "model_calls": selected_calls,
        "trace_cost_summary": trace_detail.get("trace_cost_summary") or {},
        "assistant_messages": [
            {
                "trace_id": item.get("trace_id"),
                "content": item.get("content"),
                "metadata": item.get("metadata"),
            }
            for item in assistant_messages
        ],
        "trace_event_types": [
            item.get("event_type") or item.get("type") or item.get("event")
            for item in (trace_detail.get("trace_events") or [])
        ],
        "ledger_keys": sorted(ledger.keys()),
    }
    inspection_path = case_path.with_name(case_path.stem + "-inspection.json")
    inspection_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "raw": {
                    "trace_detail": trace_detail,
                    "evidence": evidence,
                    "snapshot": snapshot,
                    "history": history,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"INSPECTION_FILE={inspection_path}")
    return 0


def run_single_chat(args: argparse.Namespace) -> int:
    if args.question_base64:
        question = base64.b64decode(args.question_base64).decode("utf-8")
    else:
        question = args.question
    if not question:
        raise ValueError("--question or --question-base64 is required")

    base_url = args.base_url.rstrip("/")
    session_payload = request_json(
        f"{base_url}/api/chat/sessions?{urllib.parse.urlencode({'user_id': args.user_id})}",
        method="POST",
        body={},
        timeout=args.timeout,
    )
    session = session_payload.get("session") or {}
    session_id = session.get("session_id")
    if not session_id:
        raise RuntimeError("session creation returned no session_id")

    body = json.dumps(
        {"message": question, "session_id": session_id, "user_id": args.user_id},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat/stream",
        data=body,
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.time()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"chat stream returned HTTP {response.status}")
        # parse_sse_lines consumes the iterator through physical EOF.  It never
        # returns merely because route, delta, final, or done has arrived.
        events = parse_sse_lines(response)

    terminal = terminal_event(events)
    start_event = next((event for event in events if event["event"] == "start"), None)
    trace_id = None
    if start_event and isinstance(start_event.get("data"), dict):
        trace_id = start_event["data"].get("trace_id")
    if not trace_id and terminal and isinstance(terminal.get("data"), dict):
        trace_id = terminal["data"].get("trace_id")

    evidence = {
        "case": args.label,
        "question": question,
        "session": session,
        "session_id": session_id,
        "trace_id": trace_id,
        "http_status": 200,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "event_names": [event["event"] for event in events],
        "events": events,
        "terminal": terminal,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"s10b2-{args.label.lower()}-{session_id}.json"
    output_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"SESSION_ID={session_id}")
    print(f"TRACE_ID={trace_id or ''}")
    print("EVENT_SEQUENCE=" + ",".join(evidence["event_names"]))
    print(f"TERMINAL_EVENT={(terminal or {}).get('event', '')}")
    print(f"EVIDENCE_FILE={output_path}")
    sys.stdout.flush()

    if terminal is None:
        print("ACCEPTANCE_CLIENT_FAIL: physical EOF before done/error", file=sys.stderr)
        return 3
    if terminal["event"] == "error":
        print("ACCEPTANCE_STREAM_ERROR: server emitted error", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--question")
    parser.add_argument("--question-base64")
    parser.add_argument("--label", default="case")
    parser.add_argument("--user-id", default="web-user")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--output-dir", default="/tmp/s10b2-acceptance")
    parser.add_argument("--inspect-evidence-file")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_test()
        return 0
    try:
        if args.inspect_evidence_file:
            return inspect_persisted_evidence(args)
        return run_single_chat(args)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ACCEPTANCE_CLIENT_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
