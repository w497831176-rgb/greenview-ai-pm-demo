"""Replay the saved N241 Router response without calling any Provider.

The source response remains in its original Agno AgentSession.  This script
only reads that response, applies the production parser and contracts, and
writes a machine-readable derived result file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from agents.router import _strict_json_object
from app.runtime.contracts import LaneDecision, ResponseMode, RuntimeLane
from app.runtime.coordinator import _answer_contract_for
from app.runtime.snapshot_resolver import resolve_snapshot
from db import get_postgres_db
from db.property_db import get_chat_trace, get_model_calls_for_trace


def _json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def replay(args: argparse.Namespace) -> Dict[str, Any]:
    trace = get_chat_trace(args.trace_id)
    if not trace:
        raise RuntimeError("source N241 Trace not found")
    source_calls = get_model_calls_for_trace(args.trace_id)
    if len(source_calls) != 1:
        raise RuntimeError("source N241 must retain exactly one original Provider call")

    router_session = get_postgres_db().get_session(args.router_session_id)
    runs = list(getattr(router_session, "runs", []) or []) if router_session else []
    if len(runs) != 1:
        raise RuntimeError("source N241 must retain exactly one Router run")
    raw = getattr(runs[0], "content", None)
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("source N241 raw Provider response is unavailable")

    snapshot = resolve_snapshot(args.chat_session_id)
    decision = LaneDecision.model_validate(_strict_json_object(raw))
    contract = _answer_contract_for(decision)

    if decision.lane != RuntimeLane.ISOLATED_GENERAL:
        raise RuntimeError("N241 replay Lane is not C_ISOLATED_GENERAL")
    if contract.response_mode != ResponseMode.SAFE_GENERAL:
        raise RuntimeError("N241 replay AnswerContract is not safe_general")
    if any(
        value != "skipped"
        for value in (contract.skill_policy, contract.rag_policy, contract.tool_policy)
    ):
        raise RuntimeError("N241 replay unexpectedly enables a property capability")
    if contract.write_policy != "forbidden" or contract.handoff_policy != "skipped":
        raise RuntimeError("N241 replay write/handoff boundary is inconsistent")

    source_call = source_calls[0]
    usage_normalized = _json_object(source_call.get("usage_normalized"))
    result = {
        "id": "N241",
        "expected_lane": "C_ISOLATED_GENERAL",
        "actual_lane": decision.lane.value,
        "schema_valid": True,
        "passed": True,
        "offline_replay": True,
        "provider_called": False,
        "provider_request_count": 0,
        "source_provider_request_count": 1,
        "source_trace_id": args.trace_id,
        "source_session_id": args.chat_session_id,
        "source_router_session_id": args.router_session_id,
        "source_model_call_id": source_call.get("id"),
        "source_provider_request_id": usage_normalized.get("provider_request_id"),
        "source_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "source_response_unchanged": True,
        "decision": decision.model_dump(mode="json"),
        "answer_contract": contract.model_dump(mode="json"),
        "capabilities": {
            "skill": "skipped",
            "rag": "skipped",
            "mcp": "skipped",
            "tool": "skipped",
            "action_gateway": "skipped",
        },
        "evidence_source": "saved_agno_provider_response_offline_replay",
    }
    if len(get_model_calls_for_trace(args.trace_id)) != len(source_calls):
        raise RuntimeError("offline replay unexpectedly changed Provider call evidence")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-id", default="ca0b933962854cf9")
    parser.add_argument("--chat-session-id", default="s10b2-fix2-router-eval-n241-61c7a84d")
    parser.add_argument(
        "--router-session-id",
        default="s10b2-fix2-router-eval-n241-61c7a84d::semantic_router::ca0b933962854cf9",
    )
    parser.add_argument("--output", default="/tmp/s10b2-fix2-n241-offline-replay.json")
    args = parser.parse_args()
    result = replay(args)
    print(json.dumps({key: value for key, value in result.items() if key not in {"decision", "answer_contract"}}, ensure_ascii=False, indent=2))
    print(f"RESULT_FILE={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
