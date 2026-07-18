"""Fast, no-model contract checks for the V1.5.3 human-copilot boundary."""

import ast
from pathlib import Path

from app.handoff_policy import evaluate_handoff_policy, is_transition_allowed


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    for relative in ("app/handoff_policy.py", "app/chat.py", "db/property_db.py"):
        path = repo / relative
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    direct = evaluate_handoff_policy("装修施工允许的时间是什么？")
    expect(direct["level"] == "L0" and not direct["should_request_handoff"], "ordinary RAG question must stay L0")

    explicit = evaluate_handoff_policy("我不要 AI 了，请转人工客服")
    expect(explicit["level"] == "L3" and explicit["reason_code"] == "owner_requested", "explicit owner request must create L3")

    safety = evaluate_handoff_policy("燃气泄漏了，家里有味道")
    expect(safety["level"] == "L3" and safety["should_request_handoff"], "safety risk must escalate")

    finance = evaluate_handoff_policy("物业费重复扣费了，我要退款")
    expect(finance["level"] == "L2" and not finance["should_request_handoff"], "financial decision must require human approval, not fake auto-action")

    generic_complaint = evaluate_handoff_policy("夜间施工太吵，我要投诉")
    expect(generic_complaint["level"] == "L0", "ordinary complaint must not be over-escalated")

    failed_tool = evaluate_handoff_policy("查一下天气", mcp_calls=[{"tool_name": "weather", "status": "failed"}])
    expect(failed_tool["reason_code"] == "tool_failure", "failed tool must be visible to policy")

    expect(is_transition_allowed("requested", "active"), "request should be claimable")
    expect(is_transition_allowed("active", "waiting_user"), "staff may request owner information")
    expect(is_transition_allowed("waiting_user", "active"), "owner supplement should resume staff responsibility")
    expect(is_transition_allowed("resolved", "closed"), "resolved work should close explicitly")
    expect(not is_transition_allowed("closed", "active"), "closed handoff must not silently reopen")
    print("V1.5.3 human-copilot policy contract passed")


if __name__ == "__main__":
    main()
