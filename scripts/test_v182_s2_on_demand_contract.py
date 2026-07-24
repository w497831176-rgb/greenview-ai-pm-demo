"""Dependency-free behavior contracts for V1.8.2-S2 on-demand capabilities."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

from app.handoff_policy import evaluate_handoff_policy


ROOT = Path(__file__).resolve().parents[1]


def _load_coordinator_selector():
    source_path = ROOT / "app/runtime/coordinator.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        "STRUCTURED_WORKORDER_TOOLS",
        "KNOWLEDGE_EVIDENCE_TERMS",
        "_is_structured_realtime_query",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if targets & names:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
    }
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(source_path),
            "exec",
        ),
        namespace,
    )
    return namespace["_is_structured_realtime_query"]


is_structured_realtime_query = _load_coordinator_selector()


def _plan(server: str, tool: str):
    return SimpleNamespace(server_name=server, tool_name=tool)


def test_exact_workorder_query_is_tool_only():
    plans = [_plan("workorder-server", "get_my_work_order_by_id")]
    assert is_structured_realtime_query(
        "帮我查询工单 WO-20260714-001 现在是什么状态？",
        plans,
    )
    assert not is_structured_realtime_query(
        "请查询工单 WO-20260714-001，并依据制度说明处理时效。",
        plans,
    )
    assert not is_structured_realtime_query(
        "物业紧急维修响应时效是多少？",
        [],
    )


def test_handoff_user_request_and_negation():
    explicit = evaluate_handoff_policy(
        "这个问题我不想再和 AI 沟通了，请帮我转人工。"
    )
    assert explicit["should_request_handoff"]
    assert explicit["reason_code"] == "owner_requested"

    negated = evaluate_handoff_policy(
        "我暂时不用转人工，只想了解一下报修流程。"
    )
    assert not negated["should_request_handoff"]
    assert negated["reason_code"] == "negated_by_user"


def test_handoff_safety_overrides_negation():
    safety = evaluate_handoff_policy(
        "不用转人工，但电梯里有人被困，请告诉我怎么办。"
    )
    assert safety["should_request_handoff"]
    assert safety["reason_code"] == "safety_risk"
    assert "safety_override" in safety["matched_signals"]


def test_runtime_records_real_selected_and_skipped_reasons():
    source = (ROOT / "app/runtime/coordinator.py").read_text(encoding="utf-8")
    agent_source = (ROOT / "app/runtime/agent_factory.py").read_text(
        encoding="utf-8"
    )
    assert "enable_skills=not structured_realtime_query" in source
    assert '"skipped_structured_realtime_query"' in source
    assert '"capability_decision"' in source
    assert '"decision_summary": decision_summary' in source
    assert "enable_skills: bool = True" in agent_source


def main():
    tests = [
        test_exact_workorder_query_is_tool_only,
        test_handoff_user_request_and_negation,
        test_handoff_safety_overrides_negation,
        test_runtime_records_real_selected_and_skipped_reasons,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("V1.8.2-S2 on-demand capability contracts passed.")


if __name__ == "__main__":
    main()
