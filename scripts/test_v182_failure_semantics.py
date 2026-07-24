"""Focused dependency-free checks for V1.8 Demo failure semantics."""

from __future__ import annotations

import ast
import asyncio
import copy
import json
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _load_source_symbols(relative_path: str, names: set[str]) -> Dict[str, Any]:
    """Execute selected helpers from the real source without importing Agno."""

    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign):
            targets = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if targets & names:
                selected.append(node)
    namespace: Dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "Optional": Optional,
        "Tuple": Tuple,
        "json": json,
        "re": re,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, relative_path, "exec"), namespace)
    return namespace


def _load_async_method(
    relative_path: str,
    method_name: str,
    namespace: Dict[str, Any],
):
    """Compile one real async method body with dependency-free test doubles."""

    source = (ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == method_name
    )
    method = copy.deepcopy(method)
    method.name = f"_{method_name}_under_test"
    method.decorator_list = []
    method.returns = None
    for argument in [
        *method.args.posonlyargs,
        *method.args.args,
        *method.args.kwonlyargs,
    ]:
        argument.annotation = None
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, relative_path, "exec"), namespace)
    return namespace[method.name]


coordinator_symbols = _load_source_symbols(
    "app/runtime/coordinator.py",
    {
        "PROVIDER_FAILURE_MARKERS",
        "PROVIDER_FAILURE_PUBLIC_MESSAGE",
        "ProviderFailureError",
        "_provider_failure_prefix",
        "_provider_failure_reason",
    },
)
PROVIDER_FAILURE_PUBLIC_MESSAGE = coordinator_symbols[
    "PROVIDER_FAILURE_PUBLIC_MESSAGE"
]
ProviderFailureError = coordinator_symbols["ProviderFailureError"]
_provider_failure_prefix = coordinator_symbols["_provider_failure_prefix"]
_provider_failure_reason = coordinator_symbols["_provider_failure_reason"]

mcp_symbols = _load_source_symbols(
    "app/runtime/mcp_executor.py",
    {"_business_status", "_structured_result"},
)
_business_status = mcp_symbols["_business_status"]

badcase_symbols = _load_source_symbols(
    "app/runtime/badcase_capture.py",
    {"_failed_evaluations", "_failed_tools", "runtime_badcase_trigger"},
)
runtime_badcase_trigger = badcase_symbols["runtime_badcase_trigger"]


def test_provider_failure_text_detection():
    assert _provider_failure_reason("Insufficient Balance") == "insufficient balance"
    assert _provider_failure_prefix("Insuff")
    assert _provider_failure_reason("维修人员将在 30 分钟内到场。") is None


def test_plain_tool_result_status():
    status, summary = _business_status(
        "content='11152.0' metadata=None images=None videos=None"
    )
    assert status == "success"
    assert "11152.0" in summary

    status, _ = _business_status("Error: division by zero")
    assert status == "upstream_error"

    status, _ = _business_status(None)
    assert status == "empty"

    status, _ = _business_status('{"status":"not_found","data":null}')
    assert status == "not_found"


def test_provider_failure_badcase_has_priority():
    class FakeLedger:
        contract_violations = [
            {
                "code": "required_citation_missing",
                "detail": "citation missing because Provider failed",
            }
        ]
        evaluation_results = []
        tool_invocations = []

    ledger = FakeLedger()
    trigger = runtime_badcase_trigger(
        ledger,
        runtime_error="model Provider returned failure text: insufficient balance",
        runtime_error_type="provider_failure",
    )
    assert trigger
    assert trigger["source"] == "provider_failure"
    assert trigger["category"] == "provider_failure"
    assert trigger["root_cause_domain"] == "model_provider"


def test_tool_failure_enters_badcase_evidence():
    class FakeLedger:
        contract_violations = []
        evaluation_results = []
        tool_invocations = [
            {
                "tool_name": "calculator",
                "transport_status": "success",
                "invocation_status": "success",
                "business_status": "upstream_error",
            }
        ]

    trigger = runtime_badcase_trigger(FakeLedger())
    assert trigger
    assert trigger["source"] == "tool_failure"
    assert trigger["failed_tools"][0]["tool_name"] == "calculator"


def test_coordinator_provider_failure_terminal_behavior():
    class RuntimePathValue:
        def __init__(self, value):
            self.value = value

    class RuntimePath:
        CONSULTATION = RuntimePathValue("consultation")
        CONTROLLED_ACTION = RuntimePathValue("controlled_action")

    class RunStatus:
        RUNNING = "running"
        COMPLETED = "completed"
        FAILED = "failed"

    class RunState:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    ledgers = []

    class EvidenceLedger:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.contract_violations = []
            self.evaluation_results = []
            self.tool_invocations = []
            self.persisted = []
            self.appended = []
            ledgers.append(self)

        @property
        def contract(self):
            return self

        def violation(self, code, detail):
            self.contract_violations.append({"code": code, "detail": detail})

        def capture_state(self, state):
            self.state = state

        def persist(self, status):
            self.persisted.append(status)

        def append(self, bucket, payload):
            self.appended.append((bucket, payload))

    trace_updates = []
    trace_events = []
    badcase_calls = []

    def record_trace_event(*args, **kwargs):
        trace_events.append((args, kwargs))

    def update_chat_trace(*args, **kwargs):
        trace_updates.append((args, kwargs))

    def capture_runtime_badcase(**kwargs):
        badcase_calls.append(kwargs)
        return {"id": 901, "source": "provider_failure"}

    namespace = {
        "asyncio": asyncio,
        "time": time,
        "uuid": uuid,
        "RuntimePath": RuntimePath,
        "RunStatus": RunStatus,
        "RunState": RunState,
        "EvidenceLedger": EvidenceLedger,
        "ProviderFailureError": ProviderFailureError,
        "PROVIDER_FAILURE_PUBLIC_MESSAGE": PROVIDER_FAILURE_PUBLIC_MESSAGE,
        "ensure_chat_session": lambda _session_id: None,
        "resolve_snapshot": lambda _session_id: SimpleNamespace(
            config={},
            snapshot_id="snap_test",
            snapshot_hash="hash_test",
            release_id="release_test",
        ),
        "create_chat_trace": lambda **_kwargs: None,
        "save_chat_message": lambda **_kwargs: {"id": 1},
        "record_trace_event": record_trace_event,
        "update_chat_trace": update_chat_trace,
        "capture_runtime_badcase": capture_runtime_badcase,
        "_sse": lambda event, payload: {"event": event, "data": payload},
    }
    stream = _load_async_method(
        "app/runtime/coordinator.py",
        "stream",
        namespace,
    )

    class FakeCoordinator:
        @staticmethod
        def _select_path(_session_id, _message, _config):
            return RuntimePath.CONSULTATION

        async def _maybe_handoff(self, *_args):
            return None

        async def _stream_consultation(self, *_args):
            if False:
                yield None
            raise ProviderFailureError(
                "model Provider returned failure text: insufficient balance"
            )

    async def run():
        return [
            event
            async for event in stream(
                FakeCoordinator(),
                "需要引用的维修时效问题",
                "session_test",
                "owner_test",
            )
        ]

    events = asyncio.run(run())
    event_names = [event["event"] for event in events]
    assert event_names == ["start", "error", "done"]
    assert "Insufficient Balance" not in json.dumps(events, ensure_ascii=False)
    assert events[1]["data"]["error"] == PROVIDER_FAILURE_PUBLIC_MESSAGE
    assert events[1]["data"]["error_code"] == "provider_failure"
    assert events[1]["data"]["status"] == "failed"
    assert events[1]["data"]["auto_badcase_id"] == 901
    assert events[2]["data"]["status"] == "failed"
    assert events[2]["data"]["error_code"] == "provider_failure"
    assert events[2]["data"]["auto_badcase_id"] == 901
    assert badcase_calls[0]["runtime_error_type"] == "provider_failure"
    assert trace_updates[-1][1]["status"] == "failed"
    assert trace_events[-1][0][1:3] == ("provider_failure", "failed")
    assert ledgers[0].persisted[-1] == "failed"


def main():
    tests = [
        test_provider_failure_text_detection,
        test_plain_tool_result_status,
        test_provider_failure_badcase_has_priority,
        test_tool_failure_enters_badcase_evidence,
        test_coordinator_provider_failure_terminal_behavior,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("V1.8.2 focused failure semantics checks passed.")


if __name__ == "__main__":
    main()
