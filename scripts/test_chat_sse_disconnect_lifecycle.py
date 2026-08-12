"""Focused deterministic checks for detached owner-chat SSE delivery.

No real Coordinator, Provider, database, or HTTP server is used.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import AsyncIterator, Dict, List


TEMP_DIR = tempfile.TemporaryDirectory(prefix="yiai-chat-sse-")
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for _key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY"):
    os.environ[_key] = ""


def _load_transport_module():
    """Import the transport with deterministic stubs for unrelated business APIs."""

    work_order = ModuleType("app.work_order_workflow")
    work_order.decide_work_order_proposal = lambda **_kwargs: {}
    handoff_policy = ModuleType("app.handoff_policy")
    handoff_policy.evaluate_handoff_policy = lambda *_args, **_kwargs: {}
    property_db = ModuleType("db.property_db")
    for name in (
        "add_badcase_action",
        "cancel_handoff",
        "claim_handoff",
        "close_handoff",
        "create_badcase",
        "create_chat_feedback",
        "create_chat_session",
        "get_chat_feedback",
        "get_chat_message",
        "get_chat_session",
        "get_handoff_package",
        "get_previous_user_message",
        "get_user_feedback_badcase",
        "link_chat_feedback_badcase",
        "list_chat_messages",
        "list_handoff_sessions",
        "list_user_chat_sessions",
        "now_cn",
        "request_handoff",
        "resolve_handoff",
        "save_chat_message",
        "wait_for_handoff_user",
    ):
        setattr(property_db, name, lambda *_args, **_kwargs: None)
    property_db.list_chat_messages = lambda *_args, **_kwargs: []
    property_db.list_handoff_sessions = lambda *_args, **_kwargs: []
    property_db.list_user_chat_sessions = lambda *_args, **_kwargs: []
    sys.modules["app.work_order_workflow"] = work_order
    sys.modules["app.handoff_policy"] = handoff_policy
    sys.modules["db.property_db"] = property_db
    module_name = "app.runtime.legacy_chat"
    spec = importlib.util.spec_from_file_location(
        module_name,
        Path(__file__).resolve().parents[1] / "app/runtime/legacy_chat.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


legacy_chat = _load_transport_module()


def _frame(event: str) -> str:
    return f'event: {event}\ndata: {{"event":"{event}"}}\n\n'


def _text(chunk: object) -> str:
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8")
    return str(chunk)


def _event_name(frame: str) -> str | None:
    first = frame.splitlines()[0] if frame.splitlines() else ""
    return first.split(":", 1)[1].strip() if first.startswith("event:") else None


async def _drain(response: object) -> List[str]:
    return [_text(chunk) async for chunk in response.body_iterator]


async def _wait_registry_empty() -> None:
    for _ in range(100):
        if not legacy_chat._ACTIVE_STREAM_RUNS:
            return
        await asyncio.sleep(0)
    raise AssertionError("background stream registry was not released")


async def _wait_registry_present(session_id: str) -> None:
    for _ in range(100):
        if session_id in legacy_chat._ACTIVE_STREAM_RUNS:
            return
        await asyncio.sleep(0)
    raise AssertionError("background stream registry was not created")


class ScriptedCoordinator:
    modes: Dict[str, str] = {}
    releases: Dict[str, asyncio.Event] = {}
    entered: Dict[str, asyncio.Event] = {}
    completed: Dict[str, asyncio.Event] = {}
    cancelled: Dict[str, bool] = {}
    calls: Dict[str, int] = {}
    persisted: Dict[str, Dict[str, object]] = {}

    async def stream(
        self,
        _message: str,
        session_id: str,
        _user_id: str,
    ) -> AsyncIterator[str]:
        self.calls[session_id] = self.calls.get(session_id, 0) + 1
        self.cancelled[session_id] = False
        yield _frame("start")
        self.entered.setdefault(session_id, asyncio.Event()).set()
        try:
            mode = self.modes.get(session_id, "normal")
            if mode in {"blocked", "error"}:
                await self.releases.setdefault(session_id, asyncio.Event()).wait()
            if mode == "error":
                raise RuntimeError("symbolic-producer-failure")
            for event in ("lane", "delta", "final", "done"):
                yield _frame(event)
            self.persisted[session_id] = {
                "assistant_history": "symbolic-complete-answer",
                "trace_status": "complete",
                "provider_attempts": [
                    {
                        "stage": "router",
                        "usage_source": "provider_actual",
                        "usage": {
                            "cache_hit_tokens": 2,
                            "cache_miss_tokens": 3,
                            "output_tokens": 5,
                            "total_tokens": 10,
                        },
                    },
                    {
                        "stage": "vertical_agent",
                        "usage_source": "provider_actual",
                        "usage": {
                            "cache_hit_tokens": 7,
                            "cache_miss_tokens": 11,
                            "output_tokens": 13,
                            "total_tokens": 31,
                        },
                    },
                ],
                "usage_complete": True,
                "trace_events": [],
            }
            self.completed.setdefault(session_id, asyncio.Event()).set()
        except asyncio.CancelledError:
            self.cancelled[session_id] = True
            raise


async def test_normal_order_and_transport_flush() -> None:
    session_id = "sse-normal"
    response = await legacy_chat.chat_stream(
        legacy_chat.ChatRequest(message="symbolic-normal", session_id=session_id)
    )
    frames = await _drain(response)
    names = [_event_name(frame) for frame in frames]
    assert names == ["start", "lane", "delta", "final", "done", None]
    assert frames[-1].startswith(": transport-flush ")
    assert frames[-1].endswith("\n\n")
    assert frames[-1].count(" ") >= 4096
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["connection"] == "keep-alive"
    assert response.headers["x-accel-buffering"] == "no"
    await _wait_registry_empty()
    assert ScriptedCoordinator.calls[session_id] == 1
    print("PASS test_normal_order_and_transport_flush")


async def _disconnect_asgi_response(response: object) -> None:
    """Exercise Starlette's ASGI 2.4 send-failure disconnect path."""

    sent_bodies = 0

    async def receive() -> Dict[str, object]:
        return {"type": "http.disconnect"}

    async def send(message: Dict[str, object]) -> None:
        nonlocal sent_bodies
        if message.get("type") != "http.response.body":
            return
        sent_bodies += 1
        if sent_bodies >= 1:
            raise OSError("symbolic-client-disconnect")

    try:
        await response(
            {"type": "http", "method": "GET", "path": "/", "asgi": {"spec_version": "2.4"}},
            receive,
            send,
        )
    except Exception as exc:
        assert type(exc).__name__ == "ClientDisconnect"
    else:
        raise AssertionError("ASGI send disconnect was not surfaced")


async def test_consumer_cancel_does_not_cancel_producer() -> None:
    session_id = "sse-detach"
    ScriptedCoordinator.modes[session_id] = "blocked"
    response = await legacy_chat.chat_stream(
        legacy_chat.ChatRequest(message="symbolic-detach", session_id=session_id)
    )
    await _wait_registry_present(session_id)
    run = legacy_chat._ACTIVE_STREAM_RUNS[session_id]
    await _disconnect_asgi_response(response)
    assert run.consumer_attached is False
    assert session_id in legacy_chat._ACTIVE_STREAM_RUNS
    assert ScriptedCoordinator.cancelled[session_id] is False
    ScriptedCoordinator.releases[session_id].set()
    await asyncio.wait_for(
        ScriptedCoordinator.completed.setdefault(session_id, asyncio.Event()).wait(),
        1,
    )
    await _wait_registry_empty()
    assert ScriptedCoordinator.cancelled[session_id] is False
    assert run.queue.empty()
    persisted = ScriptedCoordinator.persisted[session_id]
    assert persisted["assistant_history"] == "symbolic-complete-answer"
    assert persisted["trace_status"] == "complete"
    assert persisted["usage_complete"] is True
    assert len(persisted["provider_attempts"]) == 2
    assert all(
        attempt["usage_source"] == "provider_actual"
        and set(attempt["usage"])
        == {
            "cache_hit_tokens",
            "cache_miss_tokens",
            "output_tokens",
            "total_tokens",
        }
        for attempt in persisted["provider_attempts"]
    )
    assert "client_stream_cancelled" not in persisted["trace_events"]
    print("PASS test_consumer_cancel_does_not_cancel_producer")


async def test_same_session_conflict_is_409() -> None:
    session_id = "sse-conflict"
    ScriptedCoordinator.modes[session_id] = "blocked"
    first = await legacy_chat.chat_stream(
        legacy_chat.ChatRequest(message="symbolic-first", session_id=session_id)
    )
    await _wait_registry_present(session_id)
    assert _event_name(_text(await anext(first.body_iterator))) == "start"
    for submit in (
        lambda: legacy_chat.chat_stream(
            legacy_chat.ChatRequest(message="symbolic-second", session_id=session_id)
        ),
        lambda: legacy_chat.chat_stream_get(
            message="symbolic-second", session_id=session_id, user_id="symbolic"
        ),
    ):
        try:
            await submit()
        except legacy_chat.HTTPException as exc:
            assert exc.status_code == 409
        else:
            raise AssertionError("same-Session duplicate was not rejected")
    assert ScriptedCoordinator.calls[session_id] == 1
    ScriptedCoordinator.releases[session_id].set()
    await _drain(first)
    await _wait_registry_empty()

    get_session_id = "sse-conflict-get-owner"
    ScriptedCoordinator.modes[get_session_id] = "blocked"
    get_first = await legacy_chat.chat_stream_get(
        message="symbolic-get-first",
        session_id=get_session_id,
        user_id="symbolic",
    )
    assert _event_name(_text(await anext(get_first.body_iterator))) == "start"
    try:
        await legacy_chat.chat_stream(
            legacy_chat.ChatRequest(
                message="symbolic-post-second", session_id=get_session_id
            )
        )
    except legacy_chat.HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("POST bypassed an active GET producer")
    assert ScriptedCoordinator.calls[get_session_id] == 1
    ScriptedCoordinator.releases[get_session_id].set()
    await _drain(get_first)
    await _wait_registry_empty()
    print("PASS test_same_session_conflict_is_409")


async def test_producer_exception_is_observed_and_cleaned() -> None:
    session_id = "sse-error"
    ScriptedCoordinator.modes[session_id] = "error"
    response = await legacy_chat.chat_stream(
        legacy_chat.ChatRequest(message="symbolic-error", session_id=session_id)
    )
    await _wait_registry_present(session_id)
    producer = legacy_chat._ACTIVE_STREAM_RUNS[session_id].task
    assert producer is not None
    original_log_error = legacy_chat.logger.error
    observed_errors: List[str] = []
    legacy_chat.logger.error = (
        lambda message, *args, **_kwargs: observed_errors.append(message % args)
    )
    ScriptedCoordinator.releases.setdefault(session_id, asyncio.Event()).set()
    try:
        try:
            await asyncio.wait_for(_drain(response), 1)
        except RuntimeError as exc:
            assert str(exc) == "symbolic-producer-failure"
        else:
            raise AssertionError("producer exception was not exposed to the consumer")
    finally:
        await asyncio.sleep(0)
        legacy_chat.logger.error = original_log_error
    await asyncio.sleep(0)
    assert producer.done()
    assert isinstance(producer.exception(), RuntimeError)
    assert observed_errors == [
        f"Background chat producer failed for session {session_id}"
    ]
    await _wait_registry_empty()
    ScriptedCoordinator.modes[session_id] = "normal"
    retry = await legacy_chat.chat_stream(
        legacy_chat.ChatRequest(message="symbolic-retry", session_id=session_id)
    )
    assert _event_name((await _drain(retry))[-2]) == "done"
    await _wait_registry_empty()
    print("PASS test_producer_exception_is_observed_and_cleaned")


async def test_detached_run_releases_only_after_completion() -> None:
    session_id = "sse-release"
    ScriptedCoordinator.modes[session_id] = "blocked"
    response = await legacy_chat.chat_stream(
        legacy_chat.ChatRequest(message="symbolic-release", session_id=session_id)
    )
    await _wait_registry_present(session_id)
    await _disconnect_asgi_response(response)
    try:
        await legacy_chat.chat_stream(
            legacy_chat.ChatRequest(message="symbolic-too-soon", session_id=session_id)
        )
    except legacy_chat.HTTPException as exc:
        assert exc.status_code == 409
    else:
        raise AssertionError("detached run released before producer completion")
    ScriptedCoordinator.releases[session_id].set()
    await asyncio.wait_for(
        ScriptedCoordinator.completed.setdefault(session_id, asyncio.Event()).wait(),
        1,
    )
    await _wait_registry_empty()
    ScriptedCoordinator.modes[session_id] = "normal"
    next_response = await legacy_chat.chat_stream(
        legacy_chat.ChatRequest(message="symbolic-after", session_id=session_id)
    )
    assert _event_name((await _drain(next_response))[-2]) == "done"
    await _wait_registry_empty()
    print("PASS test_detached_run_releases_only_after_completion")


async def main() -> None:
    coordinator_module = ModuleType("app.runtime.coordinator")
    coordinator_module.RuntimeCoordinator = ScriptedCoordinator
    original = sys.modules.get("app.runtime.coordinator")
    sys.modules["app.runtime.coordinator"] = coordinator_module
    try:
        await test_normal_order_and_transport_flush()
        await test_consumer_cancel_does_not_cancel_producer()
        await test_same_session_conflict_is_409()
        await test_producer_exception_is_observed_and_cleaned()
        await test_detached_run_releases_only_after_completion()
        assert not legacy_chat._ACTIVE_STREAM_RUNS
        print("Chat SSE disconnect lifecycle: PASS (5 checks; Provider requests: 0)")
    finally:
        for run in list(legacy_chat._ACTIVE_STREAM_RUNS.values()):
            if run.task is not None and not run.task.done():
                run.task.cancel()
        await asyncio.sleep(0)
        legacy_chat._ACTIVE_STREAM_RUNS.clear()
        if original is None:
            sys.modules.pop("app.runtime.coordinator", None)
        else:
            sys.modules["app.runtime.coordinator"] = original
        TEMP_DIR.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
