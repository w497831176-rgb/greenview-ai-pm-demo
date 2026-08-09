"""Deterministic target-2 hot binding and session snapshot lifecycle checks.

The test owns a temporary SQLite database, never invokes a model or MCP
process, and never touches the production RuntimeRelease pointer.  Symbolic
catalog fixtures are admitted before the object-count baseline; the behavior
under test only binds, publishes, unbinds, publishes, and rolls back those
already-existing fixture objects.
"""

from __future__ import annotations

import asyncio
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(
    os.getenv("TARGET2_REPO_ROOT") or Path(__file__).resolve().parents[1]
).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-v182-target2-hot-binding-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
os.environ["RUNTIME_ENGINE"] = "v18"
# Construction of an Agno model object is part of Agent assembly.  The value
# below is deliberately non-secret and the test never invokes that object.
os.environ["DEEPSEEK_API_KEY"] = "target2-test-not-used"

from db.property_db import (  # noqa: E402
    count_runtime_releases,
    create_agent,
    create_knowledge_doc,
    create_mcp_server,
    create_skill,
    get_current_runtime_release,
    get_evidence_ledger,
    get_runtime_release,
    init_db,
    list_agents,
    list_knowledge_docs,
    list_mcp_servers,
    list_mcp_tools,
    list_skills,
    rollback_runtime_release,
    save_mcp_tool,
)


TARGET_AGENT_ID = "target2-symbolic-agent"
SKILL_TRIGGER = "SYMBOLIC_TRIGGER"
SERVER_NAME = "target2-symbolic-read-server"
TOOL_NAME = "target2_symbolic_read"
QUERY = SKILL_TRIGGER


def _target_agent(config: dict[str, Any]) -> dict[str, Any]:
    return next(
        item
        for item in config.get("agents") or []
        if item.get("agent_id") == TARGET_AGENT_ID
    )


def _catalog_counts() -> dict[str, int]:
    return {
        "agents": len(list_agents()),
        "skills": len(list_skills()),
        "knowledge": len(list_knowledge_docs()),
        "mcp_servers": len(list_mcp_servers()),
        "mcp_tools": len(list_mcp_tools()),
    }


def _install_symbolic_catalog_fixtures() -> tuple[int, int]:
    agent = create_agent(
        agent_id=TARGET_AGENT_ID,
        name="Target 2 Symbolic Agent",
        description="Deterministic hot-binding lifecycle fixture",
        instructions="Use only capabilities bound in the immutable snapshot.",
        category="vertical",
        domain_scope="property",
        enabled=True,
    )
    assert agent["agent_id"] == TARGET_AGENT_ID

    skill = create_skill(
        name="Target 2 Symbolic Skill",
        description="Activated only by the symbolic lifecycle marker",
        instructions="SYMBOLIC_SKILL_PAYLOAD",
        category="target2-test",
        enabled=True,
        trigger_condition=SKILL_TRIGGER,
        skill_metadata={
            "version": "target2-test-v1",
            "positive_triggers": [SKILL_TRIGGER],
        },
    )
    document = create_knowledge_doc(
        title="Target 2 Symbolic Evidence",
        content=(
            "SYMBOLIC_TRIGGER is immutable evidence for the target-2 "
            "hot-binding lifecycle."
        ),
        category="target2-test",
        source_type="business",
        index_status="indexed",
    )
    server = create_mcp_server(
        name=SERVER_NAME,
        command="target2-never-executed",
        args=[],
        env={},
        description="Symbolic read-only MCP fixture",
        enabled=True,
        source_type="manual",
        runtime_type="stdio",
        install_status="ready",
    )
    tool = save_mcp_tool(
        int(server["id"]),
        TOOL_NAME,
        description="Return a symbolic read-only value",
        input_schema={"type": "object", "properties": {}},
        tool_metadata={
            "effect": "read",
            "effect_source": "operator_declared",
            "risk_level": "L1",
            "execution_mode": "auto_preinvoke",
            "natural_language_intents": [],
            "trigger_keywords": [],
            "trigger_mode": "any",
            "argument_bindings": {},
            "result_contract": {
                "success_statuses": ["success"],
                "non_success_statuses": ["empty", "failed"],
                "claim_rule": "Only explicit success may be described as success.",
            },
        },
    )
    assert tool["name"] == TOOL_NAME
    return int(skill["id"]), int(document["id"])


async def _set_draft_bindings(
    *,
    skill_ids: list[int],
    knowledge_doc_ids: list[int],
    server_names: list[str],
) -> None:
    # Exercise the same public handler used by the platform management UI.
    from app.agents import AgentUpdate, update_agent

    result = await update_agent(
        TARGET_AGENT_ID,
        AgentUpdate(
            skill_ids=skill_ids,
            knowledge_doc_ids=knowledge_doc_ids,
            tool_names=server_names,
        ),
    )
    agent = result["agent"]
    assert agent["skill_ids"] == skill_ids
    assert agent["knowledge_doc_ids"] == knowledge_doc_ids
    assert [item["tool_name"] for item in agent["tools"]] == server_names


def _assert_snapshot_identity(actual: Any, expected: Any) -> None:
    assert actual.release_id == expected.release_id
    assert actual.snapshot_hash == expected.snapshot_hash
    assert actual.config == expected.config


def _assert_binding_state(
    snapshot: Any,
    *,
    skill_ids: list[int],
    knowledge_doc_ids: list[int],
    server_names: list[str],
) -> None:
    agent = _target_agent(snapshot.config)
    assert agent["skill_ids"] == skill_ids
    assert agent["knowledge_doc_ids"] == knowledge_doc_ids
    assert agent["mcp_server_names"] == server_names


def _assert_preview_binding_change(
    preview: dict[str, Any],
    *,
    direction: str,
    skill_id: int,
    knowledge_doc_id: int,
) -> None:
    assert preview["has_changes"] is True
    assert preview["can_publish"] is True, preview["validation"]
    assert preview["persisted"] is False
    changed = next(
        item
        for item in preview["diff"]["agents"]
        if item["agent_id"] == TARGET_AGENT_ID
    )
    capabilities = changed["capabilities"]
    key = "added" if direction == "bind" else "removed"
    assert [item["skill_id"] for item in capabilities["skills"][key]] == [
        skill_id
    ]
    assert [
        item["knowledge_doc_id"]
        for item in capabilities["knowledge"][key]
    ] == [knowledge_doc_id]
    assert [item["name"] for item in capabilities["mcp_servers"][key]] == [
        SERVER_NAME
    ]


def _assemble_capabilities(snapshot: Any) -> dict[str, Any]:
    """Assemble capabilities without invoking a model, MCP process, or DB write."""

    from app.runtime import mcp_executor
    from app.runtime.agent_factory import (
        build_agent_from_snapshot,
        vertical_agent_cards,
    )
    from app.runtime.citation_renderer import build_evidence_set
    from app.runtime.coordinator import _results_from_snapshot

    build = build_agent_from_snapshot(
        snapshot,
        TARGET_AGENT_ID,
        QUERY,
        tools=[],
    )
    agent_config = _target_agent(snapshot.config)
    allowed_document_ids = {
        int(item) for item in agent_config.get("knowledge_doc_ids") or []
    }
    knowledge_versions = {
        int(item["knowledge_doc_id"]): item
        for item in snapshot.config.get("knowledge") or []
    }
    results, used_snapshot = _results_from_snapshot(
        QUERY,
        [],
        knowledge_versions,
        allowed_document_ids,
        5,
        0.0,
    )
    evidence = build_evidence_set(
        QUERY,
        results,
        knowledge_versions=knowledge_versions,
        allowed_document_ids=allowed_document_ids,
        retrieval_status=(
            "completed_snapshot_fallback" if used_snapshot else "completed"
        ),
    )

    class _NoProcessToolkit:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    original_toolkit = mcp_executor.GovernedMCPTools
    mcp_executor.GovernedMCPTools = _NoProcessToolkit
    try:
        toolkits = mcp_executor.build_model_native_read_tools(
            snapshot.config,
            TARGET_AGENT_ID,
            QUERY,
        )
    finally:
        mcp_executor.GovernedMCPTools = original_toolkit

    card = next(
        item
        for item in vertical_agent_cards(snapshot.config)
        if item["agent_id"] == TARGET_AGENT_ID
    )
    return {
        "build": build,
        "evidence": evidence,
        "toolkits": toolkits,
        "card": card,
    }


def _persist_trace_projection(snapshot: Any, assembled: dict[str, Any], suffix: str):
    """Persist the real assembly result without fabricating a Tool invocation."""

    from app.runtime.contracts import RunState, RunStatus, RuntimePath
    from app.runtime.evidence_ledger import EvidenceLedger

    trace_id = f"trace_target2_{suffix}"
    state = RunState(
        run_id=f"run_target2_{suffix}",
        trace_id=trace_id,
        session_id=snapshot.session_id,
        snapshot_id=snapshot.snapshot_id,
        path=RuntimePath.CONSULTATION,
        status=RunStatus.COMPLETED,
    )
    state.activated_skills = list(assembled["build"].activated_skills)
    state.retrieval_evidence = assembled["evidence"]
    ledger = EvidenceLedger(
        trace_id=trace_id,
        session_id=snapshot.session_id,
        config_snapshot=snapshot.config,
        release_id=snapshot.release_id,
        config_hash=snapshot.snapshot_hash,
        runtime_path=RuntimePath.CONSULTATION.value,
    )
    ledger.capture_state(state)
    ledger.persist("complete")
    return get_evidence_ledger(trace_id)


def test_bind_publish_unbind_publish_and_session_pinning() -> None:
    from app.runtime.release_compiler import (
        preview_runtime_release,
        publish_current_runtime_config,
    )
    from app.runtime.snapshot_resolver import resolve_snapshot

    skill_id, document_id = _install_symbolic_catalog_fixtures()
    asyncio.run(
        _set_draft_bindings(
            skill_ids=[],
            knowledge_doc_ids=[],
            server_names=[],
        )
    )
    catalog_counts = _catalog_counts()

    baseline_result = publish_current_runtime_config(created_by="target2-test")
    assert baseline_result["published"] is True, baseline_result
    baseline_release = baseline_result["release"]
    baseline_release_id = baseline_release["release_id"]
    baseline_row = copy.deepcopy(get_runtime_release(baseline_release_id))
    release_count_after_baseline = count_runtime_releases()

    session_before_bind = resolve_snapshot("target2-session-before-bind")
    _assert_binding_state(
        session_before_bind,
        skill_ids=[],
        knowledge_doc_ids=[],
        server_names=[],
    )

    # Draft bind: current release and an already-started Session cannot move.
    asyncio.run(
        _set_draft_bindings(
            skill_ids=[skill_id],
            knowledge_doc_ids=[document_id],
            server_names=[SERVER_NAME],
        )
    )
    assert get_current_runtime_release()["release_id"] == baseline_release_id
    assert count_runtime_releases() == release_count_after_baseline
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-before-bind"),
        session_before_bind,
    )
    bind_preview = preview_runtime_release(created_by="target2-test")
    _assert_preview_binding_change(
        bind_preview,
        direction="bind",
        skill_id=skill_id,
        knowledge_doc_id=document_id,
    )
    assert count_runtime_releases() == release_count_after_baseline

    bind_result = publish_current_runtime_config(created_by="target2-test")
    assert bind_result["published"] is True, bind_result
    bound_release = bind_result["release"]
    bound_release_id = bound_release["release_id"]
    bound_row = copy.deepcopy(get_runtime_release(bound_release_id))
    assert count_runtime_releases() == release_count_after_baseline + 1

    session_after_bind = resolve_snapshot("target2-session-after-bind")
    _assert_binding_state(
        session_after_bind,
        skill_ids=[skill_id],
        knowledge_doc_ids=[document_id],
        server_names=[SERVER_NAME],
    )
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-before-bind"),
        session_before_bind,
    )

    bound_assembly = _assemble_capabilities(session_after_bind)
    assert bound_assembly["build"].activated_skills == []
    assert [item.skill_id for item in bound_assembly["build"].bound_skills] == [
        skill_id
    ]
    assert [
        int(item.document_id) for item in bound_assembly["evidence"].items
    ] == [document_id]
    assert len(bound_assembly["toolkits"]) == 1
    assert bound_assembly["toolkits"][0].kwargs[
        "allowed_function_names"
    ] == [TOOL_NAME]
    assert [item["id"] for item in bound_assembly["card"]["skills"]] == [
        skill_id
    ]
    assert bound_assembly["card"]["mcp_tools"] == [SERVER_NAME]
    assert [
        item["id"]
        for item in bound_assembly["card"]["capability_card"][
            "knowledge_docs"
        ]
    ] == [document_id]
    bound_trace = _persist_trace_projection(
        session_after_bind,
        bound_assembly,
        "bound",
    )
    # Binding exposes a candidate to the frozen Agent. It is not a real use
    # until that Agent calls get_skill_instructions in a model run.
    assert bound_trace["ledger"]["activated_skills"] == []
    assert [
        int(item["document_id"])
        for item in bound_trace["ledger"]["retrieval_evidence"]
    ] == [document_id]
    # Availability is not an invocation; Trace must not manufacture one.
    assert bound_trace["ledger"]["tool_invocations"] == []
    assert bound_trace["ledger"]["model_calls"] == []

    # Draft unbind: the bound release and Session remain unchanged until publish.
    asyncio.run(
        _set_draft_bindings(
            skill_ids=[],
            knowledge_doc_ids=[],
            server_names=[],
        )
    )
    assert get_current_runtime_release()["release_id"] == bound_release_id
    assert count_runtime_releases() == release_count_after_baseline + 1
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-after-bind"),
        session_after_bind,
    )
    unbind_preview = preview_runtime_release(created_by="target2-test")
    _assert_preview_binding_change(
        unbind_preview,
        direction="unbind",
        skill_id=skill_id,
        knowledge_doc_id=document_id,
    )

    unbind_result = publish_current_runtime_config(created_by="target2-test")
    assert unbind_result["published"] is True, unbind_result
    unbound_release = unbind_result["release"]
    unbound_release_id = unbound_release["release_id"]
    unbound_row = copy.deepcopy(get_runtime_release(unbound_release_id))
    assert count_runtime_releases() == release_count_after_baseline + 2

    session_after_unbind = resolve_snapshot("target2-session-after-unbind")
    _assert_binding_state(
        session_after_unbind,
        skill_ids=[],
        knowledge_doc_ids=[],
        server_names=[],
    )
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-before-bind"),
        session_before_bind,
    )
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-after-bind"),
        session_after_bind,
    )

    unbound_assembly = _assemble_capabilities(session_after_unbind)
    assert unbound_assembly["build"].activated_skills == []
    assert unbound_assembly["build"].bound_skills == []
    assert unbound_assembly["build"].skill_tool_calls == []
    assert unbound_assembly["evidence"].items == []
    assert unbound_assembly["toolkits"] == []
    assert unbound_assembly["card"]["skills"] == []
    assert unbound_assembly["card"]["mcp_tools"] == []
    assert unbound_assembly["card"]["capability_card"]["knowledge_docs"] == []
    unbound_trace = _persist_trace_projection(
        session_after_unbind,
        unbound_assembly,
        "unbound",
    )
    assert unbound_trace["ledger"]["activated_skills"] == []
    assert unbound_trace["ledger"]["retrieval_evidence"] == []
    assert unbound_trace["ledger"]["tool_invocations"] == []
    assert unbound_trace["ledger"]["model_calls"] == []

    # Rollback moves only the current pointer. It creates no release and never
    # rewrites a pre-existing Session snapshot.
    rollback_runtime_release(bound_release_id)
    assert get_current_runtime_release()["release_id"] == bound_release_id
    assert count_runtime_releases() == release_count_after_baseline + 2
    session_after_rollback = resolve_snapshot("target2-session-after-rollback")
    _assert_binding_state(
        session_after_rollback,
        skill_ids=[skill_id],
        knowledge_doc_ids=[document_id],
        server_names=[SERVER_NAME],
    )
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-before-bind"),
        session_before_bind,
    )
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-after-bind"),
        session_after_bind,
    )
    _assert_snapshot_identity(
        resolve_snapshot("target2-session-after-unbind"),
        session_after_unbind,
    )

    # Release status may change as the pointer moves; immutable content may not.
    for original in (baseline_row, bound_row, unbound_row):
        current = get_runtime_release(original["release_id"])
        assert current["config_hash"] == original["config_hash"]
        assert current["config"] == original["config"]
    assert _catalog_counts() == catalog_counts


def test_current_release_parent_diff_ui_contract() -> None:
    frontend = (REPO_ROOT / "frontend/index.html").read_text(encoding="utf-8")
    assert 'id="runtime-current-parent-diff"' in frontend
    assert "与上一版本比较" in frontend
    assert (
        "/api/runtime/releases/${encodeURIComponent(current.release_id)}/diff?include_details=true"
        in frontend
    )


def main() -> None:
    try:
        init_db()
        test_bind_publish_unbind_publish_and_session_pinning()
        print("PASS test_bind_publish_unbind_publish_and_session_pinning")
        test_current_release_parent_diff_ui_contract()
        print("PASS test_current_release_parent_diff_ui_contract")
    finally:
        TEMP_DIR.cleanup()
    print("Target 2 hot-binding lifecycle contracts passed without model calls.")


if __name__ == "__main__":
    main()
