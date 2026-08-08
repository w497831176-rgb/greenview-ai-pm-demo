"""Offline deterministic contract for the trusted catalog/hot orchestration.

The script uses a new temporary SQLite database, symbolic identities from the
code-reviewed manifest and no Provider/model/network calls.  It never reads or
changes production configuration, business data or RuntimeRelease state.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict


ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-target2-trusted-catalog-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
for _key in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY"):
    os.environ[_key] = ""


from db.property_db import (  # noqa: E402
    _get_conn,
    create_agent,
    create_runtime_release,
    get_current_runtime_release,
    get_runtime_release,
    init_db,
    list_agents,
    list_runtime_releases,
    rollback_runtime_release,
    set_agent_knowledge_bindings,
    set_agent_skills,
    set_agent_tools,
    update_agent,
)


init_db()


import app.runtime.capability_catalog as capability_catalog_module  # noqa: E402
from app.runtime.capability_catalog import (  # noqa: E402
    CapabilityCatalogError,
    list_trusted_capabilities,
    set_trusted_agent_bindings,
    set_trusted_capability_enabled,
)
from app.runtime.contracts import ToolEffect  # noqa: E402
from app.runtime.mcp_executor import invoke_confirmed_write  # noqa: E402
from app.runtime.release_compiler import (  # noqa: E402
    _compile_graph,
    preview_runtime_release,
    publish_current_runtime_config,
    validate_release_graph,
)
from app.runtime.snapshot_resolver import resolve_snapshot  # noqa: E402
from app.runtime.agent_factory import validate_agent_binding_isolation  # noqa: E402
from app.runtime.tool_gateway import ToolGateway, ToolPolicyError  # noqa: E402


RESULTS: list[Dict[str, Any]] = []


def check(name: str, condition: bool, detail: Any = None) -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail!r}")
    RESULTS.append({"name": name, "passed": True, "detail": detail})


def expect_catalog_error(action: Callable[[], Any]) -> str:
    try:
        action()
    except CapabilityCatalogError as exc:
        return str(exc)
    raise AssertionError("CapabilityCatalogError was not raised")


def _prepare_isolated_catalog_fixture() -> None:
    """Create only a small manifest-backed subset with production-stable IDs."""

    skill_reference = Path(TEMP_DIR.name) / "skills" / "2" / "references" / "rule.txt"
    skill_reference.parent.mkdir(parents=True, exist_ok=True)
    skill_reference.write_text("symbolic-reviewed-reference", encoding="utf-8")
    mcp_package = Path(TEMP_DIR.name) / "mcp_packages" / "fixture-general"
    mcp_package.mkdir(parents=True, exist_ok=True)
    mcp_entrypoint = mcp_package / "server.py"
    mcp_entrypoint.write_text("SYMBOLIC_VALUE = 1\n", encoding="utf-8")

    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM agent_tools")
    cursor.execute("DELETE FROM agent_skills")
    cursor.execute("DELETE FROM agent_knowledge_bindings")
    cursor.execute("DELETE FROM agent_knowledge_scopes")
    cursor.execute("DELETE FROM mcp_tools")
    cursor.execute("DELETE FROM mcp_servers")
    cursor.execute(
        """
        INSERT OR REPLACE INTO skills (
            id, name, description, instructions, category, enabled,
            trigger_condition, skill_metadata, storage_path, model_id,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            13,
            "symbolic-general-skill",
            "symbolic",
            "symbolic",
            "test",
            1,
            "",
            json.dumps({"version": "1.0.0"}),
            "",
            None,
            "2026-08-08 00:00",
            "2026-08-08 00:00",
        ),
    )
    cursor.execute(
        """
        INSERT OR REPLACE INTO knowledge_docs (
            id, title, content, category, source_type, is_indexed,
            index_status, chunk_count, chunk_size, chunk_overlap, split_strategy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (123, "symbolic-general-doc", "symbolic", "test", "business", 1, "ready", 1, 128, 0, "fixed"),
    )
    cursor.execute(
        "UPDATE knowledge_docs SET is_indexed = 1, index_status = 'ready' WHERE id = 1"
    )
    cursor.executemany(
        """
        INSERT INTO mcp_servers (
            id, name, command, args, env, description, enabled, is_builtin,
            source_type, runtime_type, package_path, detected_entrypoint,
            install_status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (8, "fixture-property-mcp", "python", "[]", "{}", "symbolic", 1, 1, "reviewed", "python", None, None, "ready", "2026", "2026"),
            (500, "fixture-general-mcp", "python", json.dumps([str(mcp_entrypoint)]), "{}", "symbolic", 1, 0, "reviewed", "python", str(mcp_package), "server.py", "ready", "2026", "2026"),
        ],
    )
    tool_metadata = json.dumps({"result_contract": {"success_statuses": ["success"]}})
    cursor.executemany(
        """
        INSERT INTO mcp_tools (
            id, server_id, name, description, input_schema, tool_metadata
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (1, 8, "fixture_property_read", "symbolic", "{}", tool_metadata),
            (2, 8, "fixture_property_read_two", "symbolic", "{}", tool_metadata),
            (3423, 500, "fixture_general_read", "symbolic", "{}", tool_metadata),
        ],
    )
    conn.commit()
    conn.close()

    if not any(item.get("agent_id") == "儿童教育Agent" for item in list_agents()):
        create_agent(
            agent_id="儿童教育Agent",
            name="儿童教育Agent",
            description="symbolic",
            instructions="symbolic",
            category="vertical",
            domain_scope="isolated_general",
            enabled=True,
        )
    for agent in list_agents():
        agent_id = str(agent.get("agent_id") or "")
        if agent_id in {"router", "maintenance", "儿童教育Agent"}:
            expected_scope = (
                "isolated_general" if agent_id == "儿童教育Agent" else "property"
            )
            update_agent(
                int(agent["id"]),
                domain_scope=expected_scope,
                enabled=True,
            )
        else:
            update_agent(int(agent["id"]), enabled=False)
        set_agent_skills(agent_id, [])
        set_agent_tools(agent_id, [])
        set_agent_knowledge_bindings(agent_id, [])

    # This isolated database is its own code-reviewed fixture admission. Patch
    # only the in-process manifest hashes to the deterministic fixture rows;
    # production manifest constants remain untouched.
    for entry in capability_catalog_module._MANIFEST_ENTRIES:
        capability_type = str(entry["capability_type"])
        stable_id = entry["stable_id"]
        row = capability_catalog_module._runtime_object(
            capability_type,
            stable_id,
        )
        if row is not None:
            entry["reviewed_content_hash"] = capability_catalog_module._safe_hash(
                capability_type,
                row,
            )
            artifact = capability_catalog_module._artifact_state(
                capability_type,
                row,
            )
            if artifact.get("hash") is not None:
                entry["reviewed_artifact_hash"] = artifact["hash"]

    set_trusted_agent_bindings(
        "maintenance",
        [2],
        [1],
        mcp_server_ids=[8],
        system_tool_ids=[],
    )
    set_trusted_agent_bindings(
        "儿童教育Agent",
        [13],
        [123],
        mcp_server_ids=[500],
        system_tool_ids=[],
    )


def _find_agent(config: Dict[str, Any], agent_id: str) -> Dict[str, Any]:
    return next(item for item in config.get("agents") or [] if item.get("agent_id") == agent_id)


def test_catalog_and_domain_boundaries() -> None:
    catalog_before = list_trusted_capabilities()
    counts_before = dict(catalog_before["raw_counts"])
    listed_ids = {
        (item["capability_type"], str(item["stable_id"]))
        for item in catalog_before["items"]
    }
    check("trusted catalog lists reviewed identities", ("skill", "2") in listed_ids)
    check("unregistered system Tool is explicit", catalog_before["system_tool_model"] == "not_present")

    before_general = set_trusted_agent_bindings(
        "儿童教育Agent", [13], [123], mcp_server_ids=[500], system_tool_ids=[]
    )
    error = expect_catalog_error(
        lambda: set_trusted_agent_bindings(
            "儿童教育Agent", [2], [123], mcp_server_ids=[500], system_tool_ids=[]
        )
    )
    after_general = set_trusted_agent_bindings(
        "儿童教育Agent", [13], [123], mcp_server_ids=[500], system_tool_ids=[]
    )
    check("property Skill to C fails without partial mutation", before_general == after_general, error)
    check(
        "property RAG to C fails",
        "cross-domain" in expect_catalog_error(
            lambda: set_trusted_agent_bindings(
                "儿童教育Agent", [13], [1], mcp_server_ids=[500], system_tool_ids=[]
            )
        ),
    )
    check(
        "property MCP to C fails",
        "cross-domain" in expect_catalog_error(
            lambda: set_trusted_agent_bindings(
                "儿童教育Agent", [13], [123], mcp_server_ids=[8], system_tool_ids=[]
            )
        ),
    )
    check(
        "unknown and system Tool bindings fail closed",
        bool(expect_catalog_error(lambda: set_trusted_agent_bindings("maintenance", [999999], [], system_tool_ids=[])))
        and bool(expect_catalog_error(lambda: set_trusted_agent_bindings("maintenance", [], [], system_tool_ids=["x"]))),
    )
    check("catalog object counts unchanged by bindings", list_trusted_capabilities()["raw_counts"] == counts_before)


def test_compile_publish_pin_and_rollback() -> None:
    baseline = publish_current_runtime_config(created_by="target2-contract")
    check("baseline trusted release publishes", baseline.get("published") is True, baseline)
    release_one = baseline["release"]
    snapshot_old = resolve_snapshot("target2-old-session")
    old_hash = snapshot_old.snapshot_hash
    old_release_id = snapshot_old.release_id

    tamper_cases = (
        ("agent", "maintenance", "agents", "agent_id", "maintenance", "instructions"),
        ("skill", 2, "skills", "id", 2, "instructions"),
        ("knowledge", 1, "knowledge_docs", "id", 1, "content"),
        ("mcp_server", 8, "mcp_servers", "id", 8, "command"),
        ("mcp_tool", 1, "mcp_tools", "id", 1, "input_schema"),
    )
    for capability_type, stable_id, table, key_column, key_value, field in tamper_cases:
        conn = _get_conn()
        row = conn.execute(
            f"SELECT {field} AS value FROM {table} WHERE {key_column} = ?",
            (key_value,),
        ).fetchone()
        original = row["value"]
        conn.execute(
            f"UPDATE {table} SET {field} = ? WHERE {key_column} = ?",
            (f"{original or ''}__symbolic_tamper__", key_value),
        )
        conn.commit()
        conn.close()
        catalog_after_tamper = list_trusted_capabilities()
        preview_after_tamper = preview_runtime_release(
            created_by="target2-contract-tamper"
        )
        tamper_blocked = (
            any(
                item.get("capability_type") == capability_type
                and str(item.get("stable_id")) == str(stable_id)
                for item in catalog_after_tamper.get("errors") or []
            )
            and preview_after_tamper.get("can_publish") is False
            and get_current_runtime_release()["release_id"] == old_release_id
        )
        conn = _get_conn()
        conn.execute(
            f"UPDATE {table} SET {field} = ? WHERE {key_column} = ?",
            (original, key_value),
        )
        conn.commit()
        conn.close()
        check(
            f"same-ID {capability_type} implementation tamper fails closed",
            tamper_blocked,
            preview_after_tamper.get("validation"),
        )

    artifact_tamper_cases = (
        (
            "skill",
            2,
            Path(TEMP_DIR.name) / "skills" / "2" / "references" / "rule.txt",
        ),
        (
            "mcp_server",
            500,
            Path(TEMP_DIR.name)
            / "mcp_packages"
            / "fixture-general"
            / "server.py",
        ),
    )

    mcp_package = (
        Path(TEMP_DIR.name) / "mcp_packages" / "fixture-general"
    )
    mcp_server = capability_catalog_module._runtime_object("mcp_server", 500)
    baseline_mcp_hash = capability_catalog_module._artifact_state(
        "mcp_server", mcp_server
    )["hash"]
    deployment_env = mcp_package / ".env"
    deployment_env_local = mcp_package / ".env.local"
    deployment_env.write_text("SYMBOLIC_SECRET=one\n", encoding="utf-8")
    deployment_env_local.write_text("SYMBOLIC_SECRET=two\n", encoding="utf-8")
    env_present_hash = capability_catalog_module._artifact_state(
        "mcp_server", mcp_server
    )["hash"]
    deployment_env.write_text("SYMBOLIC_SECRET=changed\n", encoding="utf-8")
    deployment_env_local.write_text(
        "SYMBOLIC_SECRET=also_changed\n", encoding="utf-8"
    )
    env_changed_hash = capability_catalog_module._artifact_state(
        "mcp_server", mcp_server
    )["hash"]
    deployment_env.unlink()
    deployment_env_local.unlink()
    env_missing_hash = capability_catalog_module._artifact_state(
        "mcp_server", mcp_server
    )["hash"]
    check(
        "MCP artifact hash ignores deployment env presence and content",
        len(
            {
                baseline_mcp_hash,
                env_present_hash,
                env_changed_hash,
                env_missing_hash,
            }
        )
        == 1,
    )

    for capability_type, stable_id, artifact_path in artifact_tamper_cases:
        original = artifact_path.read_bytes()
        artifact_path.write_bytes(original + b"\nSYMBOLIC_TAMPER")
        catalog_after_tamper = list_trusted_capabilities()
        preview_after_tamper = preview_runtime_release(
            created_by="target2-contract-artifact-tamper"
        )
        blocked = (
            any(
                item.get("capability_type") == capability_type
                and str(item.get("stable_id")) == str(stable_id)
                for item in catalog_after_tamper.get("errors") or []
            )
            and preview_after_tamper.get("can_publish") is False
            and get_current_runtime_release()["release_id"] == old_release_id
        )
        artifact_path.write_bytes(original)
        check(
            f"same-path {capability_type} external artifact tamper fails closed",
            blocked,
            preview_after_tamper.get("validation"),
        )

    set_trusted_capability_enabled("skill", 2, False)
    current_before_publish = get_current_runtime_release()
    preview = preview_runtime_release(created_by="target2-contract")
    check(
        "Draft toggle produces valid Diff without changing current",
        preview["can_publish"] is True
        and current_before_publish["release_id"] == old_release_id
        and resolve_snapshot("target2-old-session").snapshot_hash == old_hash,
        preview,
    )
    release_two_result = publish_current_runtime_config(created_by="target2-contract")
    release_two = release_two_result["release"]
    snapshot_new = resolve_snapshot("target2-new-session")
    check(
        "publish affects new Session only",
        release_two_result.get("published") is True
        and snapshot_new.release_id == release_two["release_id"]
        and snapshot_new.snapshot_hash != old_hash
        and resolve_snapshot("target2-old-session").release_id == old_release_id,
    )
    rollback_runtime_release(release_one["release_id"])
    snapshot_rollback = resolve_snapshot("target2-after-rollback")
    release_two_history = get_runtime_release(release_two["release_id"])
    check(
        "rollback restores target for new Session and keeps history immutable",
        snapshot_rollback.release_id == release_one["release_id"]
        and snapshot_rollback.snapshot_hash == release_one["config_hash"]
        and release_two_history["config_hash"] == release_two["config_hash"]
        and len(list_runtime_releases()) == 2,
    )


async def test_compile_gateway_execution_fail_closed() -> None:
    graph, policies = _compile_graph()
    valid = validate_release_graph(graph, policies)
    check("trusted graph validates", valid["valid"] is True, valid)

    legacy_snapshot = copy.deepcopy(graph)
    legacy_c = copy.deepcopy(_find_agent(legacy_snapshot, "儿童教育Agent"))
    legacy_c["agent_id"] = "乱七八糟agent"
    legacy_c["name"] = "symbolic-legacy-c"
    legacy_c["enabled"] = True
    legacy_c["skill_ids"] = []
    legacy_c["knowledge_doc_ids"] = []
    legacy_c["mcp_server_names"] = ["fixture-untrusted-legacy-mcp"]
    legacy_snapshot["agents"].append(legacy_c)
    legacy_snapshot["mcp_servers"].append(
        {
            "server_id": 499,
            "name": "fixture-untrusted-legacy-mcp",
            "enabled": True,
            "domain_scope": "isolated_general",
            "tools": [],
        }
    )
    selected_b_scope = validate_agent_binding_isolation(
        legacy_snapshot,
        "maintenance",
        expected_scope="property",
    )
    try:
        validate_agent_binding_isolation(
            legacy_snapshot,
            "乱七八糟agent",
            expected_scope="isolated_general",
        )
    except ValueError:
        invalid_selected_c_blocked = True
    else:
        invalid_selected_c_blocked = False
    check(
        "unrelated legacy C drift cannot break frozen B but selected C fails",
        selected_b_scope == "property" and invalid_selected_c_blocked,
    )

    c_agent = _find_agent(graph, "儿童教育Agent")
    invalid_skill = copy.deepcopy(graph)
    _find_agent(invalid_skill, "儿童教育Agent")["skill_ids"] = [2]
    check(
        "compiler rejects property Skill bound to C",
        not validate_release_graph(invalid_skill, policies)["valid"],
    )
    invalid_rag = copy.deepcopy(graph)
    _find_agent(invalid_rag, "儿童教育Agent")["knowledge_doc_ids"] = [1]
    check(
        "compiler rejects property RAG bound to C",
        not validate_release_graph(invalid_rag, policies)["valid"],
    )
    invalid_mcp = copy.deepcopy(graph)
    _find_agent(invalid_mcp, "儿童教育Agent")["mcp_server_names"] = ["fixture-property-mcp"]
    check(
        "compiler rejects property MCP bound to C",
        not validate_release_graph(invalid_mcp, policies)["valid"],
    )
    duplicate_mcp_name = copy.deepcopy(graph)
    duplicate_mcp_name["mcp_servers"][1]["name"] = duplicate_mcp_name[
        "mcp_servers"
    ][0]["name"]
    check(
        "compiler rejects duplicate MCP binding names",
        not validate_release_graph(duplicate_mcp_name, policies)["valid"],
    )
    missing_domain = copy.deepcopy(graph)
    missing_domain["skills"][0].pop("domain_scope", None)
    check(
        "compiler rejects missing structured domain",
        not validate_release_graph(missing_domain, policies)["valid"],
    )
    missing_version_hash = copy.deepcopy(graph)
    missing_version_hash["skills"][0].pop("catalog_version", None)
    missing_version_hash["skills"][0].pop("content_hash", None)
    check(
        "compiler rejects capability without version or content hash",
        not validate_release_graph(missing_version_hash, policies)["valid"],
    )
    unknown_capability = copy.deepcopy(graph)
    unknown_skill = copy.deepcopy(unknown_capability["skills"][0])
    unknown_skill["skill_id"] = 999999
    unknown_capability["skills"].append(unknown_skill)
    check(
        "compiler rejects unknown catalog identity",
        not validate_release_graph(unknown_capability, policies)["valid"],
    )
    write_graph = copy.deepcopy(graph)
    write_graph["mcp_servers"][0]["tools"][0]["policy"]["effect"] = ToolEffect.CREATE.value
    check(
        "compiler rejects write MCP",
        not validate_release_graph(write_graph, policies)["valid"],
    )

    gateway = ToolGateway(graph)
    try:
        gateway.write_policy("fixture-property-mcp", "fixture_property_read", "maintenance")
    except ToolPolicyError:
        gateway_blocked = True
    else:
        gateway_blocked = False
    try:
        await invoke_confirmed_write(
            graph,
            "maintenance",
            "fixture-property-mcp",
            "fixture_property_read",
            {},
        )
    except PermissionError:
        execution_blocked = True
    else:
        execution_blocked = False
    check("Gateway and execution reject MCP writes", gateway_blocked and execution_blocked)


async def test_mutating_api_is_gone() -> None:
    from fastapi import HTTPException
    from app.agents import (
        AgentCreate,
        AgentUpdate,
        create_agent as create_agent_api,
        delete_agent as delete_agent_api,
        update_agent as update_agent_api,
    )
    from app.badcases import (
        accept_capability_gap_endpoint,
        apply_capability_gap_draft,
        apply_knowledge_draft,
        apply_skill_prompt_draft,
        publish_knowledge_draft,
        publish_skill_prompt_draft_endpoint,
    )
    from app.knowledge import (
        KnowledgeDocCreate,
        KnowledgeDocUpdate,
        RetrievalSettingsUpdate,
        approve_knowledge_draft,
        create_knowledge_doc as create_doc_api,
        delete_knowledge_doc,
        reindex_doc,
        update_knowledge_doc,
        update_retrieval_settings,
    )
    from app.mcp import (
        McpGitImportRequest,
        McpServerCreate,
        McpServerUpdate,
        McpToolPolicyUpdate,
        McpToolRuntimePolicyUpdate,
        create_mcp_server as create_mcp_api,
        delete_mcp_server,
        discover_mcp_server_tools,
        import_mcp_server_from_git,
        update_mcp_server,
        update_mcp_tool_policy,
        update_mcp_tool_runtime_policy,
    )
    from app.runtime.api import (
        KnowledgeBindingRequest,
        RollbackRequest,
        bind_agent_knowledge,
        extension_acceptance,
        publish_existing_release,
        rollback_release,
    )
    from app.skills import (
        ApplyDarwinRequest,
        GitImportRequest,
        SkillCreate,
        SkillMdUpdate,
        SkillUpdate,
        apply_darwin_optimization,
        create_skill as create_skill_api,
        delete_skill,
        delete_skill_file,
        import_skill_from_git,
        import_skill_from_zip,
        rollback_skill_version,
        update_skill,
        update_skill_md,
        upload_skill_file,
    )

    current_before_draft_rollback = get_current_runtime_release()
    draft_id = "rr_draft_target2_contract"
    create_runtime_release(
        release_id=draft_id,
        version=9999,
        config_hash="symbolic-draft-hash",
        config=current_before_draft_rollback["config"],
        validation=current_before_draft_rollback.get("validation") or {},
        parent_release_id=current_before_draft_rollback["release_id"],
        created_by="target2-contract",
        status="draft",
    )
    try:
        await rollback_release(RollbackRequest(release_id=draft_id))
    except HTTPException as exc:
        draft_rollback_status = exc.status_code
    else:
        draft_rollback_status = None
    check(
        "Draft RuntimeRelease rollback API rejects without moving current",
        draft_rollback_status == 409
        and get_current_runtime_release()["release_id"]
        == current_before_draft_rollback["release_id"],
        draft_rollback_status,
    )

    calls: list[tuple[str, Callable[[], Awaitable[Any]]]] = [
        ("agent.create", lambda: create_agent_api(AgentCreate(name="symbolic"))),
        ("agent.implementation.update", lambda: update_agent_api("maintenance", AgentUpdate(name="symbolic"))),
        ("agent.delete", lambda: delete_agent_api("maintenance")),
        ("skill.create", lambda: create_skill_api(SkillCreate(name="symbolic"))),
        ("skill.import_git", lambda: import_skill_from_git(GitImportRequest(git_url="symbolic"))),
        ("skill.import_zip", lambda: import_skill_from_zip(None, "", True)),
        ("skill.implementation.update", lambda: update_skill(2, SkillUpdate(name="symbolic"))),
        ("skill.skill_md.update", lambda: update_skill_md(2, SkillMdUpdate())),
        ("skill.implementation.rollback", lambda: rollback_skill_version(2, "symbolic")),
        ("skill.file.upload", lambda: upload_skill_file(2, None, "")),
        ("skill.file.delete", lambda: delete_skill_file(2, "symbolic")),
        ("skill.darwin.apply", lambda: apply_darwin_optimization(2, ApplyDarwinRequest(suggested_prompt="symbolic"))),
        ("skill.delete", lambda: delete_skill(2)),
        ("knowledge.create", lambda: create_doc_api(KnowledgeDocCreate(title="symbolic", content="symbolic"))),
        ("knowledge.implementation.update", lambda: update_knowledge_doc(1, KnowledgeDocUpdate(title="symbolic", content="symbolic"))),
        ("knowledge.delete", lambda: delete_knowledge_doc(1)),
        ("knowledge.reindex", lambda: reindex_doc(1)),
        ("knowledge.draft.apply", lambda: approve_knowledge_draft(1)),
        ("retrieval_policy.update", lambda: update_retrieval_settings(RetrievalSettingsUpdate())),
        ("mcp.create", lambda: create_mcp_api(McpServerCreate(name="symbolic"))),
        ("mcp.import_git", lambda: import_mcp_server_from_git(McpGitImportRequest(git_url="symbolic"))),
        ("mcp.implementation.update", lambda: update_mcp_server("8", McpServerUpdate(name="symbolic"))),
        ("mcp.tool.policy.update", lambda: update_mcp_tool_policy("8", 1, McpToolPolicyUpdate(effect="read"))),
        ("mcp.tool.runtime_policy.update", lambda: update_mcp_tool_runtime_policy("8", 1, McpToolRuntimePolicyUpdate(effect="read"))),
        ("mcp.tool.discover", lambda: discover_mcp_server_tools("8")),
        ("mcp.delete", lambda: delete_mcp_server("8")),
        ("badcase.knowledge.alias_apply", lambda: publish_knowledge_draft(1, 1)),
        ("badcase.skill.alias_apply", lambda: publish_skill_prompt_draft_endpoint(1, 1)),
        ("badcase.gap.alias_apply", lambda: accept_capability_gap_endpoint(1, 1)),
        ("badcase.knowledge.apply", lambda: apply_knowledge_draft(1, 1)),
        ("badcase.skill.apply", lambda: apply_skill_prompt_draft(1, 1)),
        ("badcase.gap.apply", lambda: apply_capability_gap_draft(1, 1)),
        ("runtime.stale_release.publish", lambda: publish_existing_release("symbolic")),
        ("runtime.fragmented_binding.publish", lambda: bind_agent_knowledge("maintenance", KnowledgeBindingRequest(knowledge_doc_ids=[1], publish=True))),
        ("runtime.extension_acceptance", extension_acceptance),
    ]
    statuses: Dict[str, int] = {}
    for name, call in calls:
        try:
            await call()
        except HTTPException as exc:
            statuses[name] = exc.status_code
    check(
        "all supply-chain mutation and legacy direct-publish APIs return 410",
        len(statuses) == len(calls) and set(statuses.values()) == {410},
        statuses,
    )


async def main() -> None:
    _prepare_isolated_catalog_fixture()
    test_catalog_and_domain_boundaries()
    await test_compile_gateway_execution_fail_closed()
    test_compile_publish_pin_and_rollback()
    await test_mutating_api_is_gone()
    for result in RESULTS:
        print(f"PASS {result['name']}")
    print(f"PASS target2_trusted_catalog_tests={len(RESULTS)}")


if __name__ == "__main__":
    asyncio.run(main())
