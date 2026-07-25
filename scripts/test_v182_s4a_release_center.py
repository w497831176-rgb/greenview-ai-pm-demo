"""Deterministic V1.8.2-S4-A RuntimeRelease center contracts."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-v182-s4a-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name
os.environ["RUNTIME_ENGINE"] = "v18"

from app.runtime.contracts import content_hash
from db.property_db import (
    count_runtime_releases,
    create_runtime_release,
    get_current_runtime_release,
    init_db,
    list_runtime_release_summaries,
    next_runtime_release_version,
    publish_runtime_release,
)

init_db()

from app.runtime import api as runtime_api
from app.runtime import release_compiler
from app.runtime.snapshot_resolver import resolve_snapshot


LONG_PROMPT_TAIL = "S4A-FULL-PROMPT-TAIL-MUST-BE-LAZY"


def _base_config() -> dict:
    return {
        "schema_version": "1.0",
        "agents": [
            {
                "agent_id": "router",
                "name": "路由 Agent",
                "description": "选择垂直 Agent",
                "instructions": "只负责路由",
                "category": "router",
                "enabled": True,
                "model_id": "router-model",
                "skill_ids": [],
                "mcp_server_names": [],
                "knowledge_doc_ids": [],
            },
            {
                "agent_id": "maintenance",
                "name": "维修 Agent",
                "description": "处理维修咨询",
                "instructions": ("旧 Prompt " * 40) + LONG_PROMPT_TAIL,
                "category": "vertical",
                "enabled": True,
                "model_id": "model-a",
                "skill_ids": [1],
                "mcp_server_names": ["workorder-server"],
                "knowledge_doc_ids": [10],
            },
        ],
        "skills": [
            {
                "skill_id": 1,
                "name": "维修判断",
                "description": "判断维修问题",
                "version": "1.0.0",
                "enabled": True,
                "trigger_condition": "漏水",
                "content_hash": "skill-hash-1",
                "reference_snapshots": [],
                "instructions_fallback": "维修判断正文",
            }
        ],
        "knowledge": [
            {
                "knowledge_doc_id": 10,
                "title": "维修制度",
                "category": "维修",
                "document_version": "doc-v1",
                "document_hash": "doc-hash-1",
                "index_status": "indexed",
                "chunk_count": 1,
                "chunk_size": 512,
                "chunk_overlap": 64,
                "split_strategy": "auto",
                "chunk_snapshots": [
                    {
                        "chunk_index": 0,
                        "content": "维修制度完整正文不应进入轻量概览",
                        "chunk_hash": "chunk-hash-1",
                    }
                ],
            }
        ],
        "mcp_servers": [
            {
                "server_id": 1,
                "name": "workorder-server",
                "description": "工单查询",
                "enabled": True,
                "is_builtin": True,
                "command": "python",
                "args": ["server.py"],
                "env_keys": ["WORKORDER_TOKEN"],
                "tools": [
                    {
                        "name": "get_work_order",
                        "description": "查询工单",
                        "input_schema": {"type": "object"},
                        "tool_metadata": {"execution_mode": "auto_preinvoke"},
                        "policy": {
                            "effect": "read",
                            "risk_level": "L1",
                            "enabled": True,
                            "requires_confirmation": False,
                        },
                    }
                ],
            }
        ],
        "bindings": {
            "agent_skill": [{"agent_id": "maintenance", "skill_id": 1}],
            "agent_mcp": [
                {
                    "agent_id": "maintenance",
                    "server_name": "workorder-server",
                }
            ],
            "agent_knowledge": [
                {
                    "agent_id": "maintenance",
                    "knowledge_doc_id": 10,
                }
            ],
        },
        "model_policy": {
            "version": "v1.8",
            "default": {"model_id": "model-a"},
            "available": [{"model_id": "model-a"}],
        },
        "price_snapshots": [],
        "budget_policy": {"monthly_budget": 100},
        "retrieval_policy": {"top_k": 5},
    }


def _changed_config() -> dict:
    config = copy.deepcopy(_base_config())
    config["agents"][1].update(
        {
            "name": "维修与巡检 Agent",
            "description": "处理维修咨询和巡检",
            "instructions": "新 Prompt：先判断风险，再查询证据。",
            "model_id": "model-b",
            "enabled": False,
            "category": "specialized",
            "skill_ids": [2],
            "mcp_server_names": ["inspection-server"],
            "knowledge_doc_ids": [11],
        }
    )
    config["skills"].append(
        {
            "skill_id": 2,
            "name": "巡检判断",
            "description": "判断巡检问题",
            "version": "2.0.0",
            "enabled": True,
            "trigger_condition": "巡检",
            "content_hash": "skill-hash-2",
            "reference_snapshots": [],
            "instructions_fallback": "巡检正文",
        }
    )
    config["knowledge"].append(
        {
            "knowledge_doc_id": 11,
            "title": "巡检制度",
            "category": "巡检",
            "document_version": "doc-v2",
            "document_hash": "doc-hash-2",
            "index_status": "indexed",
            "chunk_count": 1,
            "chunk_size": 512,
            "chunk_overlap": 64,
            "split_strategy": "by_heading",
            "chunk_snapshots": [],
        }
    )
    config["mcp_servers"].append(
        {
            "server_id": 2,
            "name": "inspection-server",
            "description": "巡检查询",
            "enabled": True,
            "is_builtin": False,
            "command": "node",
            "args": ["index.js"],
            "env_keys": [],
            "tools": [
                {
                    "name": "create_inspection",
                    "description": "创建巡检记录",
                    "input_schema": {"type": "object"},
                    "tool_metadata": {"execution_mode": "proposal"},
                    "policy": {
                        "effect": "create",
                        "risk_level": "L2",
                        "enabled": True,
                        "requires_confirmation": True,
                    },
                }
            ],
        }
    )
    config["model_policy"]["default"] = {"model_id": "model-b"}
    config["retrieval_policy"]["top_k"] = 3
    config["budget_policy"]["monthly_budget"] = 80
    return config


def _validation(config: dict) -> dict:
    return {
        "valid": True,
        "errors": [],
        "warnings": [],
        "counts": {
            "agents": len(config["agents"]),
            "skills": len(config["skills"]),
            "knowledge_docs": len(config["knowledge"]),
            "mcp_servers": len(config["mcp_servers"]),
            "tool_policies": 0,
        },
    }


def _create_and_publish(config: dict, *, created_by: str) -> dict:
    version = next_runtime_release_version()
    release = create_runtime_release(
        release_id=f"rr_test_{version:04d}",
        version=version,
        config_hash=content_hash(config),
        config=config,
        validation=_validation(config),
        parent_release_id=(
            (get_current_runtime_release() or {}).get("release_id")
        ),
        created_by=created_by,
    )
    return publish_runtime_release(release["release_id"])


def _assert_no_forbidden_overview_payload(payload: dict) -> None:
    raw = json.dumps(payload, ensure_ascii=False)
    assert '"config"' not in raw
    assert "chunk_snapshots" not in raw
    assert "instructions_fallback" not in raw
    assert "input_schema" not in raw
    assert LONG_PROMPT_TAIL not in raw


def test_agent_parameter_and_binding_diff() -> None:
    lightweight = release_compiler.diff_runtime_configs(
        _base_config(),
        _changed_config(),
    )
    maintenance = next(
        item
        for item in lightweight["agents"]
        if item["agent_id"] == "maintenance"
    )
    changed_fields = {item["field"] for item in maintenance["fields"]}
    assert {
        "name",
        "description",
        "instructions",
        "model_id",
        "enabled",
        "category",
    }.issubset(changed_fields)
    assert maintenance["change_type"] == "disabled"
    assert maintenance["capabilities"]["skills"]["added"] == [
        {"skill_id": 2, "name": "巡检判断"}
    ]
    assert maintenance["capabilities"]["skills"]["removed"] == [
        {"skill_id": 1, "name": "维修判断"}
    ]
    assert maintenance["capabilities"]["knowledge"]["added"][0]["name"] == (
        "巡检制度"
    )
    assert maintenance["capabilities"]["mcp_servers"]["added"][0]["name"] == (
        "inspection-server"
    )
    assert any(
        item["name"] == "inspection-server / create_inspection"
        and item["change_type"] == "added"
        and any(
            field["field"] == "effect" and field["new"] == "create"
            for field in item["fields"]
        )
        for item in lightweight["mcp_tools"]
    )
    assert LONG_PROMPT_TAIL not in json.dumps(lightweight, ensure_ascii=False)

    detailed = release_compiler.diff_runtime_configs(
        _base_config(),
        _changed_config(),
        include_details=True,
    )
    detailed_agent = next(
        item
        for item in detailed["agents"]
        if item["agent_id"] == "maintenance"
    )
    prompt_change = next(
        item
        for item in detailed_agent["fields"]
        if item["field"] == "instructions"
    )
    assert LONG_PROMPT_TAIL in prompt_change["old"]


def test_preview_and_publish_no_changes_do_not_persist() -> None:
    baseline = _base_config()
    original_compile = release_compiler._compile_graph
    release_compiler._compile_graph = lambda: (copy.deepcopy(baseline), [])
    try:
        before = count_runtime_releases()
        preview = release_compiler.preview_runtime_release(
            created_by="s4a-test"
        )
        after_preview = count_runtime_releases()
        result = release_compiler.publish_current_runtime_config(
            created_by="s4a-test"
        )
        after_publish_attempt = count_runtime_releases()
    finally:
        release_compiler._compile_graph = original_compile
    assert preview["has_changes"] is False
    assert preview["can_publish"] is False
    assert preview["block_reason"] == "no_changes"
    assert preview["persisted"] is False
    assert result["published"] is False
    assert result["created"] is False
    assert result["reason"] == "no_changes"
    assert before == after_preview == after_publish_attempt


def test_real_change_creates_one_release_and_keeps_session_snapshot() -> None:
    old_snapshot = resolve_snapshot("s4a-old-session")
    changed = _changed_config()
    original_compile = release_compiler._compile_graph
    release_compiler._compile_graph = lambda: (copy.deepcopy(changed), [])
    try:
        before = count_runtime_releases()
        result = release_compiler.publish_current_runtime_config(
            created_by="s4a-change-test"
        )
        after = count_runtime_releases()
    finally:
        release_compiler._compile_graph = original_compile
    assert result["published"] is True
    assert result["created"] is True
    assert after == before + 1
    old_snapshot_again = resolve_snapshot("s4a-old-session")
    new_snapshot = resolve_snapshot("s4a-new-session")
    assert old_snapshot_again.release_id == old_snapshot.release_id
    assert old_snapshot_again.snapshot_hash == old_snapshot.snapshot_hash
    assert new_snapshot.release_id == result["release"]["release_id"]


def test_historical_parent_diff_and_rollback_contract() -> None:
    current = get_current_runtime_release()
    parent = runtime_api.get_runtime_release(current["parent_release_id"])
    diff = release_compiler.diff_runtime_configs(
        parent["config"],
        current["config"],
        include_details=True,
    )
    assert diff["has_changes"] is True
    assert diff["summary"]["affected_agents"] == 1

    rolled_back = publish_runtime_release(parent["release_id"])
    assert rolled_back["release_id"] == parent["release_id"]
    assert resolve_snapshot("s4a-old-session").release_id == (
        parent["release_id"]
    )
    assert resolve_snapshot("s4a-new-session").release_id == (
        current["release_id"]
    )

    duplicate_version = next_runtime_release_version()
    duplicate = create_runtime_release(
        release_id=f"rr_test_duplicate_{duplicate_version}",
        version=duplicate_version,
        config_hash=parent["config_hash"],
        config=parent["config"],
        validation=parent["validation"],
        parent_release_id=parent["release_id"],
        created_by="duplicate-test",
    )
    try:
        publish_runtime_release(duplicate["release_id"])
    except ValueError as exc:
        assert "no_changes" in str(exc)
    else:
        raise AssertionError("duplicate config hash must not be published")


def test_lightweight_overview_and_history_summary() -> None:
    changed = _changed_config()
    original_compile = release_compiler._compile_graph
    release_compiler._compile_graph = lambda: (copy.deepcopy(changed), [])
    try:
        overview = asyncio.run(runtime_api.release_overview())
    finally:
        release_compiler._compile_graph = original_compile
    _assert_no_forbidden_overview_payload(overview)
    assert overview["initial_load_contract"] == {
        "full_config_loaded": False,
        "acceptance_cases_loaded": False,
        "acceptance_runs_loaded": False,
        "history_limit": 10,
    }
    assert overview["draft"]["diff"]["agents"]
    assert len(json.dumps(overview, ensure_ascii=False).encode("utf-8")) < (
        100 * 1024
    )

    summaries = list_runtime_release_summaries(limit=10)
    assert summaries
    assert all("config" not in item for item in summaries)
    assert all("change_summary" in item for item in summaries)


def test_credentials_remain_redacted() -> None:
    redacted = runtime_api._redact_release(
        {
            "config": {
                "mcp_servers": [
                    {
                        "name": "secret-server",
                        "env": {"TOKEN": "secret-value"},
                    }
                ],
                "model_policy": {
                    "default": {
                        "model_id": "secret-model",
                        "api_key": "secret-key",
                    },
                    "available": [],
                },
            }
        }
    )
    raw = json.dumps(redacted, ensure_ascii=False)
    assert "secret-value" not in raw
    assert "secret-key" not in raw
    assert raw.count("***configured***") == 2


def test_frontend_initial_load_and_lazy_detail_contract() -> None:
    source = (REPO_ROOT / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    new_start = source.rindex("async function renderRuntimePage(container)")
    new_end = source.index("async function renderAgentsPage", new_start)
    runtime_source = source[new_start:new_end]
    assert "apiGet('/api/runtime/releases/overview')" in runtime_source
    assert "/api/runtime/releases/preview/diff?include_details=true" in (
        runtime_source
    )
    assert "/api/runtime/acceptance/cases" not in runtime_source
    assert "/api/runtime/acceptance/runs" not in runtime_source
    assert "_renderRuntimePageLegacy(container)" in runtime_source
    assert "完整配置仅在点击后加载" in runtime_source
    assert "当前没有待发布变更" in runtime_source


def main() -> None:
    baseline = _create_and_publish(_base_config(), created_by="s4a-baseline")
    assert baseline["status"] == "published"
    tests = [
        test_agent_parameter_and_binding_diff,
        test_preview_and_publish_no_changes_do_not_persist,
        test_real_change_creates_one_release_and_keeps_session_snapshot,
        test_historical_parent_diff_and_rollback_contract,
        test_lightweight_overview_and_history_summary,
        test_credentials_remain_redacted,
        test_frontend_initial_load_and_lazy_detail_contract,
    ]
    try:
        for test in tests:
            test()
            print(f"PASS {test.__name__}")
    finally:
        TEMP_DIR.cleanup()
    print("V1.8.2-S4-A release-center contracts passed.")


if __name__ == "__main__":
    main()
