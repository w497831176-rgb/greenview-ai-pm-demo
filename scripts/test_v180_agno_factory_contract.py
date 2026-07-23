"""No-model Agno factory assembly checks for the V1.8 runtime."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEMP_DIR = tempfile.TemporaryDirectory(
    prefix="yiai-v180-factory-",
    ignore_cleanup_errors=True,
)
os.environ["PROPERTY_DATA_DIR"] = TEMP_DIR.name

from agno.factory import RequestContext

from db.property_db import init_db


def test_agent_factory_assembly() -> None:
    from app.runtime.agent_factory import build_agent_from_snapshot
    from app.runtime.contracts import RunConfigSnapshot

    snapshot = RunConfigSnapshot(
        snapshot_id="snap_factory_contract",
        release_id="release_factory_contract",
        snapshot_hash="factory-contract",
        session_id="factory-agent",
        created_at="2026-07-19T00:00:00+08:00",
        config={
            "agents": [
                {
                    "agent_id": "maintenance",
                    "name": "维修 Agent",
                    "enabled": True,
                    "category": "maintenance",
                    "instructions": "处理维修咨询。",
                    "skill_ids": [8],
                    "mcp_server_names": [],
                    "knowledge_doc_ids": [],
                }
            ],
            "skills": [
                {
                    "skill_id": 8,
                    "name": "维修工单处理",
                    "description": "维修咨询方法",
                    "version": "1.0.0",
                    "enabled": True,
                    "trigger_condition": "漏水,维修",
                    "metadata": {"positive_triggers": ["漏水", "维修"]},
                    "content_hash": "skill-eight",
                    "reference_snapshots": [],
                    "instructions_fallback": "先核实位置和风险。",
                }
            ],
        },
    )
    build = build_agent_from_snapshot(
        snapshot,
        "maintenance",
        "只咨询漏水维修服务承诺，不创建工单。",
    )
    assert build.agent.id == "maintenance"
    assert build.agent.model is not None
    assert build.agent.skills is not None
    assert [item.skill_id for item in build.activated_skills] == [8]
    assert build.skill_tool_calls == [
        {
            "tool_name": "get_skill_instructions",
            "arguments": {"skill_name": "skill-8"},
            "status": "success",
            "invocation_mode": "policy_preinvoke",
            "skill_id": 8,
            "skill_version": "1.0.0",
            "skill_content_hash": "skill-eight",
        }
    ]
    assert "先核实位置和风险。" in "\n".join(build.agent.instructions)


def test_composite_workflow_assembly() -> None:
    from app.runtime.workflow_factory import build_runtime_workflow

    workflow = build_runtime_workflow(
        RequestContext(
            session_id="factory-workflow",
            user_id="contract",
            input={
                "path": "composite_acceptance",
                "message": "组合验收",
                "consultation_message": "儿童滑梯安全制度是什么？",
                "action_message": "3号楼漏水，请创建工单",
            },
        )
    )
    assert [step.name for step in workflow.steps] == [
        "resolve_snapshot",
        "execute_consultation",
        "collect_action_proposal",
        "commit_confirmed_action",
    ]
    commit_step = workflow.steps[-1]
    assert commit_step.requires_confirmation is True


def main() -> None:
    try:
        init_db()
        from app.runtime.release_compiler import ensure_bootstrap_release

        ensure_bootstrap_release()
        test_agent_factory_assembly()
        print("PASS test_agent_factory_assembly")
        test_composite_workflow_assembly()
        print("PASS test_composite_workflow_assembly")
    finally:
        TEMP_DIR.cleanup()
    print("V1.8 Agno factory no-model contracts passed.")


if __name__ == "__main__":
    main()
